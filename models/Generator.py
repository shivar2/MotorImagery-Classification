import os

import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.serialization import load_concat_dataset

from DCGan import DCGAN_G, DCGAN_D


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_data(imageSize):

    # Dataset
    dataset = load_concat_dataset(
        path='../Dataset-Files/data-file/bnci-raw/1',
        preload=True,
        target_name=None,

    )
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(-0.5 * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.

    data = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
        window_size_samples=imageSize,
        window_stride_samples=1,
        drop_bad_windows=True,
    )

    return data


class Generator(nn.Module):

    def __init__(self, nc=22):
        super(Generator, self).__init__()

        # Training parm
        self.batchSize = 64
        self.nz = 100                           # size of the latent z vector
        self.ngf = 64
        self.ndf = 64
        self.niter = 25                         # number of epochs to training for
        self.gen_iterations = 0
        self.clamp_lower = -0.01
        self.clamp_upper = 0.01

        # Dataset features:
        self.nc = nc                            # input image channels
        self.imageSize = 1024                   # the height / width of the input image to network

        # Model specific parameters (Noise generation, Dropout for overfitting reduction, etc...):
        self.netG = ''                          # path to netG (to continue training)
        self.netD = ''                          # path to netD (to continue training)
        self.Diters = 5,                        # number of D iters per each G iter

        self.n_extra_layers = 0      # Number of extra layers on gen and disc

        # GPU
        self.ngpu = 1                # number of GPUs to use

        # Store model
        self.experiment = None,                 # Where to store samples and models
        if self.experiment is None:
            self.experiment = 'samples'
        os.system('mkdir {0}'.format(self.experiment))

        # create input arry
        self.input = torch.FloatTensor(self.batchSize, self.nc, self.imageSize, self.imageSize)
        # create noise
        self.noise = torch.FloatTensor(self.batchSize, self.nz, 1, 1)
        self.fixed_noise = torch.FloatTensor(self.batchSize, self.nz, 1, 1).normal_(0, 1)

        # For backward
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1

        # model  G
        self.netG = DCGAN_G(self.imageSize,
                            self.nz,
                            self.nc,
                            self.ngf,
                            self.ngpu,
                            self.n_extra_layers)

        self.netG.apply(weights_init)

        # model  D
        self.netD = DCGAN_D(self.imageSize,
                            self.nz, self.nc,
                            self.ngf,
                            self.ngpu,
                            self.n_extra_layers)

        self.netD.apply(weights_init)

        # Choosing Adam optimiser for both generator and discriminator to feed in to the model:
        self.lrD = 0.00005                      # learning rate for Critic
        self.lrG = 0.00005                      # learning rate for Generator
        self.beta1 = 0.5                        # beta1 for adam.

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lrD, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lrG, betas=(self.beta1, 0.999))

    def make_fakedata(self):

        noise = self.noise

        noise.resize_(self.batchSize, self.nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise, volatile=True)  # totally freeze netG
        fake = Variable(self.netG(noisev).data)

        return fake

    def discriminator_loss(self, real_output, fake_output):

        # loss_D
        errD = real_output - fake_output

        return errD

    def generator_loss(self):

        # Loss_G
        errG = self.netD(self.fake)
        errG.backward(self.one)
        self.optimizerG.step()

        return errG

    def train_step(self, signal):
        '''
        This training step function that follows from the official TensorFlow documentation.
        It is in the form of tf.function which allows it to be compiled, rather than
        compiling the combined models alone everytime. More specificially, it makes use
        of GradientTape() function to train both generator and discriminator separately.
        :return: Discriminator and Generator loss
        '''

        i = 0
        # (1) Update D network
        errD = 0
        for p in self.netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        # training the discriminator Diters times
        if self.gen_iterations < 25 or self.gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = self.Diters
        j = 0
        while j < Diters and i < len(signal):
            j += 1
            # clamp parameters to a cube
            for p in self.netD.parameters():
                p.data.clamp_(self.clamp_lower, self.clamp_upper)

            # dataloader.__getitem__(i)
            # data = next(iter(signal))
            i += 1
            # Loss_D_real
            real_output = self.netD(self.input)
            real_output.backward(self.one)

            # Training with fake
            inputv = self.make_fakedata()

            # Loss_D_fake
            fake_output = self.netD(inputv)
            fake_output.backward(self.mone)

            errD = self.discriminator_loss(real_output, fake_output)
            self.optimizerD.step()

        # (2) Update G network
        for p in self.netD.parameters():
            p.requires_grad = False  # to avoid computation

        self.netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        self.noise.resize_(self.batchSize, self.nz, 1, 1).normal_(0, 1)
        noisev = Variable(self.noise)
        self.fake = self.netG(noisev)
        errG = self.generator_loss()
        self.gen_iterations += 1

        return errD, errG

    def train(self, data):

        '''
        The training function that has a loop which trains the model on
        every epoch/iteration. Calls the train_step() compiled function
        which trains the combined model at the same time.
        '''

        gen_loss, disc_loss = [], []
        g_tot, d_tot = [], []

        # Seed
        manualSeed = random.randint(1, 10000)               # fix seed
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        for epoch in range(self.niter):
            for i, signal_batch in enumerate(data):
                disc_loss_batch, gen_loss_batch = self.train_step(signal_batch)

                gen_loss.append(gen_loss_batch)
                disc_loss.append(disc_loss_batch)

                print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                      % (epoch, self.niter, i, len(data), self.gen_iterations,
                         self.errD.data[0], self.errG.data[0], self.errD_real.data[0], self.errD_fake.data[0]))
                if self.gen_iterations % 500 == 0:
                    # real_cpu = real_cpu.mul(0.5).add(0.5)
                    # vutils.save_image(real_cpu, '{0}/real_samples.png'.format(self.experiment))
                    fake = self.netG(Variable(self.fixed_noise, volatile=True))
                    fake.data = fake.data.mul(0.5).add(0.5)
                    vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(self.experiment, self.gen_iterations))

            g_loss = sum(gen_loss)/len(gen_loss)
            d_loss = sum(disc_loss)/len(disc_loss)

            g_tot.append(g_loss)
            d_tot.append(d_loss)

            # do checkpointing
            torch.save(self.netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(self.experiment, epoch))
            torch.save(self.netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(self.experiment, epoch))

        #     if epoch % sample_interval == 0:
        #         print("epoch: {}, generator loss: {}, discriminator loss: {}".format
        #             (epoch, g_loss, d_loss))
        #
        #         # Allows us to generate the signal and get the fake one for a
        #         # Arbitrary trial number. Plots it and save it every sample_interval
        #         # Which is 100 in this case.
        #         generated_signal, _ = self.make_fakedata(noise_shape=100)
        #         trial_num, channel = 30, 0
        #         real_signal = np.expand_dims(dataset[trial_num], axis=0)
        #
        #         # Plots the generated samples for the selected channels.
        #         # Recall the channels are chosen during the Load_and_Preprocess Script
        #         # Here they just correspond to C3 only (channel 7 was selected).
        #         fig, axs = plt.subplots(1, 2)
        #         fig.suptitle('Comparison of Generated vs. Real Signal (Spectrogram) for one trial, one channel')
        #         fig.tight_layout()
        #         axs[0].imshow(generated_signal[0, :, :, channel], aspect='auto')
        #         axs[0].set_title('Generated Signal', size=10)
        #         axs[0].set_xlabel('Time Sample')
        #         axs[0].set_ylabel('Frequency Sample')
        #         axs[1].imshow(real_signal[0, :, :, channel], aspect='auto')
        #         axs[1].set_title('Fake Signal', size=10)
        #         axs[1].set_xlabel('Time Sample')
        #         axs[1].set_ylabel('Frequency Sample')
        #         plt.show()
        #
        #         # Save the generated samples within the current working dir
        #         # in a folder called 'EEG Samples', every 100 epochs.
        #         if not os.path.exists(self.dir):
        #             os.makedirs(self.dir)
        #
        #         plt.savefig("%s/%d.png" % (self.dir, epoch))
        #         plt.close()
        #
        # # Plot the generator and discriminator losses for all the epochs
        # plt.figure()
        # plt.plot(g_tot, 'r')
        # plt.plot(d_tot, 'b')
        # plt.title('Loss history')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend(['Generator', 'Discriminator'])
        # plt.grid()
        # plt.show()

dataset = get_data(imageSize=1024)

net = Generator(nc=22)
net.train(dataset)
