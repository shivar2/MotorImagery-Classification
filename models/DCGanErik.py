import os
import numpy as np

import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.nn as nn
import torch

from models.DCModels import Generator, Discriminator


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class DCGAN(nn.Module):
    def __init__(self, subject=1, n_epochs=10, batch_size=64, time_sample=32, channels=3, sample_interval=400,
                 freq_sample=34):

        super(DCGAN, self).__init__()

        self.subject = subject
        self.n_epochs = n_epochs                            # number of epochs of training
        self.batch_size = batch_size                        # size of the batches
        self.lr = 0.0002                                    # adam: learning rate
        self.b1 = 0.5                                       # adam: decay of first order momentum of gradient
        self.b2 = 0.999                                     # adam: decay of first order momentum of gradient
        self.n_cpu = 8                                      # number of cpu threads to use during batch generation
        self.noise = 100                                    # dimensionality of the latent space
        self.time_sample = time_sample                      # size of each image dimension
        self.channels = channels                            # number of image channels
        self.sample_interval = sample_interval              # Stride between windows, in samples
        self.dir = 'EEG_samples'

        self.freq_sample = freq_sample

        self.cuda = True if torch.cuda.is_available() else False


        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()

        # Initialize generator and discriminator
        self.generator = Generator(time_sample=self.time_sample, channels=self.channels, freq_sample=freq_sample)
        self.discriminator = Discriminator(time_sample=self.time_sample, channels=self.channels, freq_sample=freq_sample)

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()

        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

    def train(self, dataset):

        gen_loss, disc_loss = [], []
        g_tot, d_tot = [], []

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # ----------
        #  Training
        # ----------
        data_batches = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.n_epochs):
            for i, signal_batch in enumerate(data_batches):

                    # Adversarial ground truths
                    valid = Variable(Tensor(signal_batch.shape[0], 1).fill_(1.0), requires_grad=False)
                    fake = Variable(Tensor(signal_batch.shape[0], 1).fill_(0.0), requires_grad=False)

                    # Configure input
                    real_imgs = Variable(signal_batch.type(Tensor))

                    # -----------------
                    #  Train Generator
                    # -----------------

                    self.optimizer_G.zero_grad()

                    # Sample noise as generator input
                    z = Variable(Tensor(np.random.normal(0, 1, (signal_batch.shape[0], self.noise))))

                    # Generate a batch of images
                    gen_imgs = self.generator(z)

                    # Loss measures generator's ability to fool the discriminator
                    g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)

                    g_loss.backward()
                    self.optimizer_G.step()

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    self.optimizer_D.zero_grad()

                    # Measure discriminator's ability to classify real from generated samples
                    real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                    fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2

                    d_loss.backward()
                    self.optimizer_D.step()

                    gen_loss.append(fake_loss)
                    disc_loss.append(real_loss)

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, self.n_epochs, i, len(data_batches), d_loss.item(), g_loss.item())
                    )
                    batches_done = epoch * len(data_batches) + i
                    if batches_done % self.sample_interval == 0:

                        # ---------------------
                        #  PLOT
                        # ---------------------

                        # from https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/WasserGAN_Final.py

                        # Plots the generated samples for the selected channels.
                        # Recall the channels are chosen during the Load_and_Preprocess Script

                        # Here they just correspond to C3 only (channel 7 was selected).
                        channel = 0

                        fig, axs = plt.subplots(1, 2)
                        fig.suptitle('Comparison of Generated vs. Real Signal (Spectrogram) for one trial, one channel')
                        fig.tight_layout()

                        gen_imgs = Variable(gen_imgs, requires_grad=True)
                        gen_imgs = gen_imgs.detach().numpy()

                        axs[0].imshow(gen_imgs[0, :, :, channel], aspect='auto')
                        axs[0].set_title('Generated Signal', size=10)
                        axs[0].set_xlabel('Time Sample')
                        axs[0].set_ylabel('Frequency Sample')

                        real_imgs = Variable(real_imgs, requires_grad=True)
                        real_imgs = real_imgs.detach().numpy()

                        axs[1].imshow(real_imgs[0, :, :, channel], aspect='auto')
                        axs[1].set_title('Fake Signal', size=10)
                        axs[1].set_xlabel('Time Sample')
                        axs[1].set_ylabel('Frequency Sample')
                        plt.show()

                        # Save the generated samples within the current working dir
                        # in a folder called 'EEG Samples', every 100 epochs.
                        if not os.path.exists(self.dir):
                            os.makedirs(self.dir)

                        plt.savefig("%s/%d.png" % (self.dir, epoch))
                        plt.close()

            # from https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/WasserGAN_Final.py
            g_loss = sum(gen_loss) / len(gen_loss)
            d_loss = sum(disc_loss) / len(disc_loss)

            g_tot.append(g_loss)
            d_tot.append(d_loss)

        # ---------------------
        #  PLOT
        # ---------------------
        # Plot the generator and discriminator losses for all the epochs
        plt.figure()
        plt.plot(g_tot, 'r')
        plt.plot(d_tot, 'b')
        plt.title('Loss history')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Generator', 'Discriminator'])
        plt.grid()
        plt.show()

        # Save subject and task data such that it can be used to generate
        # Fake samples later
        fp = os.path.join(os.getcwd(), 'EEG_Samples')
        sp = os.path.join(fp, 'Subject{}WGAN_Model_Data_For_Task{}.h5'.format(self.subject, self.task))
        self.generator.save(sp)
