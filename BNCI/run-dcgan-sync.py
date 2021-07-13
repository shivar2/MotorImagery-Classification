from __future__ import print_function
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import os

from models import DCGan as dcgan


from braindecode.datautil.windowers import create_windows_from_events, create_fixed_length_windows

from braindecode.datautil.serialization import load_concat_dataset

if __name__=="__main__":

    # config run
    workers = 2
    batchSize = 64
    imageSize = 1024          # the height / width of the input image to network
    nc = 22                  # input image channels
    nz = 100                # size of the latent z vector
    ngf = 64
    ndf = 64
    niter = 25              # number of epochs to training for
    lrD = 0.00005           # learning rate for Critic
    lrG = 0.00005           # learning rate for Generator
    beta1 = 0.5             # beta1 for adam.
    cuda = False            # enables cuda
    ngpu = 1                # number of GPUs to use
    netG = ''               # path to netG (to continue training)
    netD = ''               # path to netD (to continue training)
    clamp_lower = -0.01
    clamp_upper = 0.01
    Diters = 5,             # number of D iters per each G iter
    noBN = True             # use batchnorm or not (only for DCGAN
    mlp_G = False           # use MLP for G
    mlp_D = False           # use MLP for D
    n_extra_layers = 0      # Number of extra layers on gen and disc
    experiment = None,       # Where to store samples and models
    adam = True             # Whether to use adam (default is rmsprop

    if experiment is None:
        experiment = 'samples'
    os.system('mkdir {0}'.format(experiment))

    manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # dataset
    dataset = load_concat_dataset(
                path='../Dataset-Files/data-file/bnci-raw/1',
                preload=False,
                target_name=None,

            )
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(-0.5 * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    dataloader = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=False,
        window_size_samples=imageSize,
        window_stride_samples=1,
        drop_bad_windows=True,

    )

    # for x, y, window_ind in windows_dataset:
    #     print(x.shape, y, window_ind)

    nc = 22
    # imageSize = (6, 3072)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    # model  G
    netG = dcgan.DCGAN_G(imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    netG.apply(weights_init)

    # model  D
    netD = dcgan.DCGAN_D(imageSize, nz, nc, ndf, ngpu, n_extra_layers)
    netD.apply(weights_init)

    # create input arry
    input = torch.FloatTensor(batchSize, nc, imageSize, imageSize)
    # create noise
    noise = torch.FloatTensor(batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)

    one = torch.FloatTensor([1])
    mone = one * -1

    # setup optimizer
    if adam:
        optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr = lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr = lrG)

    gen_iterations = 0
    for epoch in range(niter):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # training the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(clamp_lower, clamp_upper)

                # dataloader.__getitem__(i)
                data = next(iter(dataloader))
                i += 1

                # training with real
                # real_cpu, _ = data
                # netD.zero_grad()
                # batch_size = real_cpu.size(0)

                # if cuda:
                #     real_cpu = real_cpu.cuda()
                # input.resize_as_(real_cpu).copy_(real_cpu)
                # inputv = Variable(input)

                # loss_D_real
                errD_real = netD(input)
                errD_real.backward(one)

                # training with fake
                noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile = True) # totally freeze netG
                fake = Variable(netG(noisev).data)
                inputv = fake

                # Loss_D_fake
                errD_fake = netD(inputv)
                errD_fake.backward(mone)

                # loss_d
                errD = errD_real - errD_fake
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)

            # Loss_G
            errG = netD(fake)
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, niter, i, len(dataloader), gen_iterations,
                errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            if gen_iterations % 500 == 0:
                real_cpu = real_cpu.mul(0.5).add(0.5)
                vutils.save_image(real_cpu, '{0}/real_samples.png'.format(experiment))
                fake = netG(Variable(fixed_noise, volatile=True))
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(experiment, gen_iterations))

        # do checkpointing
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(experiment, epoch))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(experiment, epoch))
