import os
import numpy as np

from torchvision.utils import save_image

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
    def __init__(self, img_size=32, channels=3):

        super(DCGAN, self).__init__()

        self.n_epochs = 10              # number of epochs of training
        self.batch_size = 64            # size of the batches
        self.lr = 0.0002                # adam: learning rate
        self.b1 = 0.5                   # adam: decay of first order momentum of gradient
        self.b2 = 0.999                 # adam: decay of first order momentum of gradient
        self.n_cpu = 8                  # number of cpu threads to use during batch generation
        self.latent_dim = 100           # dimensionality of the latent space
        self.img_size = img_size        # size of each image dimension
        self.channels = channels        # number of image channels
        self.sample_interval = 400      # interval between image sampling

        self.cuda = True if torch.cuda.is_available() else False

        os.makedirs("images", exist_ok=True)

        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()

        # Initialize generator and discriminator
        self.generator = Generator(img_size=self.img_size, latent_dim=self.latent_dim, channels=self.channels)
        self.discriminator = Discriminator(img_size=self.img_size, channels=self.img_size)

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

    def train(self, data):

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # ----------
        #  Training
        # ----------

        for epoch in range(self.n_epochs):
            for i, signal_batch in enumerate(data):

                # Adversarial ground truths
                valid = Variable(Tensor(signal_batch.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(signal_batch.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(torch.tensor(signal_batch))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (signal_batch.shape[0], self.latent_dim))))

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

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.n_epochs, i, len(data), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(data) + i
                if batches_done % self.sample_interval == 0:
                    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)