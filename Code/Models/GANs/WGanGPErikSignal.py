
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.autograd as autograd
import torch

from Code.Models.GANs import WGanGPSignalModels, WGanGPSignalModelsHalfSize


class WGANGP(nn.Module):
    def __init__(self, subject=1, n_epochs=10, batch_size=64, time_sample=32, channels=3, window_size=1000, sample_interval=400):

        super(WGANGP, self).__init__()

        self.subject = subject
        self.n_epochs = n_epochs                            # number of epochs of training
        self.batch_size = batch_size                        # size of the batches
        self.lr = 0.001                                    # adam: learning rate
        self.b1 = 0.5                                      # adam: decay of first order momentum of gradient
        self.b2 = 0.999                                     # adam: decay of first order momentum of gradient
        self.n_cpu = 8                                      # number of cpu threads to use during batch generation
        self.noise = 100                                    # dimensionality of the latent space
        self.time_sample = time_sample                      # size of each image dimension
        self.channels = channels                            # number of image channels

        self.n_critic = 5                                   # number of training steps for discriminator per iter
        self.clip_value = 0.01                              # lower and upper clip value for disc. weights
        self.lambda_gp = 10                                 # Loss weight for gradient penalty

        self.sample_interval = sample_interval              # Stride between windows, in samples

        self.cuda = True if torch.cuda.is_available() else False

        self.window_size = window_size

        # Initialize generator and discriminator
        if self.window_size == 1000:
            self.generator = WGanGPSignalModels.Generator(time_sample=self.time_sample, channels=self.channels)
            self.discriminator = WGanGPSignalModels.Discriminator(time_sample=self.time_sample, channels=self.channels)
        else:
            self.generator = WGanGPSignalModelsHalfSize.Generator(time_sample=self.time_sample, channels=self.channels)
            self.discriminator = WGanGPSignalModelsHalfSize.Discriminator(time_sample=self.time_sample, channels=self.channels)

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, dataset, save_model_path, gen_loss, disc_loss, last_epoch=0):

        batches_done = 0
        d_tot, g_tot = [], []
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # ----------
        #  Training
        # ----------
        data_batches = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(last_epoch, last_epoch+self.n_epochs):
            for i, signal_batch in enumerate(data_batches):

                # Configure input
                real_imgs = Variable(signal_batch.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (signal_batch.shape[0], self.noise))))

                # Generate a batch of images
                fake_imgs = self.generator(z)

                # Real images
                real_validity = self.discriminator(real_imgs)
                # Fake images
                fake_validity = self.discriminator(fake_imgs)
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(self.discriminator, real_imgs.data, fake_imgs.data)
                # Adversarial loss
                d_loss_batch = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

                d_loss_batch.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % self.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_imgs = self.generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(fake_imgs)
                    g_loss_batch = -torch.mean(fake_validity)

                    g_loss_batch.backward()
                    self.optimizer_G.step()

                    gen_loss.append(g_loss_batch)
                    disc_loss.append(d_loss_batch)

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, self.n_epochs, i, len(data_batches), d_loss_batch.item(), g_loss_batch.item())
                    )

                    batches_done += self.n_critic

            # from https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/WasserGAN_Final.py
            g_loss = sum(gen_loss) / len(gen_loss)
            d_loss = sum(disc_loss) / len(disc_loss)

            g_tot.append(g_loss)
            d_tot.append(d_loss)

        # ---------------------
        #  Save Generator
        # ---------------------
        torch.save({
            'epoch': self.n_epochs,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer_G.state_dict(),
            'loss': gen_loss,
            }, save_model_path + 'generator_state_dict.pth')

        # ---------------------
        #  Save Generator
        # ---------------------
        torch.save({
            'epoch': self.n_epochs,
            'model_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer_D.state_dict(),
            'loss': disc_loss,
            }, save_model_path + 'discriminator_state_dict.pth')

        g_tot = torch.FloatTensor(g_tot)
        d_tot = torch.FloatTensor(d_tot)

        return d_tot, g_tot
