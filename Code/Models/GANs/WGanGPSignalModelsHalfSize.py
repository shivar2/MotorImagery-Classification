
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, time_sample=1000, noise=100, channels=3,):
        super(Generator, self).__init__()

        self.noise = noise
        self.time_sample = time_sample
        self.channels = channels
        self.eeg_shape = (self.channels, self.time_sample)
        # img_shape = (opt.channels, opt.img_size, opt.img_size)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.noise, 62, normalize=False),
            *block(62, 125),
            *block(125, 250),
            *block(250, 500),
            nn.Linear(500, int(np.prod(self.eeg_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], * self.eeg_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, time_sample=1000, channels=3):

        super(Discriminator, self).__init__()

        self.time_sample = time_sample
        self.channels = channels
        self.eeg_shape = (self.channels, self.time_sample)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.eeg_shape)), 250),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(250, 125),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(125, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
