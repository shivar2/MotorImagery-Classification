import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, time_sample=1000, noise=100, channels=3, freq_sample=34):
        super(Generator, self).__init__()

        self.sfreq = 250
        self.time_sample = time_sample
        self.channels = channels
        self.freq_sample = freq_sample
        self.eeg_shape = (self.freq_sample, self.time_sample, self.channels)

        self.init_size = (self.sfreq * self.time_sample) * 4

        self.noise = noise

        # self.l1 = nn.Sequential(nn.Linear(self.channels, self.time_sample*128))
        # self.l1 = nn.Sequential(nn.Linear(self.noise, self.init_size * self.channels))
        self.l1 = nn.Sequential(nn.Linear(self.noise, self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.sfreq),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(self.sfreq, self.sfreq // 2, (3, 3), stride=1, padding=1),
            # nn.Conv2d(in_channels=1, out_channels=1,kernel_size=1,stride=1, padding=1),
            nn.BatchNorm2d(self.sfreq // 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(self.sfreq // 2, self.sfreq // 4, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.sfreq // 4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.sfreq // 4, self.channels, (3, 3), stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        # out =  out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = out.view(-1, self.sfreq, self.time_sample, 4)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, time_sample=1000, channels=3, freq_sample=34):
        super(Discriminator, self).__init__()
        self.sfreq = 256
        self.time_sample = time_sample
        self.channels = channels
        self.freq_sample = freq_sample

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, self.channels, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        if (self.channels < self.sfreq // 8):

            self.model = nn.Sequential(
                *discriminator_block(self.channels, self.sfreq // 8, bn=False),
                *discriminator_block(self.sfreq // 8,  self.sfreq // 4),
                *discriminator_block(self.sfreq // 4,  self.sfreq // 2),
                *discriminator_block(self.sfreq // 2,  self.sfreq),
            )
        else:
            self.model = nn.Sequential(
                *discriminator_block(self.channels, self.sfreq // 4, bn=False),
                *discriminator_block(self.sfreq // 4, self.sfreq // 2),
                *discriminator_block(self.sfreq // 2, self.sfreq),
            )

        # The height and width of downsampled image
        ds_size = self.time_sample // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(self.sfreq * self.time_sample * self.channels, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity