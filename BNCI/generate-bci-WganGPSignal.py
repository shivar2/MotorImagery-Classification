
import os
import numpy as np
import matplotlib.pyplot as plt

import torch.utils.data
from torch.autograd import Variable

from models.WGanGPSignalModels import Generator

# for macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


subject_id = 1

# number of images to generate
batch_size = 64
nimages = 128

# gan info
time_sample = 500
noise = 100

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


tasks = ['feet', 'left_hand', 'right_hand', 'tongue']


all_channels = ['Fz',
                'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                'P1', 'Pz', 'P2',
                'POz']

for task in tasks:
    for channel in all_channels:
        # path to generator weights .pth file
        saved_models_path = '../saved_models/WGan-GP-Signal/' + str(subject_id) + '/' + task + '/'
        saved_models_path += channel + 'generator_state_dict.pth'

        # path to to fake eeg directory
        fake_data_path = '../Dataset-Files/fake-data/WGan-GP-Signal/' + str(subject_id) + '/' + task + '/'
        if not os.path.exists(fake_data_path):
            os.makedirs(fake_data_path)

        netG = Generator(time_sample=time_sample, noise=noise, channels=1)

        # load weights -tl
        # netG = torch.load(saved_models_path)
        netG.load_state_dict(torch.load(saved_models_path, map_location=torch.device('cpu')))

        if cuda:
            netG.cuda()

        # initialize noise
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, noise))))

        fake_eeg = netG(z)
        # fake.data = fake.data.mul(0.5).add(0.5)         # why?

        # ---------------------
        #  PLOT
        # ---------------------

        # from https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/WasserGAN_Final.py

        # Plots the generated samples for the selected channels.
        # Recall the channels are chosen during the Load_and_Preprocess Script

        # Here they correspond channel.
        fig, axs = plt.subplots(1, 1)
        fig.suptitle('Fake signal for subject ' + str(subject_id))
        fig.tight_layout()

        fake_eeg = Variable(fake_eeg, requires_grad=True)
        fake_eeg = fake_eeg.detach().cpu().numpy()

        axs.imshow(fake_eeg[0, :, :], aspect='auto')
        axs.set_title('Generated Signal', size=10)
        axs.set_xlabel('Time Sample')
        axs.set_ylabel('Frequency Sample')

        # Save the generated samples within the current working dir
        # in a folder called 'EEG Samples', every 100 epochs.

        plt.show()
        plt.close()

        sp = fake_data_path + 'Subject {} _WGANSignal_Model_Data_For_Task {} _Channel {}.pt'.format(subject_id, task, channel)
        # netG.save(sp)
        torch.save(fake_eeg, sp)
