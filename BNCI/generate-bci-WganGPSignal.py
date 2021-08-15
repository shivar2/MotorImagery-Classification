
import os
import numpy as np
import mne
import matplotlib.pyplot as plt

import torch.utils.data
from torch.autograd import Variable

from braindecode.datautil import create_from_X_y

from models.WGanGPSignalModels import Generator

# for macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


subject_id = 1

# number of images to generate
batch_size = 24
nimages = 128

# gan info
sfreq = 250
time_sample = 500
noise = 100

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


tasks = ['feet', 'left_hand', 'right_hand', 'tongue']
task_dict = {
    # Select just 'feet' task
    'feet': 0,
    'left_hand': 1,
    'right_hand': 2,
    'tongue': 3
}

all_channels = ['Fz',
                'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                'P1', 'Pz', 'P2',
                'POz']


for task in tasks:
    task_channels_X = np.empty(shape=(batch_size, 0, time_sample))

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
        netG.load_state_dict(torch.load(saved_models_path, map_location=torch.device('cpu')))

        if cuda:
            netG.cuda()

        # initialize noise
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, noise))))

        gen_sig = netG(z)

        task_channels_X = np.append(task_channels_X, gen_sig.detach().cpu().numpy(), axis=1)

    # task_channels_y = np.ones(shape=(batch_size, time_sample)) * target

    # ---------------------
    #  Merge channels
    # ---------------------

    # ---------------------
    #  MNE Create Raw
    # ---------------------

    # Creating Info objects
    ch_types = ['eeg'] * 22
    info = mne.create_info(all_channels, ch_types=ch_types, sfreq=sfreq)
    info.set_montage('standard_1020')
    info['description'] = 'My custom dataset'

    # Creating Info objects
    # data = np.array(task_channels_X[0])
    #
    # simulated_raw = mne.io.RawArray(data, info)
    # simulated_raw.plot(show_scrollbars=False, show_scalebars=False)

    target = task_dict[task]
    events = np.column_stack((np.arange(0, sfreq * batch_size, sfreq),
                              np.ones(batch_size, dtype=int) * 1000,
                              np.ones(shape=(batch_size), dtype=int) * target))

    event_dict = {task: target}

    epoch_data = np.array(task_channels_X)
    mne.EpochsArray(epoch_data, info)
    simulated_epochs = mne.EpochsArray(epoch_data, info, tmin=-0.5, events=events,
                                       event_id=event_dict)
    simulated_epochs.plot(show_scrollbars=False, events=events,
                          event_id=event_dict)

    print("end")
    # windows_dataset = create_from_X_y(
    #     X=task_channels_X,
    #     y=task_channels_y,
    #     drop_last_window=False,
    #     sfreq=sfreq,
    #     ch_names=all_channels,
    #     window_stride_samples=500,
    #     window_size_samples=500,
    # )
    # print(windows_dataset)
