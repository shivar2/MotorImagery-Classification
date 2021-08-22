
import os
import random
import numpy as np
import pandas as pd
import mne

import torch.utils.data
from torch.autograd import Variable

from braindecode.datasets.base import WindowsDataset, BaseConcatDataset

from Code.Models.GANs.WGanGPSignalModels import Generator

# for macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


subject_id = 1

# number of images to generate
batch_size = 36
nimages = 128

# GAN info
sfreq = 250
time_sample = 1000
noise = 100

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


task_dict = {
    # Select just 'feet' task
    'feet': 0,
    'left_hand': 1,
    'right_hand': 2,
    'tongue': 3
}
tasks = ['feet', 'left_hand', 'right_hand', 'tongue']

all_channels = ['Fz',
                'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                'P1', 'Pz', 'P2',
                'POz']

start = 0
task_trials_epoch = []
for task in tasks:
    task_channels_trials = np.empty(shape=(batch_size, 0, time_sample))

    for channel in all_channels:
        # path to generator weights .pth file
        saved_models_path = '../../Model_Params/GANs/WGan-GP-Signal/' + str(subject_id) + '/' + task + '/' + channel + '/'
        saved_models_path += 'generator_state_dict.pth'

        netG = Generator(time_sample=time_sample, noise=noise, channels=1)

        # load weights -tl
        netG.load_state_dict(torch.load(saved_models_path, map_location=torch.device('cpu')))

        if cuda:
            netG.cuda()

        # initialize noise
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, noise))))

        gen_sig = netG(z)

        task_channels_trials = np.append(task_channels_trials, gen_sig.detach().cpu().numpy(), axis=1)

    # ---------------------
    #  Merge channels
    #  MNE Create Task's Trial - Epoch
    # ---------------------

    # Creating Info objects
    ch_types = ['eeg'] * len(all_channels)
    info = mne.create_info(all_channels, ch_types=ch_types, sfreq=sfreq)
    info.set_montage('standard_1020')
    info['description'] = 'My custom dataset'

    target = task_dict[task]
    event_dict = {task: target}

    # Creating Epoch objects
    for task_channels_trial in task_channels_trials:

        events = np.column_stack((np.arange(start, start + int(sfreq/2), sfreq),
                                  np.ones(1, dtype=int) * 1000,
                                  target))

        metadata = pd.DataFrame({
            'i_window_in_trial': np.arange(len(events)),
            'i_start_in_trial': start + events[0][2],
            'i_stop_in_trial': start + events[0][0] + time_sample,
            'target': len(events) * [target]
        })

        epoch_data = np.array(task_channels_trial)
        epoch_data = np.reshape(epoch_data, newshape=(-1, len(all_channels), time_sample))

        simulated_epoch = mne.EpochsArray(epoch_data, info, tmin=-0.5, events=events, event_id=event_dict, metadata=metadata)
        # simulated_epoch.plot(show_scrollbars=False, events=events, event_id=event_dict)
        task_trials_epoch.append(simulated_epoch)
        start += sfreq

# ---------------------
#  Merge Task
#  Shuffle Tasks
#  Create Fake Dataset - WindowsDataset
# ---------------------

random.shuffle(task_trials_epoch)
session = mne.concatenate_epochs(task_trials_epoch)

# Save fake dataset as BaseConcatDataset obj
wdataset = WindowsDataset(session)
fake_dataset = BaseConcatDataset([wdataset])


# path to to fake eeg directory
fake_data_path = '../../Data/Fake_Data/WGan-GP-Signal/' + str(subject_id) + '/' + 'Runs' + '/' + '3/'
if not os.path.exists(fake_data_path):
    os.makedirs(fake_data_path)

fake_dataset.save(path=fake_data_path, overwrite=True)


