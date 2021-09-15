
import os
import random
import numpy as np
import pandas as pd
import mne

import torch.utils.data
from torch.autograd import Variable

from braindecode.datasets.base import WindowsDataset, BaseConcatDataset
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.serialization import load_concat_dataset

from Code.Models.GANs.WGanGPSignalModels import Generator

# for macOS
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def calculate_mean_std(train_set):

    merged_window = np.concatenate(train_set.datasets)[:, 0]
    mean = np.mean(merged_window)
    sigma = np.std(merged_window)

    return mean, sigma


def unNormalizeTanh(normal_data, mean, sigma):
    data = (100 * np.arctanh(2 * normal_data) * sigma) + mean
    return data


def get_data_mean_sigma(dataset,
                        time_sample=1000,
                        window_stride_samples=1,
                        mapping=None,
                        pick_channels=None):

    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    trial_start_offset_samples = int(-0.5 * sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
        window_size_samples=time_sample,
        window_stride_samples=window_stride_samples,
        drop_bad_windows=True,
        picks=pick_channels,
        mapping=mapping,
    )

    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']

    # calculate mean and sigma for unnormalize
    mean, sigma = calculate_mean_std(train_set)

    return mean, sigma


subject_id_list = [4]

# number of images to generate
batch_size = 24

# GAN info
sfreq = 250
time_sample = 1000
noise = 100

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# mapping to HGD tasks
task_dict = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}
tasks = ['feet', 'left_hand', 'right_hand', 'tongue']

all_channels = [
        'FC1', 'FC2',
        'C3', 'Cz', 'C4',
        'CP1', 'CP2',
        'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
        'CP3', 'CPz', 'CP4',
        'Fz', 'P1', 'Pz', 'P2', 'POz']

for subject_id in subject_id_list:
    # Load subject data
    data_load_path = os.path.join('../../Data/Real_Data/BCI/bnci-raw/0-38/' + str(subject_id)) + '/'
    dataset = load_concat_dataset(
        path=data_load_path,
        preload=False,
        target_name=None,
    )

    for run in range(0, 6):
        start = 0
        task_trials_epoch = []
        for task in tasks:
            task_channels_trials = np.empty(shape=(batch_size, 0, time_sample))

            for channel in all_channels:
                # path to generator weights .pth file
                saved_models_path = '../../Model_Params/GANs/WGan-GP-Signal-VERSION4-NORMAL18/' + str(subject_id) + '/' + task + '/' + channel + '/'
                saved_models_path += 'generator_state_dict.pth'

                # Calculate mean and varians for unNormalize output later
                mean, sigma = get_data_mean_sigma(dataset, mapping={task: task_dict[task]}, pick_channels=[channel])

                netG = Generator(time_sample=time_sample, noise=noise, channels=1)

                # load weights -tl
                netG.load_state_dict(torch.load(saved_models_path))

                if cuda:
                    netG.cuda()

                # initialize noise
                z = Variable(Tensor(np.random.normal(0, 1, (batch_size, noise))))

                gen_sig = netG(z)

                # Unnormalize
                gen_sig_unn = unNormalizeTanh(gen_sig.detach().cpu().numpy(), mean, sigma)

                task_channels_trials = np.append(task_channels_trials, gen_sig_unn, axis=1)

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
        fake_data_path = '../../Data/Fake_Data/WGan-GP-Signal-VERSION4-NORMAL18/' + str(subject_id) + '/' + 'Runs' + '/' + str(run) +'/'
        if not os.path.exists(fake_data_path):
            os.makedirs(fake_data_path)

        fake_dataset.save(path=fake_data_path, overwrite=True)


