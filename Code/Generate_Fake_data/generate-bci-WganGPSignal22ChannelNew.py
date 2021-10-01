import os
import random
import numpy as np
import pandas as pd
import mne
import torch.utils.data
from torch.autograd import Variable

from braindecode.datasets.base import WindowsDataset, BaseConcatDataset
from braindecode.util import set_random_seeds

from Code.Models.GANs.WGanGPSignalModels import Generator


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed, cuda=cuda)


subject_id_list = [8]
normalizer_name = 'MaxNormalized/'       # 'tanhNormalized/'

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


# number of images to generate
batch_size = 24 * 4

# GAN info
sfreq = 250
time_sample = 250
noise = 100

for subject_id in subject_id_list:

    for run in range(0, 10):
        start = 0
        task_trials_epoch = []

        for task in tasks:

            # path to generator weights .pth file
            saved_models_path = '../../Model_Params/GANs/WGan-GP-Signal-VERSION7/' +\
                                    normalizer_name + str(subject_id) + '/' + task + '/'
            saved_models_path += 'generator_state_dict.pth'

            netG = Generator(time_sample=time_sample, noise=noise, channels=22)

            # load weights
            netG.load_state_dict(torch.load(saved_models_path))

            if cuda:
                netG.cuda()

            # initialize noise
            z = Variable(Tensor(np.random.normal(0, 1, (batch_size, noise))))

            gen_sig = netG(z)

            task_channels_trial = gen_sig.detach().cpu().numpy()

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

            # merge 4s together to create 1 epoch
            task_channels_trial = task_channels_trial.reshape(-1, 22, 1000)
            # Creating Epoch objects

            for tct in task_channels_trial:
                    events = np.column_stack((np.arange(start, start + int(sfreq/2), sfreq),
                                              np.ones(1, dtype=int) * 1000,
                                              target))

                    metadata = pd.DataFrame({
                        'i_window_in_trial': np.arange(len(events)),
                        'i_start_in_trial': start + events[0][2],
                        'i_stop_in_trial': start + events[0][0] + time_sample,
                        'target': len(events) * [target]
                    })

                    epoch_data = np.array(tct)
                    epoch_data = np.reshape(epoch_data, newshape=(-1, len(all_channels), time_sample*4))

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
        fake_data_path = '../../Data/Fake_Data/WGan-GP-Signal-VERSION7/' + str(subject_id) + '/' + 'Runs' + '/' + str(run) +'/'
        if not os.path.exists(fake_data_path):
            os.makedirs(fake_data_path)

        fake_dataset.save(path=fake_data_path, overwrite=True)


