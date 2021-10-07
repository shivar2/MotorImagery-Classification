import os
import random
import numpy as np
import pandas as pd
import mne
import torch.utils.data
from torch.autograd import Variable

from braindecode.datasets.base import WindowsDataset, BaseConcatDataset, BaseDataset
from braindecode.util import set_random_seeds

from Code.Models.GANs.WGanGPSignalModels import Generator


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


subject_id_list = [1]

normalize_type = '-zmax/'   # '-zmax'
freq = '0-f/'

gan_epoch_dir = '/7500/'

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
batch_size = 12

# GAN info
sfreq = 250
time_sample = 1000
noise = 100
fake_num = 6

seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)
# initialize noise
noise_z = Variable(Tensor(np.random.normal(0, 1, (fake_num, len(tasks), batch_size, noise))))


for subject_id in subject_id_list:
    for run in range(0, fake_num):
        datasets, target_list, trials = [], [], []
        task_i = 0
        for task in tasks:
            
            # path to generator weights .pth file
            saved_models_path = '../../Model_Params/GANs/WGan-GP-Signal-VERSION9' + normalize_type + freq +\
                                str(subject_id) +\
                                gan_epoch_dir + task + '/'
            saved_models_path += 'generator_state_dict.pth'

            netG = Generator(time_sample=time_sample, noise=noise, channels=22)

            # load weights
            checkpoint_g = torch.load(saved_models_path)
            netG.load_state_dict(checkpoint_g['model_state_dict'])

            if cuda:
                netG.cuda()

            z = noise_z[run][task_i]
            gen_sig = netG(z)

            task_trial = gen_sig.detach().cpu().numpy()
            trials.extend(task_trial)

            # target = task_dict[task]
            target_list.extend([task for i in range(0, batch_size)])

        # ---------------------
        #  Merge Tasks
        #  MNE Create RAW
        # ---------------------

        # Creating Info objects
        ch_types = ['eeg'] * len(all_channels)
        info = mne.create_info(ch_names=all_channels, ch_types=ch_types, sfreq=sfreq)
        info.set_montage('standard_1020')
        info['description'] = 'My custom dataset'

        #  Shuffle Tasks Trials and Target same order
        temp = list(zip(trials, target_list))
        random.shuffle(temp)
        trials, target_list = zip(*temp)

        # Merge trials
        trials = np.swapaxes(trials, 0, 1)
        trials = np.reshape(trials, (len(all_channels), -1))

        #  MNE Create RAW
        raw = mne.io.RawArray(trials, info)

        #  MNE RAW Annotation
        n_times = time_sample * batch_size * len(tasks)
        inds = np.linspace(int(sfreq * 2), int(n_times - sfreq * 2), num=batch_size*4).astype(int)
        onset = raw.times[inds]

        duration = [4] * batch_size * len(tasks)
        description = target_list

        anns = mne.Annotations(onset, duration, description)
        raw = raw.set_annotations(anns)

        # Add Target to Raw
        fake_descrition = pd.Series(
            data=[target_list, "session_T"],
            index=["target", "session"])

        # Raw to BaseDataset
        base_ds = BaseDataset(raw, fake_descrition, target_name="target")
        datasets.append(base_ds)
        dataset = BaseConcatDataset(datasets)

        # Save fake dataset as BaseDataset obj
        # path to to fake eeg directory
        fake_data_path = '../../Data/Fake_Data/WGan-GP-Signal-VERSION9' +\
                         normalize_type + freq + str(subject_id) + gan_epoch_dir + 'Runs' + '/' + str(run) + '/'

        if not os.path.exists(fake_data_path):
            os.makedirs(fake_data_path)

        dataset.save(path=fake_data_path, overwrite=True)


