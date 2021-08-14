"""
main base code:
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
"""
import os
import torch
import numpy as np

from skorch.dataset import unpack_data
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.util import set_random_seeds

from models.WGanGPErikSignal import WGANGP


def get_data(data_directory='bnci-raw/', subject_id=1, time_sample=32,
             window_stride_samples=1, mapping=None, pick_channels=None):

    # Dataset
    dataset = load_concat_dataset(
        path='../Dataset-Files/data-file/' + data_directory + str(subject_id),
        preload=False,
        target_name=None,

    )

    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    trial_start_offset_samples = int(-0.5 * sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=False,
        window_size_samples=time_sample,
        window_stride_samples=window_stride_samples,
        drop_bad_windows=True,
        picks=pick_channels,
        mapping=mapping,
    )
    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']
    n_chans = windows_dataset[0][0].shape[0]

    i = 0
    events_num = train_set.datasets[0].windows.events.shape[0]
    runs_num = len(dataset.datasets)
    epochs_num = events_num * runs_num

    data = np.empty(shape=(epochs_num, n_chans, time_sample))
    for x, y, window_ind in train_set:
        data[i] = x
        i += 1

    return data, n_chans


#########################
# load data             #
#########################
data_directory = 'bnci-raw/'
subject_id = 1

mapping = {
    # Select just 'feet' task
    'right_hand': 0,
}

pick_channels = ['C3']              # For All channels set None


time_sample = 500
window_stride_samples = 467

tasks_name = ''
for key, value in mapping.items():
    tasks_name += key
    tasks_name += '_'

channels_name = ''
for ch in pick_channels:
    channels_name += ch
    channels_name += '_'

save_result_path = 'results/WGanGP_EEG_samples/' + str(subject_id) + '/' + channels_name + '/' + tasks_name
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)

save_model_path = '../saved_models/WGan-Gp/' + str(subject_id) + '/' + channels_name + '/' + tasks_name
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

cuda = True if torch.cuda.is_available() else False

seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed, cuda=cuda)


data, n_chans = get_data(data_directory=data_directory,
                         subject_id=subject_id,
                         time_sample=time_sample,
                         window_stride_samples=window_stride_samples,
                         mapping=mapping,
                         pick_channels=pick_channels
                         )

#########################
# Running params        #
#########################

batchsize = 32
epochs = 1500

net = WGANGP(subject=subject_id,
             n_epochs=epochs,
             batch_size=batchsize,
             time_sample=time_sample,
             channels=n_chans,
             sample_interval=window_stride_samples,
             result_path=save_result_path,
             )

net.train(data, save_model_path=save_model_path)
