"""
main base code:
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
"""
import os
import torch

from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.util import set_random_seeds

from models.WGanGPErik import WGANGP
from Preprocess.MIpreprocess import get_normalized_cwt_data


def get_data(data_directory='bnci-raw/', subject_id=1, time_sample=32, low_cut_hz=4., high_cut_hz=38.,
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

    cwt_data = get_normalized_cwt_data(dataset=train_set,
                                       low_cut_hz=low_cut_hz,
                                       high_cut_hz=high_cut_hz,
                                       n_channels=n_chans,
                                       windows_time=time_sample)

    return cwt_data, n_chans


#########################
# load data             #
#########################
data_directory = 'bnci-raw/'
subject_id = 1

mapping = {
    # Select just 'feet' task
    'feet': 0,
    'left_hand': 1,
    'right_hand': 2,
    'tongue': 3
}

low_cut_hz = 4.
high_cut_hz = 38.

time_sample = 250
window_stride_samples = 467

all_channels = ['Fz',
                   'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                   'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                   'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                   'P1', 'Pz', 'P2',
                   'POz']

cuda = True if torch.cuda.is_available() else False

seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed, cuda=cuda)
for channel in all_channels:
    for key, value in mapping.items():

        tasks_name = key
        channels_name = channel

        pick_channels = [channel]
        task_mapping = {
            key: value
        }

        save_result_path = 'results/WGanGP_EEG_samples/' + str(subject_id) + '/' + tasks_name + '/' + channels_name
        if not os.path.exists(save_result_path):
            os.makedirs(save_result_path)

        save_model_path = '../saved_models/WGan-Gp/' + str(subject_id) + '/' + tasks_name + '/' + channels_name
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        cwt_data, n_chans = get_data(data_directory=data_directory,
                                     subject_id=subject_id,
                                     time_sample=time_sample,
                                     low_cut_hz=low_cut_hz,
                                     high_cut_hz=high_cut_hz,
                                     window_stride_samples=window_stride_samples,
                                     mapping=task_mapping,
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
                     channels_name=pick_channels,
                     freq_sample=int(high_cut_hz - low_cut_hz),
                     result_path=save_result_path,
                     )

        net.train(cwt_data, save_model_path=save_model_path)
