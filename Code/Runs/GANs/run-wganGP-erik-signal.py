"""
main base code:
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
"""
import os
import torch
import numpy as np

from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.util import set_random_seeds
from braindecode.datautil.preprocess import preprocess, Preprocessor

from Code.Models.GANs.WGanGPErikSignal import WGANGP
from Code.Preprocess import tanhNormalize, MaxNormalize


def get_data(data_load_path,
             time_sample=32,
             window_stride_samples=1,
             mapping=None,
             pick_channels=None):
    # Dataset
    dataset = load_concat_dataset(
        path=data_load_path,
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
        preload=True,
        window_size_samples=time_sample,
        window_stride_samples=window_stride_samples,
        drop_bad_windows=True,
        picks=pick_channels,
        mapping=mapping,
    )
    # max normalize
    # preprocess(windows_dataset, [Preprocessor(MaxNormalize)])
    # tanh normalize
    preprocess(windows_dataset, [Preprocessor(tanhNormalize)])

    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']
    n_chans = windows_dataset[0][0].shape[0]

    i = 0
    events_num = train_set.datasets[0].windows.events.shape[0]
    runs_num = len(train_set.datasets)
    epochs_num = events_num * runs_num

    data = np.empty(shape=(epochs_num, n_chans, time_sample))
    for x, y, window_ind in train_set:
        data[i] = x
        i += 1

    data = data.reshape(-1, n_chans, 250)
    return data, n_chans


#########################
# load data             #
#########################
subject_id = 8
data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/0-38/22channels/' + str(subject_id)) + '/'

normalizer_name = 'tanhNormalized/'

time_sample = 1000
window_stride_samples = 467
mapping = {'left_hand': 0,
           'right_hand': 1, 'feet': 2, 'tongue': 3
           }

all_channels = ['Fz',
                 'FC1', 'FC2',
                 'C3', 'Cz', 'C4', 'CP1', 'CP2',
                 'Pz', 'POz', 'FC3', 'FCz', 'FC4',
                 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4',
                 'P1', 'P2']

for key, value in mapping.items():
    tasks_name = key
    for channel in all_channels:
        print(channel)

        channels_name = channel
        pick_channels = [channel]
        task_mapping = {
            key: value
        }

        save_result_path = '../../../Result/GANs/WGan-GP-Signal/' + normalizer_name + str(
            subject_id) + '/' + tasks_name + '/' + channels_name + '/'
        if not os.path.exists(save_result_path):
            os.makedirs(save_result_path)

        save_model_path = '../../../Model_Params/GANs/WGan-GP-Signal/' + normalizer_name + str(
            subject_id) + '/' + tasks_name + '/' + channels_name + '/'
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        cuda = True if torch.cuda.is_available() else False

        seed = 20200220  # random seed to make results reproducible
        set_random_seeds(seed=seed, cuda=cuda)

        data, n_chans = get_data(data_load_path=data_load_path,
                                 time_sample=time_sample,
                                 window_stride_samples=window_stride_samples,
                                 mapping=task_mapping,
                                 pick_channels=pick_channels
                                 )

        #########################
        # Running params        #
        #########################

        batchsize = 64
        epochs = 1000

        net = WGANGP(subject=subject_id,
                     n_epochs=epochs,
                     batch_size=batchsize,
                     time_sample=250,
                     channels=n_chans,
                     sample_interval=400,
                     result_path=save_result_path,
                     )

        net.train(data, save_model_path=save_model_path)

print("end")

