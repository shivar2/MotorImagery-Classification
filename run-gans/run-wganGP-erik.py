"""
main base code:
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
"""
import os
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.serialization import load_concat_dataset

from models.WGanGPErik import WGANGP
from Preprocess.MIpreprocess import get_normalized_cwt_data


def get_data(data_directory='bnci-raw/', subject_id=1, time_sample=32, low_cut_hz=4., high_cut_hz=38., window_stride_samples=1):

    # Dataset
    dataset = load_concat_dataset(
        path='../Dataset-Files/data-file/' + data_directory + str(subject_id),
        preload=False,
        target_name=None,

    )

    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    trial_start_offset_samples = int(-0.5 * sfreq)

    pick_channels = ['C3', 'C4']              # For All channels set None

    mapping = {
        # Select just 'feet' task
        'feet': 0,
    }

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

low_cut_hz = 4.
high_cut_hz = 40.

time_sample = 500
window_stride_samples = 467

save_model_path = '../saved_models/WGan-Gp/' + str(subject_id) + '/'
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

# cuda, device = detect_device()
#
# seed = 20200220  # random seed to make results reproducible
# set_random_seeds(seed=seed, cuda=cuda)

cwt_data, n_chans = get_data(data_directory=data_directory,
                             subject_id=subject_id,
                             time_sample=time_sample,
                             low_cut_hz=low_cut_hz,
                             high_cut_hz=high_cut_hz,
                             window_stride_samples=window_stride_samples)

#########################
# Running params        #
#########################

batchsize = 64
epochs = 2500

net = WGANGP(subject=subject_id,
             n_epochs=epochs,
             batch_size=batchsize,
             time_sample=time_sample,
             channels=n_chans,
             sample_interval=window_stride_samples,
             freq_sample=int(high_cut_hz - low_cut_hz))

net.train(cwt_data, save_model_path=save_model_path)
