import os
import mne
import numpy as np


from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import exponential_moving_demean

from braindecode.datautil.serialization import load_concat_dataset


def set_mne_path(dataset_name="BNCI", mne_data_path='../Dataset-Files/mne_data'):
    # set MNE config
    dataset_config = 'MNE_DATASETS_' + dataset_name + '_PATH'
    mne.set_config(dataset_config, mne_data_path)


def load_data_moabb(dataset_name="BNCI2014001", subject_id=1):
    # download or load dataset
    dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=[subject_id])
    return dataset


def load_preprocessed_data(data_path='../Dataset-Files/data-file/', dataset_folder='hgd-raw/', subject_id=1):
    dataset = load_concat_dataset(
        path=data_path + dataset_folder + str(subject_id),
        preload=True,
        target_name=None,
    )
    return dataset


def basic_preprocess(dataset, low_cut_hz=4., high_cut_hz=38., factor_new=1e-3, init_block_size=1000):
    low_cut_hz = low_cut_hz  # low cut frequency for filtering
    high_cut_hz = high_cut_hz  # high cut frequency for filtering

    # Parameters for exponential moving standardization
    factor_new = factor_new
    init_block_size = init_block_size
    exponential_moving_fn = 'standardize'  # , 'demean'

    C_44_sensors = [
        'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5',
        'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
        'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h',
        'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h',
        'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h', 'CCP1h',
        'CCP2h', 'CPP1h', 'CPP2h']

    C_22_sensors = [
        'FC1', 'FC2',
        'C3', 'Cz', 'C4',
        'CP1', 'CP2',
        'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
        'CP3', 'CPz', 'CP4',
        'Fz', 'P1', 'Pz', 'P2', 'POz']

    moving_fn = {'standardize': exponential_moving_standardize,
                 'demean': exponential_moving_demean}[exponential_moving_fn]

    preprocessors = [
        MNEPreproc(fn='pick_channels', ch_names=C_22_sensors, ordered=True),
        NumpyPreproc(fn=lambda x: x * 1e6),
        NumpyPreproc(fn=lambda x: np.clip(x, -800, 800)),
    ]

    # preprocessors.append(MNEPreproc(fn='set_eeg_reference', ref_channels='average'), )

    preprocessors.extend([
        # MNEPreproc(fn='resample', sfreq=250),
        # bandpass filter
        MNEPreproc(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
        # exponential moving standardization
        NumpyPreproc(fn=moving_fn, factor_new=factor_new,
                     init_block_size=init_block_size),
    ])

    # Transform the data
    preprocess(dataset, preprocessors)
    return dataset


def save_data(dataset, saving_path, subject_id=1):
    # mkdir path to save
    path = os.path.join(saving_path + str(subject_id))
    if not os.path.exists(path):
        os.makedirs(path)

    # save sets
    dataset.save(path=saving_path + str(subject_id), overwrite=True)
    return 1


def get_normalized_cwt_data(dataset, low_cut_hz=4., high_cut_hz=38., n_channels=22, windows_time=1000):

    i = 0
    events_num = dataset.datasets[0].windows.events.shape[0]
    runs_num = len(dataset.datasets)
    epochs_num = events_num * runs_num

    data = np.empty(shape=(epochs_num, n_channels, windows_time))
    for x, y, window_ind in dataset:
            data[i] = x
            i += 1

    f_num = high_cut_hz - low_cut_hz
    freqs = np.logspace(*np.log10([low_cut_hz, high_cut_hz]), num=int(f_num))
    n_cycles = freqs / 2.
    sfreq = 250

    data_MEpoch = mne.time_frequency.tfr_array_morlet(data,
                                                      sfreq=sfreq,
                                                      freqs=freqs,
                                                      n_cycles=n_cycles,
                                                      use_fft=True,
                                                      decim=1,
                                                      output='complex',
                                                      n_jobs=1)

    # Perform a Morlet CWT on each epoch for feature extraction
    data_MEpoch = np.abs(data_MEpoch)

    # Swap the axes to feed into GAN Models later
    data_MEpoch = np.swapaxes(np.swapaxes(data_MEpoch, 1, 3), 1, 2)

    # ... then normalise the data for faster training
    norm_data_MEpoch = 2 * (data_MEpoch - np.min(data_MEpoch, axis=0)) / \
                       (np.max(data_MEpoch, axis=0) - np.min(data_MEpoch, axis=0)) - 1

    return norm_data_MEpoch


def tanhNormalize(data):
    zscored = data - np.mean(data, keepdims=True, axis=-1)
    zscored = zscored / np.std(zscored, keepdims=True, axis=-1)
    tanhN = 0.5 * (np.tanh(0.01 * zscored))

    if hasattr(data, '_data'):
        data._data = tanhN
    return tanhN
