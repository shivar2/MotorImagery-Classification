import os
import mne
import numpy as np


from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import (
    exponential_moving_standardize, preprocess, Preprocessor)

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

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                     factor_new=factor_new, init_block_size=init_block_size)
    ]

    # Transform the data
    preprocess(dataset, preprocessors)
    return dataset


def select_3_channels(dataset):
    # c channel for testing gan
    C_sensors = ['C3', 'Cz', 'C4']

    preprocessors = [Preprocessor('pick_channels', ch_names=C_sensors)]  # Pick Channels
    preprocess(dataset, preprocessors)

    return dataset


def select_22_channels(dataset):
    # bci channel
    BCI_sensors = ['Fz',
                   'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                   'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                   'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                   'P1', 'Pz', 'P2',
                   'POz']

    preprocessors = [Preprocessor('pick_channels', ch_names=BCI_sensors)]  # Pick Channels
    preprocess(dataset, preprocessors)

    return dataset


def select_44_channels(dataset):
    # pick only sensors with C in their name
    # as they cover motor cortex
    BCI_44_sensors = ['Fz',
                      'FFC1h', 'FFC2h', 'FFC3h', 'FFC4h',
                      'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                      'FCC5h', 'FCC3h', 'FCC1h', 'FCC2h', 'FCC4h', 'FCC6h',
                      'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                      'CCP5h', 'CCP3h', 'CCP1h', 'CCP2h', 'CCP4h', 'CCP6h',
                      'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                      'CPP3h', 'CPP1h', 'CPP2h', 'CPP4h',
                      'P1', 'Pz', 'P2',
                      'PPO1', 'PPO2',
                      'POz']

    preprocessors = [Preprocessor('pick_channels', ch_names=BCI_44_sensors)]  # Pick Channels
    preprocess(dataset, preprocessors)

    return dataset


def reorder(dataset):
    # bci channel
    BCI_sensors = ['Fz',
                   'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                   'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                   'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                   'P1', 'Pz', 'P2',
                   'POz']

    preprocessors = [Preprocessor('reorder_channels', ch_names=BCI_sensors)]  # Reorder Channels
    preprocess(dataset, preprocessors)

    return dataset


def reorder_based_hgd_channels(dataset):
    # bci channel
    HGD_oredr = ['Fz',
                 'FC1', 'FC2',
                 'C3', 'Cz', 'C4', 'CP1', 'CP2',
                 'Pz', 'POz', 'FC3', 'FCz', 'FC4',
                 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4',
                 'P1', 'P2']

    preprocessors = [Preprocessor('reorder_channels', ch_names=HGD_oredr)]  # Pick Channels
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
    data = np.empty(shape=(6*144, n_channels, windows_time))
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

    # Swap the axes to feed into GAN models later
    data_MEpoch = np.swapaxes(np.swapaxes(data_MEpoch, 1, 3), 1, 2)

    # ... then normalise the data for faster training
    norm_data_MEpoch = 2 * (data_MEpoch - np.min(data_MEpoch, axis=0)) / \
                       (np.max(data_MEpoch, axis=0) - np.min(data_MEpoch, axis=0)) - 1

    return norm_data_MEpoch

