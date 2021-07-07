import os
import mne

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
                      'PPO1h', 'PPO2h',
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


def save_data(dataset, saving_path, subject_id=1):
    # mkdir path to save
    path = os.path.join(saving_path + str(subject_id))
    if not os.path.exists(path):
        os.makedirs(path)

    # save sets
    dataset.save(path=saving_path + str(subject_id), overwrite=True)
    return 1
