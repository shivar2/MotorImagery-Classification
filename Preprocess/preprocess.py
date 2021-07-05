import os
import mne

from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import (
    exponential_moving_standardize, preprocess, Preprocessor)


def set_mne_path(dataset_name="BNCI", mne_data_path='../../Dataset-Files/mne_data'):
    # set MNE config
    dataset_config = 'MNE_DATASETS_' + dataset_name + '_PATH'
    mne.set_config(dataset_config, mne_data_path)


def load_data_moabb(dataset_name="BNCI2014001", subject_id=1):
    # download or load dataset
    dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=[subject_id])
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


def select_BCI_channels(dataset):
    # bci channel
    BCI_sensors = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                   'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                   'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                   'P1', 'Pz', 'P2', 'POz']

    preprocessors = [Preprocessor('pick_channels', ch_names=BCI_sensors)]  # Pick Channels
    preprocess(dataset, preprocessors)

    return dataset


def select_C_channels(dataset):
    # pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']

    preprocessors = [Preprocessor('pick_channels', ch_names=C_sensors)]  # Pick Channels
    preprocess(dataset, preprocessors)

    return dataset


def save_data(dataset, saving_path, subject_id=1):
    # mkdir path to save
    path = os.path.join(saving_path + str(subject_id))
    if not os.path.exists(path):
        os.makedirs(path)

    # save sets
    dataset.save(path=saving_path + str(subject_id), overwrite=True)

