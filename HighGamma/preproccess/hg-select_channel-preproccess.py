# set MNE config
import mne
mne.set_config('MNE_DATASETS_SCHIRRMEISTER2017_PATH', '../../mne_data')


# download or load dataset
from braindecode.datasets.moabb import MOABBDataset
subject_id = 1
dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=[subject_id])


# preprocess
from braindecode.datautil.preprocess import (exponential_moving_standardize, preprocess, Preprocessor)

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

# now pick only sensors with C in their name
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

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor('pick_channels', ch_names=C_sensors),  # Pick Channels
    Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors)


# mkdir path to save
import os
path = os.path.join('../../data-file/hgd-selected-channels-raw/' + str(subject_id))
if not os.path.exists(path):
    os.makedirs(path)

# save sets
dataset.save(path='../../data-file/hgd-selected-channels-raw/' + str(subject_id), overwrite=True)