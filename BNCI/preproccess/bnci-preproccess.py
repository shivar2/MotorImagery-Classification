# set MNE config
import mne

mne.set_config('MNE_DATASETS_BNCI_PATH', 'mne_data')


# download or load dataset
from braindecode.datasets.moabb import MOABBDataset

subject_id = 1
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])


# preprocess
from braindecode.datautil.preprocess import (
    exponential_moving_standardize, preprocess, Preprocessor)

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors)


# mkdir path to save
import os
path = os.path.join('../../data-file/bnci-raw/' + str(subject_id))
if not os.path.exists(path):
    os.makedirs(path)


# save sets
dataset.save(path='../../data-file/bnci-raw/' + str(subject_id))
