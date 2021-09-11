from Code.Preprocess import *
import numpy as np
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import exponential_moving_demean

subject_id = 1
set_mne_path(dataset_name="SCHIRRMEISTER2017", mne_data_path='../Dataset-Files/mne_data')

dataset = load_data_moabb(dataset_name="Schirrmeister2017", subject_id=subject_id)
low_cut_hz = None  # low cut frequency for filtering
high_cut_hz = None  # high cut frequency for filtering

# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000
exponential_moving_fn = ['standardize'] #, 'demean'

BCI_44_sensors = [
        'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5',
        'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
        'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h',
        'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h',
        'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h', 'CCP1h',
        'CCP2h', 'CPP1h', 'CPP2h']

C_22_sensors = [
        'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5',
        'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
        'CP3', 'CPz', 'CP4', 'Fz']

moving_fn = {'standardize': exponential_moving_standardize,
             'demean': exponential_moving_demean}[exponential_moving_fn]

preprocessors = [
    MNEPreproc(fn='pick_channels', ch_names=BCI_44_sensors, ordered=True),
    NumpyPreproc(fn=lambda x: x * 1e6),
    NumpyPreproc(fn=lambda x: np.clip(x, -800, 800)),
    ]

preprocessors.append(MNEPreproc(fn='set_eeg_reference', ref_channels='average'),)

preprocessors.extend([
        MNEPreproc(fn='resample', sfreq=250),
        # bandpass filter
        MNEPreproc(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
        # exponential moving standardization
        NumpyPreproc(fn=moving_fn, factor_new=factor_new,
                     init_block_size=init_block_size),
    ])

save_data(dataset, saving_path='../../../Data/Real_Data/HGD/hgd-44channels-raw/', subject_id=subject_id)
