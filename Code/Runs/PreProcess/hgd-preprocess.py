from Code.Preprocess import *

subject_id = 1
set_mne_path(dataset_name="SCHIRRMEISTER2017", mne_data_path='../Dataset-Files/mne_data')

dataset = load_data_moabb(dataset_name="Schirrmeister2017", subject_id=subject_id)
low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 0  # high cut frequency for filtering

# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

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

preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor('pick_channels', ch_names=BCI_44_sensors),
        Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
        Preprocessor('resample', sfreq=250),
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                     factor_new=factor_new, init_block_size=init_block_size)
]


save_data(dataset, saving_path='../../../Data/Real_Data/HGD/hgd-44channels-raw/', subject_id=subject_id)
