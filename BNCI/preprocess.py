from Preprocess.preprocess import *

subject_id = 1
set_mne_path(dataset_name="BNCI", mne_data_path='../../Dataset-Files/mne_data')

dataset = load_data_moabb(dataset_name="BNCI2014001", subject_id=subject_id)
dataset = basic_preprocess(dataset, low_cut_hz=4., high_cut_hz=38., factor_new=1e-3, init_block_size=1000)
save_data(dataset, saving_path='../../Dataset-Files/data-file/bnci-raw/', subject_id=subject_id)
