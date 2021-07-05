from Preprocess.preprocess import *

subject_id = 1
set_mne_path(dataset_name="SCHIRRMEISTER2017", mne_data_path='../../Dataset-Files/mne_data')

dataset = load_data_moabb(dataset_name="Schirrmeister2017", subject_id=subject_id)
dataset = basic_preprocess(dataset, low_cut_hz=4., high_cut_hz=38., factor_new=1e-3, init_block_size=1000)

dataset = select_C_channels(dataset)
# dataset = select_BCI_channels(dataset)

save_data(dataset, saving_path='../../Dataset-Files/Dataset-Files/data-file/hgd-raw/', subject_id=subject_id)
