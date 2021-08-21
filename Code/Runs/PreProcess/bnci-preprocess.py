from Code.Preprocess import *

subject_id = 1
# set_mne_path(dataset_name="BNCI", mne_data_path='../Dataset-Files/mne_data')
#
# dataset = load_data_moabb(dataset_name="BNCI2014001", subject_id=subject_id)
# dataset = basic_preprocess(dataset, low_cut_hz=4., high_cut_hz=38., factor_new=1e-3, init_block_size=1000)

dataset = load_preprocessed_data(data_path='../../../Data/Real_Data/BCI/', dataset_folder='bnci-raw/', subject_id=subject_id)
dataset = reorder_based_hgd_channels(dataset)
save_data(dataset, saving_path='../../../Data/Real_Data/BCI/bnci-raw/', subject_id=subject_id)
