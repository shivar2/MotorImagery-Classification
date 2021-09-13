from Code.Preprocess import *

set_mne_path(dataset_name="BNCI", mne_data_path='../../../Data/mne_data')
for subject_id in range(1, 10):

    dataset = load_data_moabb(dataset_name="BNCI2014001", subject_id=subject_id)
    dataset = basic_preprocess(dataset, low_cut_hz=None, high_cut_hz=38., factor_new=1e-3, init_block_size=1000)
    # dataset = load_preprocessed_data(data_path='../../../Data/Real_Data/BCI/', dataset_folder='bnci-raw/',
    #                                  subject_id=subject_id)

    save_data(dataset, saving_path='../../../Data/Real_Data/BCI/bnci-raw/0-38-tanh/', subject_id=subject_id)
