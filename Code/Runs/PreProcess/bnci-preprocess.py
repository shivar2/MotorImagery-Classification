from Code.Preprocess import *

set_mne_path(dataset_name="BNCI", mne_data_path='../../../Data/mne_data')
for subject_id in range(1, 10):

    dataset = load_data_moabb(dataset_name="BNCI2014001", subject_id=subject_id)
    dataset = stdmax_normalize_preprocess(dataset, low_cut_hz=None, high_cut_hz=38.)
    # dataset = load_preprocessed_data(data_path='../../../Data/Real_Data/BCI/bnci-raw/0-38/',
    #                                  subject_id=subject_id)

    # dataset = add_channel_to_raw(dataset)

    save_data(dataset, saving_path='../../../Data/Real_Data/BCI/bnci-raw/0-38/22channels-zmax/', subject_id=subject_id)
