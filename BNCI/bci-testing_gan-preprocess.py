from Preprocess.MIpreprocess import *

for subject_id in range(1, 10):
    dataset = load_preprocessed_data(data_path='../Dataset-Files/data-file/',
                                     dataset_folder='bnci-raw/',
                                     subject_id=subject_id)
    dataset = select_3_channels(dataset)
    save_data(dataset, saving_path='../Dataset-Files/data-file/bnci-3channels-raw/', subject_id=subject_id)
