import os

from Code.Classification.CroppedClassification import run_model


subject_id_list = [1]
for subject_id in subject_id_list:
    data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/0-38/' + str(subject_id)) + '/'

    # Path to saving Models
    # mkdir path to save
    save_path = os.path.join('../../../Model_Params/BCI_Models/deep4/0-38/' + str(subject_id)) + '/Run 1/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    run_model(data_load_path=data_load_path,
              model_name='deep4',
              save_path=save_path)
