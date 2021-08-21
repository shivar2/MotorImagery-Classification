import os

from Classification.cropped import run_model


# example of BNCI deep4 - subject 1
subject_id_list = [1]
data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/' + str(subject_id_list).strip('[]')) + '/'

# Path to saving models
# mkdir path to save
save_path = os.path.join('../../../Model_Params/BCI_Models/deep4/' + str(subject_id_list).strip('[]')) + '/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

run_model(data_load_path=data_load_path,
          dataset_name='BNCI',
          model_name='deep4',
          save_path=save_path)
