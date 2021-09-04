import os

from Code.Classification.HGDCroppedClassification import run_model


# example of HGD deep4 - subject 1
subject_id_list = [1]
data_load_path = os.path.join('../../../Data/Real_Data/HGD/hgd-44channels-raw/0-38/')
# Path to saving Models
# mkdir path to save
save_path = os.path.join('../../../Model_Params/Pretrained_Models/deep4/44channel/0-38/')

if not os.path.exists(save_path):
    os.makedirs(save_path)

run_model(data_load_path=data_load_path,
          dataset_name='HGD',
          model_name='deep4',
          save_path=save_path)
