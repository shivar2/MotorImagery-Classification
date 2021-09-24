import os

from Code.Classification.HGDCroppedClassification import run_model


# example of HGD deep4 - subject 1
subject_id = 1
data_load_path = os.path.join('../../../Data/Real_Data/HGD/22channels/0-f/')
# Path to saving Models
# mkdir path to save
# save_path = os.path.join('../../../Model_Params/Pretrained_Models/42channels/0-f/')
save_path = os.path.join('../../../Model_Params/Pretrained_Models/42channels/0-f/')

if not os.path.exists(save_path):
    os.makedirs(save_path)

run_model(data_load_path=data_load_path,
          subject_id=subject_id,
          model_name='deep4',
          save_path=save_path)
