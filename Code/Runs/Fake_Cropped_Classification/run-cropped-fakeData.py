import os

from Code.Classification.GanClassification import run_model

# example of BNCI & fake data deep4 - subject 1
subject_id_list = [1]

# Path to saving Models
# mkdir path to save
save_path = os.path.join('../../../Model_Params/Fake_Cropped_Classification/' + str(subject_id_list).strip('[]')) + '/Run: 1/'
data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/' + str(subject_id_list).strip('[]')) + '/'
fake_data_load_path = os.path.join('../../../Data/Fake_Data/WGan-GP-Signal/' + str(subject_id_list).strip('[]')) + '/Runs/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

run_model(data_load_path=data_load_path,
          fake_data_load_path=fake_data_load_path,
          save_path=save_path)
