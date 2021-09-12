import os

from Code.ClassificationPHASE2.GanClassification2phase import run_model

fake_k = 2
subject_id_list = [1]

for subject_id in subject_id_list:

    data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/4-38/' + str(subject_id)) + '/'

    fake_data_load_path = os.path.join('../../../Data/Fake_Data/WGan-GP-Signal-VERSION2/' + str(subject_id)) + '/Runs/'

    # Path to saving Models
    # mkdir path to save
    save_path = '../../../Model_Params/Fake_Cropped_Classification/phase2/4-38/' + str(subject_id) + '/Run 1/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    run_model(data_load_path=data_load_path,
              fake_data_load_path=fake_data_load_path,
              fake_k=fake_k,
              save_path=save_path)
