import os

from Code.Classification.FinalClassification import run_model


subject_id_list = [2]
fake_k = 4

data_load_path = os.path.join('../../../Data/Real_Data-old/BCI/bnci-raw/4-38/' + str(subject_id_list).strip('[]')) + '/'
fake_data_load_path = os.path.join('../../../Data/Fake_Data/WGan-GP-Signal/' + str(subject_id_list).strip('[]')) + '/Runs/'

# Load model path
model_load_path = '../../../Model_Params/Pretrained_Models/without-resample/4-38/1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14/'

# Save results
save_path = os.path.join('../../../Model_Params/Final_Classification/without-resample/4-38/' + str(subject_id_list).strip('[]')) + '/Run 1/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

print('BCI  -  Final Classification  -  Subject ' + str(subject_id_list))

run_model(data_load_path=data_load_path,
          fake_data_load_path=fake_data_load_path,
          fake_k=fake_k,
          double_channel=False,
          model_load_path=model_load_path,
          params_name='params_10.pt',
          save_path=save_path)
