import os

from Code.Classification.TLClassification import run_model


subject_id_list = [1]
data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/' + str(subject_id_list).strip('[]')) + '/'

# Load model path
model_load_path = '../../../Model_Params/Pretrained_Models/deep4/22/1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14/'

# Save results
save_path = os.path.join('../../../Result/TL_Classification/22/' + str(subject_id_list).strip('[]')) + '/Run-num:0/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


print('BCI  -  TL  -  Deep4  -  Subject ' + str(subject_id_list).strip('[]'))

run_model(data_load_path=data_load_path,
          double_channel=False,
          model_load_path=model_load_path,
          params_name='params_10.pt',
          save_path=save_path)
