import os

from Code.Classification.TrialsClassification import run_model


# example of HGD shallow - subject 1
subject_id_list = [1]
data_load_path = os.path.join('../../../Data/Real_Data-old/HGD/hgd-22channels-raw/' + str(subject_id_list).strip('[]')) + '/'

# Save results
save_path = os.path.join('../../../Result/Trials_Classification/HGD/' + str(subject_id_list).strip('[]')) + '/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

print('HGD  -  Trials  -  Shallow  -  Subject ' + str(subject_id_list).strip('[]'))

run_model(data_load_path=data_load_path,
          dataset_name='HGD',
          model_name='shallow',
          save_path=save_path)

