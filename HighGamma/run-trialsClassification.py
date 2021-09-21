import os

from Classification.trialsClassification import run_model


# example of HGD shallow - subject 1-all tasks-right hand
subject_id_list = [1]

# Path to saving models
# mkdir path to save
save_path = os.path.join('../saved_models/HGD/trials/shallow/' + str(subject_id_list).strip('[]')) + '/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

print('HGD  -  Trials  -  Shallow  -  Subject ' + str(subject_id_list).strip('[]'))
run_model(data_directory='hgd-raw/',
          subject_id_list=subject_id_list,
          dataset_name='HGD',
          model_name='shallow',
          save_path=save_path)


# # example of HGD deep4 - subject 1-all tasks-right hand
# subject_id_list = [1-all tasks-right hand]
#
# # Path to saving models
# # mkdir path to save
# save_path = os.path.join('../saved_models/HGD/trials/deep4/' + str(subject_id_list).strip('[]')) + '/'
#
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
#
# print('HGD  -  Trials  -  Deep4  -  Subject ' + str(subject_id_list).strip('[]')
# run_model(data_directory='hgd-raw/',
#           subject_id_list=subject_id_list,
#           dataset_name='HGD',
#           model_name='deep4',
#           save_path=save_path)
