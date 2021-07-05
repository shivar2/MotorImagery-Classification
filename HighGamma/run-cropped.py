import os

from Models.cropped import run_model


# example of HGD shallow - subject 1
subject_id_list = [1]

# Path to saving models
# mkdir path to save
save_path = os.path.join('../../saved_models/HGD/cropped/shallow/' + str(subject_id_list).strip('[]')) + '/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

run_model(data_directory='hgd-raw/',
          subject_id_list=subject_id_list,
          dataset_name='HGD',
          model_name='shallow',
          save_path=save_path)


# example of HGD deep4 - subject 1
# subject_id_list = [1]
#
# # Path to saving models
# # mkdir path to save
# save_path = os.path.join('../../saved_models/HGD/cropped/deep4/' + str(subject_id_list).strip('[]')) + '/'
#
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
#
# run_model(data_directory='hgd-raw/',
#           subject_id_list=subject_id_list,
#           dataset_name='HGD',
#           model_name='deep4',
#           save_path=save_path)
