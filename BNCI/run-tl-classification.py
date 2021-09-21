import os

from Classification.transferLearningClassification import run_model


# example of BNCI deep4 - subject 1-all tasks-right hand
subject_id_list = [1]
for subject_id_list in range(1, 10):
    # Path to saving models
    # mkdir path to save
    save_path = os.path.join('../saved_models/BNCI/pre-trained/44/conv_classifier,conv_4/' + str(subject_id_list).strip('[]')) + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load model path
    load_path = '../saved_models/HGD/selected_channels/44/1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14/'

    print('BCI  -  TL  -  Deep4  -  Subject ' + str(subject_id_list).strip('[]'))

    run_model(data_directory='bnci-raw/',
              subject_id_list=[subject_id_list],
              dataset_name='BNCI',
              model_name='deep4',
              double_channel=True,
              load_path=load_path,
              save_path=save_path)
