import os

from Classification.FinalClassification import run_model


# example of BNCI deep4 - subject 1
subject_id_list = [1]
for subject_id in subject_id_list:
    # Path to saving models
    # mkdir path to save
    save_path = os.path.join('../saved_models/BNCI/pre-trained/22/ganClassification/' + str(subject_id)) + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load model path
    load_path = '../saved_models/HGD/selected_channels/22/1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14/'

    print('BCI  -  TL  -  With Fake Data  -  Subject ' + str(subject_id))

    run_model(subject_id_list=[subject_id],
              double_channel=False,
              load_path=load_path,
              save_path=save_path)
