import os

from Code.Classification.FinalClassification import run_model

fake_k = 2
subject_id_list = [1]

for subject_id in subject_id_list:
    data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/0-38/' + str(subject_id_list)) + '/'
    fake_data_load_path = os.path.join('../../../Data/Fake_Data/WGan-GP-Signal/' + str(subject_id_list)) + '/Runs/'

    # Load model path
    model_load_path = '../../../Model_Params/Pretrained_Models/22channels/0-f/'

    # Save results
    save_path = os.path.join('../../../Model_Params/Final_Classification/phase1/22channels/0-38/' +
                             str(subject_id_list)) + '/Run 1/'
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
