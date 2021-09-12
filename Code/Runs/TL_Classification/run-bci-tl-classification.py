import os

from Code.Classification.TLClassification import run_model


subject_id_list = [1]
for subject_id in subject_id_list:
    data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/0-38/' + str(subject_id)) + '/'

    # Load model path
    model_load_path = '../../../Model_Params/Pretrained_Models/22channels/0-f/'

    # Save results
    save_path = os.path.join('../../../Model_Params/TL_Classification/phase1/22channels/0-38/' + str(subject_id)) + '/Run 1/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    run_model(data_load_path=data_load_path,
              double_channel=False,
              model_load_path=model_load_path,
              params_name='params_16.pt',
              save_path=save_path)
