import os

from Code.Classification.TLClassification import run_model


subject_id_list = [1]
for subject_id in subject_id_list:
    data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/' + str(subject_id)) + '/'

    # Load model path
    model_load_path = '../../../Model_Params/Pretrained_Models/deep4/22/1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14/'

    # Save results
    save_path = os.path.join('../../../Model_Params/TL_Classification/22/' + str(subject_id)) + '/Run-num:0/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    run_model(data_load_path=data_load_path,
              double_channel=False,
              model_load_path=model_load_path,
              params_name='params_10.pt',
              save_path=save_path)
