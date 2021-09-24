
import os
from Code.base import load_data_object, create_model_deep4,\
    create_model_newDeep4, create_model_newDeep4_3d

from Code.Classification import TLClassification

# Run Info
subject_id_list = [1]
phase_number = '2'
model_name = "deep4"

normalize = True
if normalize:
    normalize_str = 'normalize/'
else:
    normalize_str = 'notNormalize/'

# TL
param_name = "params2.pt"
double_channel = False


for subject_id in subject_id_list:
    # data
    data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/0-38/' + str(subject_id)) + '/'
    dataset = load_data_object(data_load_path)

    input_window_samples = 1000
    n_classes = 4
    n_chans = dataset[0][0].shape[0]

    # Load model path
    model_load_path = '../../../Model_Params/Pretrained_Models/22channels/0-f/'

    # if model_name == 'deep4':
    #     model = create_model_deep4(input_window_samples, n_chans, n_classes)
    #
    # elif model_name == 'deep4New':
    #     model = create_model_newDeep4(input_window_samples, n_chans, n_classes)
    #
    # else:
    #     model = create_model_newDeep4_3d(input_window_samples, n_chans, n_classes)

    # Path to saving Models
    # mkdir path to save
    save_path = os.path.join('../../../Model_Params/TL_Classification/0-38/' +
                             model_name + '/' + phase_number + ' - ' + normalize_str + str(subject_id)) + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    TLClassification.run_model(dataset=dataset, model_load_path=model_load_path + param_name,
                               normalize=normalize, double_channel=double_channel,
                               phase=phase_number, save_path=save_path)


