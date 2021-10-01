import os
import torch

from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape

from Code.base import load_data_object, create_model_deep4,\
    create_model_newDeep4, create_model_newDeep4_3d, detect_device

from Code.CroppedClassifications import TLClassification


# Run Info
subject_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
phase_number = '2'
model_name = "deep4"

normalize = True
if normalize:
    normalize_str = 'normalize/'
else:
    normalize_str = 'notNormalize/'

# TL
param_name = "params_30.pt"
double_channel = False

cuda, device = detect_device()
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

for subject_id in subject_id_list:
    # data
    if normalize:
        data_load_path = os.path.join(
            '../../../Data/Real_Data/BCI/bnci-raw/0-38/22channels-zmax/' + str(subject_id)) + '/'
    else:
        data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/0-38/22channels/' + str(subject_id)) + '/'

    dataset = load_data_object(data_load_path)

    input_window_samples = 1000
    n_classes = 4
    n_chans = dataset[0][0].shape[0]

    # Load model path
    model_load_path = '../../../Model_Params/Pretrained_Models/22channels/0-f/deep4/1 - normalize/'

    if model_name == 'deep4':
        model = create_model_deep4(input_window_samples, n_chans, n_classes)

    elif model_name == 'deep4New':
        model = create_model_newDeep4(input_window_samples, n_chans, n_classes)

    else:
        model = create_model_newDeep4_3d(input_window_samples, n_chans, n_classes)

    # Send model to GPU
    if cuda:
        model.cuda()

    to_dense_prediction_model(model)
    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    # Load model
    state_dict = torch.load(model_load_path+param_name, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    # Path to saving Models
    # mkdir path to save
    save_path = os.path.join('../../../Model_Params/TL_Classification/0-38/22channel/' +
                             model_name + '/' + phase_number + ' - ' + normalize_str + str(subject_id)) + '/step_by_step/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    TLClassification.run_model(dataset=dataset, model=model,
                               n_preds_per_input=n_preds_per_input,
                               device=device,
                               double_channel=double_channel, save_path=save_path)


