import os
import torch

from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape

from Code.base import load_data_object, create_model_deep4,\
    create_model_newDeep4, create_model_newDeep4_3d, detect_device

from Code.Classifications import TLClassification


# Run Info
subject_id_list = []
phase_number = '2'
model_name = "deep4"

freq = '0-38/'

normalize_type = '-zmax/'     # '/' for not normalize

# TL
param_name = "params_28.pt"
double_channel = False

cuda, device = detect_device()
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

for subject_id in subject_id_list:
    # data
    data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/' + freq + '22channels' +
                                  normalize_type + str(subject_id)) + '/'

    dataset = load_data_object(data_load_path)

    input_window_samples = 1000
    n_classes = 4
    n_chans = dataset[0][0].shape[0]

    # Load model path
    model_load_path = '../../../Model_Params/Pretrained_Models' + normalize_type + '22channels/' + freq + model_name + '-1/'

    if model_name == 'deep4':
        model = create_model_deep4(n_chans, n_classes)

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
    save_path = os.path.join('../../../Model_Params/TL_Classification' + normalize_type + freq +
                             model_name + '/' + phase_number + '/' + str(subject_id)) + '/classifier/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    TLClassification.run_model(dataset=dataset, model=model,
                               n_preds_per_input=n_preds_per_input,
                               device=device, load_path=model_load_path, param_name=param_name,
                               double_channel=double_channel, save_path=save_path)


