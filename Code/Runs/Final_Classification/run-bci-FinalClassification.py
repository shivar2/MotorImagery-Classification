import os

import torch
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.util import set_random_seeds

from Code.base import load_data_object, load_fake_data, detect_device, create_model_deep4

from Code.Classifications import FinalClassification

# Run Info
subject_id_list = [1]
phase_number = '2'
model_name = "deep4"
freq = '0-f/'

normalize_type = '-zmax/'     # '/' for not normalize

# TL
param_name = "params_51.pt"
double_channel = False

# Fake data info
fake_k = 3
gan_version = 'WGan-GP-Signal-VERSION9' + normalize_type
gan_epoch_dir = '/7500/'

cuda, device = detect_device()
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

for subject_id in subject_id_list:
    # data

    data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/' + freq + '22channels' +
                                  normalize_type + str(subject_id)) + '/'

    dataset = load_data_object(data_load_path)

    fake_data_load_path = os.path.join('../../../Data/Fake_Data/' + gan_version + freq + str(subject_id)) + \
                          gan_epoch_dir + 'Runs/'

    fake_set = load_fake_data(fake_data_load_path, fake_k)

    input_window_samples = 1000
    n_classes = 4
    n_chans = dataset[0][0].shape[0]

    # Load model path
    model_load_path = '../../../Model_Params/Pretrained_Models/1' +normalize_type + freq + model_name + '/'

    model = create_model_deep4(input_window_samples, n_chans, n_classes)

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
    save_path = os.path.join('../../../Model_Params/Final_Classification' + normalize_type + freq +
                             model_name + '/' + phase_number + '/' + str(subject_id)) + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    FinalClassification.run_model(dataset=dataset, fake_set=fake_set,
                                  model=model, n_preds_per_input=n_preds_per_input,
                                  double_channel=double_channel, phase=phase_number, save_path=save_path)


