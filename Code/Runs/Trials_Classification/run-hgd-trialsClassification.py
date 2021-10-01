import os

from braindecode.util import set_random_seeds

from Code.Preprocess import add_channel_to_raw
from Code.base import load_data_object, create_model_deep4_auto,\
    create_model_newDeep4, create_model_newDeep4_3d, detect_device

from Code.TrialClassifications import HGDTrialsClassification

# Run Info
subject_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
phase_number = '1'
model_name = "deep4"
channels = 42

normalize = True
if normalize:
    normalize_str = 'normalize/'
else:
    normalize_str = 'notNormalize/'

cuda, device = detect_device()
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

for subject_id in subject_id_list:
    # data
    if normalize:
        data_load_path = os.path.join('../../../Data/Real_Data/HGD/22channels-zmax/0-f/' + str(subject_id)) + '/'
    else:
        data_load_path = os.path.join('../../../Data/Real_Data/HGD/22channels/0-f/' + str(subject_id)) + '/'

    dataset = load_data_object(data_load_path)

    if channels == 42:
        dataset = add_channel_to_raw(dataset)

    input_window_samples = 1125     #?
    n_classes = 4
    n_chans = dataset[0][0].shape[0]

    if model_name == 'deep4':
        model = create_model_deep4_auto(input_window_samples, n_chans, n_classes)

    elif model_name == 'deep4New':
        model = create_model_newDeep4(input_window_samples, n_chans, n_classes)

    else:
        model = create_model_newDeep4_3d(input_window_samples, n_chans, n_classes)

    # Send model to GPU
    if cuda:
        model.cuda()

    # Path to saving Models
    # mkdir path to save
    save_path = os.path.join('../../../Model_Params/HGD_Models/' + '22channels/' + '0-f/' +
                             model_name + '/' + phase_number + ' - ' + normalize_str + str(subject_id)) + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    HGDTrialsClassification.run_model(dataset=dataset, model=model, phase=phase_number, save_path=save_path)

