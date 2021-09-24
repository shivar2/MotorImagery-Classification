import os

from Code.base import load_all_data_object, create_model_deep4,\
    create_model_newDeep4, create_model_newDeep4_3d

from Code.Preprocess.MIpreprocess import add_channel_to_raw
from Code.Classification import HGDCroppedClassification

# Run Info
subject_id_list = [1]
phase_number = '2'
model_name = "deep4"
channels = 44

normalize = True
if normalize:
    normalize_str = 'Normalize/'
else:
    normalize_str = 'notNormalize/'

for subject_id in subject_id_list:
    # data
    data_load_path = os.path.join('../../../Data/Real_Data/HGD/22channels/0-f/' + str(subject_id)) + '/'
    dataset = load_all_data_object(data_load_path)
    if channels == 42:
        dataset = add_channel_to_raw(dataset)

    input_window_samples = 1000
    n_classes = 4
    n_chans = dataset[0][0].shape[0]

    if model_name == 'deep4':
        model = create_model_deep4(input_window_samples, n_chans, n_classes)

    elif model_name == 'deep4New':
        model = create_model_newDeep4(input_window_samples, n_chans, n_classes)

    else:
        model = create_model_newDeep4_3d(input_window_samples, n_chans, n_classes)

    # Path to saving Models
    # mkdir path to save
    save_path = os.path.join('../../../Model_Params/Pretrained_Models/42channels/0-f/' +
                             model_name + '/' + phase_number + '/' + normalize_str)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    HGDCroppedClassification.run_model(dataset=dataset, model=model, normalize=normalize,
                                       phase=phase_number, save_path=save_path)

