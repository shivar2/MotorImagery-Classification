import os

from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape

from Code.base import load_all_data_object, create_model_deep4,\
    create_model_newDeep4, create_model_newDeep4_3d, detect_device

from Code.Preprocess.MIpreprocess import add_channel_to_raw
from Code.Classifications import HGDCroppedClassification
from Code.Models.deepNewUtils import deep4New3dutils

# Run Info
subject_id_list = [1]
phase_number = '1'
model_name = "deep4"
channels = 22

freq = '0-38/'

normalize_type = '-stdmax/'     # '/' for not normalize


cuda, device = detect_device()
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)
for subject_id in subject_id_list:
    # data
    data_load_path = os.path.join('../../../Data/Real_Data/HGD/' + freq+'22channels' + normalize_type)
    dataset = load_all_data_object(data_load_path)
    if channels == 42:
        dataset = add_channel_to_raw(dataset)

    input_window_samples = 1000
    n_classes = 4
    n_chans = dataset[0][0].shape[0]

    if model_name == 'deep4':
        model = create_model_deep4(n_chans, n_classes)

    elif model_name == 'deep4New':
        model = create_model_newDeep4(input_window_samples, n_chans, n_classes)

    else:
        model = create_model_newDeep4_3d(input_window_samples, n_chans, n_classes)

     # Send model to GPU
    if cuda:
        model.cuda()

    # And now we transform model with strides to a model that outputs dense prediction,
    # so we can use it to obtain predictions for all crops.
    if model_name == 'deep43D':
        deep4New3dutils.to_dense_prediction_model(model)
    else:
        to_dense_prediction_model(model)

    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    # Path to saving Models
    # mkdir path to save
    save_path = os.path.join('../../../Model_Params/Pretrained_Models' + normalize_type + '22channels/' + freq +
                             model_name + '-' + phase_number + '/')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    HGDCroppedClassification.run_model(dataset=dataset, model=model,
                                       phase=phase_number, save_path=save_path,
                                       n_preds_per_input=n_preds_per_input)

    # HGDCroppedClassification.continue_trainning(dataset=dataset, model=model,
    #                                             load_path=save_path, save_path=save_path+'continue/',
    #                                             n_preds_per_input=n_preds_per_input)


