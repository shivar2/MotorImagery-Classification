import os

from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape

from Code.base import load_data_object, create_model_deep4,\
    create_model_newDeep4, create_model_newDeep4_3d, load_fake_data_oneByOne, detect_device

from Code.Classifications import GanClassification
from Code.Models.deepNewUtils import deep4New3dutils

# Run Info
subject_id_list = [5, 6, 7, 8, 9]
phase_number = '2'
model_name = "deep4"

normalize_type = '-stdmax/'   # '-zmax'
freq = '0-f/'
window_size = '-500'
gan_epoch_dir = '/7500/'

# Fake data info

gan_version = 'WGan-GP-Signal-VERSION9' + window_size + normalize_type

cuda, device = detect_device()
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

for subject_id in subject_id_list:
    for fake_ind in range(0, 10):
        # data
        if model_name == 'deep4':
            data_load_path = os.path.join(
                    '../../../Data/Real_Data/BCI/bnci-raw/' + freq + '22channels' + normalize_type +
                    str(subject_id)) + '/'
        else:
            data_load_path = os.path.join(
                '../../../Data/Real_Data/BCI/bnci-raw/' + freq + '42channels/' + str(subject_id)) + '/'

        dataset = load_data_object(data_load_path)

        fake_data_load_path = os.path.join('../../../Data/Fake_Data/' + gan_version + freq + str(subject_id)) +\
                              gan_epoch_dir + 'Runs/'

        fake_set = load_fake_data_oneByOne(fake_data_load_path, fake_ind)

        input_window_samples = 1000
        n_classes = 4
        n_chans = dataset[0][0].shape[0]

        # Load model path
        phase1_clf_path = '../../../Model_Params/BCI_Models' + normalize_type + freq + model_name + '-2/' + \
                            str(subject_id) + '/'

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
        save_path = os.path.join('../../../Model_Params/FakeClassification-each' + window_size + normalize_type +
                                 freq + gan_version + model_name + '-' + phase_number + '/' +
                                 str(subject_id)) + gan_epoch_dir + 'fake number ' + str(fake_ind) + '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        GanClassification.run_model(dataset=dataset, fake_set=fake_set, model=model,
                                    phase=phase_number, n_preds_per_input=n_preds_per_input, device=device,
                                    phase1_clf_path=phase1_clf_path,
                                    save_path=save_path)

