import os

from Code.base import load_data_object, create_model_deep4,\
    create_model_newDeep4, create_model_newDeep4_3d, load_fake_data_oneByOne

from Code.Classification import GanClassification

# Run Info
subject_id_list = [1]
phase_number = '2'
model_name = "deep4"
normalize = True
if normalize:
    normalize_str = 'Normalize/'
else:
    normalize_str = 'notNormalize/'

# Fake data info
fake_k = 2
gan_version = 'WGan-GP-Signal-VERSION5/'

for subject_id in subject_id_list:
    for fake_ind in range(0, 6):
        # data
        data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/0-38/' + str(subject_id)) + '/'
        dataset = load_data_object(data_load_path)

        fake_data_load_path = os.path.join('../../../Data/Fake_Data/' + gan_version + str(subject_id)) + '/Runs/'
        fake_set = load_fake_data_oneByOne(fake_data_load_path, fake_k)

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
        save_path = os.path.join('../../../Model_Params/FakeClassification-each/0-38/' +
                                 model_name + '/' + phase_number + '/' + normalize_str + str(subject_id)) + '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        GanClassification.run_model(dataset=dataset, fake_set=fake_set, model=model, normalize=normalize,
                                    phase=phase_number, save_path=save_path)

