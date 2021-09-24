import os

from Code.base import load_data_object, create_model_deep4,\
    create_model_newDeep4, create_model_newDeep4_3d

from Code.Classification import CroppedClassification
from Code.ClassificationPHASE2 import CroppedClassification2phase

# Run Info
subject_id_list = [1]
phase_number = '2'
model_name = "deep4"
normalize = True

for subject_id in subject_id_list:
    # data
    data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/0-38/' + str(subject_id)) + '/'
    dataset = load_data_object(data_load_path)

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
    save_path = os.path.join('../../../Model_Params/BCI_Models/0-38/' +
                             model_name + '/' + phase_number + '/' + str(subject_id))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if phase_number == '1':
        CroppedClassification.run_model(dataset=dataset, model=model,
                                        normalize=normalize, save_path=save_path)
    else:
        CroppedClassification2phase.run_model(dataset=dataset, model=model,
                                              normalize=normalize, save_path=save_path)
