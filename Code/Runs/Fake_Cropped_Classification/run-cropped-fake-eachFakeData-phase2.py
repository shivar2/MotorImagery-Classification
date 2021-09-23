import os

import numpy as np
import torch

from skorch.callbacks import LRScheduler, Checkpoint, EarlyStopping
from skorch.helper import predefined_split

from braindecode.datasets.base import BaseConcatDataset
from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss

from Code.Classification.CroppedClassification import plot
from Code.EarlyStopClass.EarlyStopClass import EarlyStopping
from Code.base import detect_device, load_data_object,\
    load_fake_data, cut_compute_windows, create_model_deep4,\
    split_into_train_valid, get_test_data, load_fake_data_oneByOne


def train_cropped_trials(train_set_all, model, save_path, device='cpu'):
    train_set, valid_set = split_into_train_valid(train_set_all, use_final_eval=False)

    batch_size = 64
    n_epochs = 800

    # PHASE 1

    # Checkpoint will save the model with the lowest valid_loss
    cp = Checkpoint(monitor='valid_accuracy_best',
                    f_params="params1.pt",
                    f_optimizer="optimizers1.pt",
                    f_history="history1.json",
                    dirname=save_path, f_criterion=None)

    # Early_stopping
    early_stopping = EarlyStopping(monitor='valid_accuracy', lower_is_better=False, patience=80)

    callbacks = [
        "accuracy",
        ('cp', cp),
        ('patience', early_stopping),
    ]

    clf1 = EEGClassifier(
        model,
        cropped=True,
        max_epochs=n_epochs,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks,
        device=device,
    )
    # Model training for a specified number of epochs. `y` is None as it is already supplied
    # in the dataset.
    clf1.fit(train_set, y=None)

    # PHASE 2

    # Best clf1 valid accuracy
    best_valid_acc_epoch = np.argmax(clf1.history[:, 'valid_accuracy'])
    target_train_loss = clf1.history[best_valid_acc_epoch, 'train_loss']

    # Early_stopping
    early_stopping2 = EarlyStopping(monitor='valid_loss',
                                    divergence_threshold=target_train_loss,
                                    patience=80)

    # Checkpoint will save the model with the lowest valid_loss
    cp2 = Checkpoint(
                     f_params="params2.pt",
                     f_optimizer="optimizers2.pt",
                     dirname=save_path,
                     f_criterion=None)

    callbacks2 = [
        "accuracy",
        ('cp', cp2),
        ('patience', early_stopping2),
    ]

    clf2 = EEGClassifier(
        model,
        cropped=True,
        warm_start=True,
        max_epochs=n_epochs,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks2,
        device=device,
    )

    clf2.initialize()  # This is important!
    clf2.load_params(f_params=save_path + "params1.pt",
                     f_optimizer=save_path + "optimizers1.pt",
                     f_history=save_path + "history1.json")
    clf2.fit(train_set_all, y=None)
    return clf2


def run_model(data_load_path, fake_data_load_path, fake_k, save_path):
    input_window_samples = 1000
    cuda, device = detect_device()

    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    dataset = load_data_object(data_load_path)
    train_set_fake = load_fake_data_oneByOne(fake_data_load_path, fake_k)

    n_classes = 4
    n_chans = 22

    model = create_model_deep4(input_window_samples, n_chans, n_classes)

    # Send model to GPU
    if cuda:
        model.cuda()

    # And now we transform model with strides to a model that outputs dense prediction,
    # so we can use it to obtain predictions for all crops.
    to_dense_prediction_model(model)

    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    trial_start_offset_seconds = -0.5

    windows_dataset = cut_compute_windows(dataset,
                                          n_preds_per_input,
                                          input_window_samples=input_window_samples,
                                          trial_start_offset_seconds=trial_start_offset_seconds)

    train_set_all, test_set = split_into_train_valid(windows_dataset, use_final_eval=True)

    train_set_fake.append(train_set_all)
    X = BaseConcatDataset(train_set_fake)

    clf = train_cropped_trials(X,
                               model=model,
                               save_path=save_path,
                               device=device)

    plot(clf, save_path)

    # Load best Classifier and model for Test
    clf_best = EEGClassifier(
        model,
        cropped=True,
        max_epochs=1,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        iterator_train__shuffle=True,
        batch_size=64,
        device=device,
    )

    clf_best.initialize()  # This is important!
    clf_best.load_params(f_params=save_path + "params2.pt",
                         f_optimizer=save_path + "optimizers2.pt",
                         f_history=save_path + "history.json")

    # Calculate Mean Accuracy For Test set
    i = 0
    test = np.empty(shape=(len(test_set), n_chans, input_window_samples))
    target = np.empty(shape=(len(test_set)))
    for x, y, window_ind in test_set:
        test[i] = x
        target[i] = y
        i += 1
    score = clf_best.score(test, y=target)
    print("EEG Gan Classification Score (Accuracy) is:  " + str(score))

    f = open(save_path + "test-result.txt", "a")
    f.write("EEG Fake Classification Score (Accuracy) is:  " + str(score))
    f.close()


subject_id_list = [6, 7, 8, 9]

for subject_id in subject_id_list:
    for fake_ind in range(0, 6):

        data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/0-38/' + str(subject_id)) + '/'

        fake_data_load_path = os.path.join('../../../Data/Fake_Data/WGan-GP-Signal-VERSION5/MaxNormalize/' + str(subject_id)) + '/Runs/'

        # Path to saving Models
        # mkdir path to save
        save_path = '../../../Model_Params/Fake_Cropped_Classification-ForEachFakeData/phase2/0-38/' +\
                    str(subject_id) + '/fake number ' + str(fake_ind) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        run_model(data_load_path=data_load_path,
                  fake_data_load_path=fake_data_load_path,
                  fake_k=fake_ind,
                  save_path=save_path)