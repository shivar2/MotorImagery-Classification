import numpy as np
import torch

from skorch.callbacks import LRScheduler, Checkpoint
from skorch.helper import predefined_split

from braindecode.datasets.base import BaseConcatDataset
from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss

from Code.EarlyStopClass.EarlyStopClass import EarlyStopping
from Code.ClassificationBase import detect_device, load_data_object,\
    create_model_deep4, create_model_shallow, cut_compute_windows,\
    split_into_train_valid, get_test_data, plot


def train_cropped_trials(train_set, valid_set, model, save_path, model_name='shallow', device='cpu'):
    if model_name == 'shallow':
        # These values we found good for shallow network:
        lr = 0.0625 * 0.01
        weight_decay = 0
    else:
        # For deep4 they should be:
        lr = 1 * 0.01
        weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 800

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
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ]

    clf1 = EEGClassifier(
        model,
        cropped=True,
        max_epochs=n_epochs,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
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
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
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
    clf2.load_params(f_params=save_path+"params1.pt",
                     f_optimizer=save_path+"optimizers1.pt",
                     f_history=save_path+"history1.json")
    phase2_train = BaseConcatDataset([train_set, valid_set])
    clf2.fit(phase2_train, y=None)
    return clf2


def run_model(data_load_path, model_name, save_path):
    input_window_samples = 1000
    cuda, device = detect_device()

    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    dataset = load_data_object(data_load_path)

    n_classes = 4
    n_chans = dataset[0][0].shape[0]

    if model_name == 'shallow':
        model = create_model_shallow(input_window_samples, n_chans, n_classes)
    else:
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

    train_set, valid_set = split_into_train_valid(windows_dataset, use_final_eval=False)
    test_set = get_test_data(windows_dataset)

    clf = train_cropped_trials(train_set,
                               valid_set,
                               model=model,
                               save_path=save_path,
                               model_name=model_name,
                               device=device)

    plot(clf, save_path)

    # Calculate Mean Accuracy For Test set
    i = 0
    test = np.empty(shape=(len(test_set), n_chans, input_window_samples))
    target = np.empty(shape=(len(test_set)))
    for x, y, window_ind in test_set:
        test[i] = x
        target[i] = y
        i += 1

    score = clf.score(test, y=target)
    print("EEG Cropped Classification Score (Accuracy) is:  " + str(score))

    f = open(save_path + "test-result.txt", "w")
    f.write("EEG Cropped Classification Score (Accuracy) is:  " + str(score))
    f.close()

