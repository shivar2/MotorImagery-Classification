import numpy as np
import torch
from torch import nn
from torch.nn.functional import elu

from skorch.callbacks import LRScheduler, Checkpoint
from skorch.helper import predefined_split

from braindecode.training.losses import CroppedLoss

from Code.Classifier.EEGTLClassifier import EEGTLClassifier

from Code.EarlyStopClass.EarlyStopClass import EarlyStopping
from Code.base import detect_device, cut_compute_windows, split_into_train_valid, plot, get_results

from braindecode.models.modules import Expression, Ensure4d
from braindecode.models.functions import identity, transpose_time_to_spat, squeeze_final_output


def freezing_model(model, layer):
    # Freezing model
    model.requires_grad_(requires_grad=False)

    if layer == 1:
        # model.ensuredims = Ensure4d()
        # model.dimshuffle = Expression(transpose_time_to_spat)
        model.conv_time = nn.Conv2d(1, 25, kernel_size=(10, 1), stride=(1, 1))
        model.conv_spat = nn.Conv2d(25, 25, kernel_size=(1, 22), stride=(1, 1), bias=False)
        model.conv_bnorm = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        model.conv_nonlin = Expression(elu)
        model.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        model.pool_nonlin = Expression(identity)

    elif layer == 2:
        model.drop_2 = nn.Dropout(p=0.5, inplace=False)
        model.conv_2 = nn.Conv2d(25, 50, kernel_size=(10, 1), stride=(1, 1), bias=False)
        model.bnorm_2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        model.nonlin_2 = Expression(elu)
        model.pool_2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        model.pool_nonlin_2 = Expression(identity)

    elif layer == 3:
        model.drop_3 = nn.Dropout(p=0.5, inplace=False)
        model.conv_3 = nn.Conv2d(50, 100, kernel_size=(10, 1), stride=(1, 1), bias=False)
        model.bnorm_3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        model.nonlin_3 = Expression(elu)
        model.pool_3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        model.pool_nonlin_3 = Expression(identity)

    elif layer == 4:
        model.drop_4 = nn.Dropout(p=0.5, inplace=False)
        model.conv_4 = nn.Conv2d(100, 200, kernel_size=(10, 1), stride=(1, 1), bias=False)
        model.bnorm_4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        model.nonlin_4 = Expression(elu)
        model.pool_4 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        model.pool_nonlin_4 = Expression(identity)

    elif layer == 5:
        model.conv_classifier = nn.Conv2d(200, 4, kernel_size=(2, 1), stride=(1, 1))
        model.softmax = nn.LogSoftmax(dim=1)
        model.squeeze = Expression(squeeze_final_output)

    return model


def train_2phase(train_set_all, save_path, model, double_channel=False, device='cpu'):
    train_set, valid_set = split_into_train_valid(train_set_all, use_final_eval=False)

    batch_size = 64
    # Freezing model
    model.requires_grad_(requires_grad=False)

    # PHASE 1
    n_epochs = 800

    # Checkpoint will save the history
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

    model = freezing_model(model, layer=5)

    clf1 = EEGTLClassifier(
        model,
        double_channel=double_channel,
        is_freezing=True,
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
    cp2 = Checkpoint(monitor=None,
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

    clf2 = EEGTLClassifier(
        model,
        double_channel=double_channel,
        is_freezing=True,
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


def run_model(dataset, model, normalize, double_channel, phase, n_preds_per_input, device, save_path):
    input_window_samples = 1000
    n_classes = 4
    # Extract number of chans and time steps from dataset
    if double_channel:
        n_chans = dataset[0][0].shape[0] * 2
    else:
        n_chans = dataset[0][0].shape[0]

    trial_start_offset_seconds = -0.5

    windows_dataset = cut_compute_windows(dataset,
                                          n_preds_per_input,
                                          normalize=normalize,
                                          input_window_samples=input_window_samples,
                                          trial_start_offset_seconds=trial_start_offset_seconds)

    train_set, test_set = split_into_train_valid(windows_dataset, use_final_eval=True)

    clf = train_2phase(train_set, model=model, save_path=save_path,
                       double_channel=double_channel, device=device)

    plot(clf, save_path)
    torch.save(model, save_path + "model.pth")

    # Get results
    get_results(clf, test_set, save_path=save_path, n_chans=n_chans, input_window_samples=1000)

