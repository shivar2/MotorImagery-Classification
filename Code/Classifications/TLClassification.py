import numpy as np
import torch
from torch import nn
from torch.nn.functional import elu

from skorch.callbacks import LRScheduler, Checkpoint, TrainEndCheckpoint, LoadInitState
from skorch.helper import predefined_split

from braindecode.training.losses import CroppedLoss
from braindecode.models.functions import identity, transpose_time_to_spat, squeeze_final_output

from Code.Classifier.EEGTLClassifier import EEGTLClassifier

from Code.EarlyStopClass.EarlyStopClass import EarlyStopping
from Code.base import detect_device, cut_compute_windows, split_into_train_valid, plot, get_results


from braindecode.models.modules import Expression
from braindecode.models.functions import squeeze_final_output


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def freezing_model(model, layer):
    # Freezing model
    model.requires_grad_(requires_grad=False)

    if layer == 1:
        model.conv_time = nn.Conv2d(1, 25, kernel_size=(10, 1), stride=(1, 1))
        model.conv_spat = nn.Conv2d(25, 25, kernel_size=(1, 22), stride=(1, 1), bias=False)

    elif layer == 2:
        model.conv_2 = nn.Conv2d(25, 50, kernel_size=(10, 1), stride=(1, 1), bias=False)

    elif layer == 3:
        model.conv_3 = nn.Conv2d(50, 100, kernel_size=(10, 1), stride=(1, 1), bias=False)

    elif layer == 4:
        model.conv_4 = nn.Conv2d(100, 200, kernel_size=(10, 1), stride=(1, 1), bias=False)

    elif layer == 5:
        model.conv_classifier = nn.Conv2d(200, 4, kernel_size=(2, 1), stride=(1, 1), dilation=(81, 1))
        model.softmax = nn.LogSoftmax(dim=1)
        model.squeeze = Expression(squeeze_final_output)

    return model


def create_classifier(model, valid_set, n_epochs, device, train_end_cp, double_channel=False,
                      lr=0.0001, cp=False, save_path=''):

    batch_size = 64

    if cp:
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
            ("train_end_cp", train_end_cp),
        ]
    else:
        callbacks = [
            "accuracy",
        ]

    clf = EEGTLClassifier(
        model,
        warm_start=True,
        max_epochs=n_epochs,
        double_channel=double_channel,
        is_freezing=True,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        # optimizer__weight_decay=weight_decay,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks,
        device=device,
    )
    return clf

def train_StepByStep(train_set_all, save_path, model, double_channel=False, device='cpu'):

    train_set, valid_set = split_into_train_valid(train_set_all, use_final_eval=False)

    # PHASE 1

    # Layer1
    # model.apply(set_bn_eval)
    model = freezing_model(model, layer=1)
    train_end_cp1 = TrainEndCheckpoint(dirname=save_path)
    clf1 = create_classifier(model, valid_set, 20, device,train_end_cp1, double_channel)
    clf1.fit(train_set, y=None)

    # Layer2
    load_state3 = LoadInitState(train_end_cp1)
    model = freezing_model(model, layer=2)
    clf2 = create_classifier(model, valid_set, 20, device, double_channel)
    clf2.fit(train_set, y=None)

    # Layer3
    model = freezing_model(model, layer=3)
    clf3 = create_classifier(model, valid_set, 20, device, double_channel)
    clf3.fit(train_set, y=None)

    # Layer4
    model = freezing_model(model, layer=4)
    clf4 = create_classifier(model, valid_set, 20, device, double_channel)
    clf4.fit(train_set, y=None)

    # Layer5  -  PHASE1
    model = freezing_model(model, layer=5)
    clf5 = create_classifier(model, valid_set, 800, device, double_channel,
                             cp=True, save_path=save_path)
    clf5.fit(train_set, y=None)

    # PHASE 2
    # Best clf1 valid accuracy
    best_valid_acc_epoch = np.argmax(clf5.history[:, 'valid_accuracy'])
    target_train_loss = clf5.history[best_valid_acc_epoch, 'train_loss']

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
        ('patience', early_stopping2)
    ]

    clf6 = EEGTLClassifier(
        model,
        double_channel=double_channel,
        is_freezing=True,
        cropped=True,
        warm_start=True,
        max_epochs=800,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        # optimizer__lr=lr,
        # optimizer__weight_decay=weight_decay,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        batch_size=64,
        callbacks=callbacks2,
        device=device,
    )

    clf6.initialize()  # This is important!
    clf6.load_params(f_params=save_path + "params1.pt",
                     f_optimizer=save_path + "optimizers1.pt",
                     f_history=save_path + "history1.json")

    clf6.fit(train_set_all, y=None)
    return clf6
def steps(real_train_valid, save_path, model, load_path, param_name, device='cpu'):
    train_set, valid_set = split_into_train_valid(real_train_valid, use_final_eval=False)
    batch_size = 64
    n_epochs = 800
    #step1
    # Checkpoint will save the history
    cp1 = Checkpoint(monitor='valid_accuracy_best',
                     f_params="params1.pt",
                     f_optimizer="optimizers1.pt",
                     f_history="history1.json",
                     dirname=save_path, f_criterion=None)

    train_end_cp1 = TrainEndCheckpoint(dirname=save_path)
    # Early_stopping
    early_stopping1 = EarlyStopping(monitor='valid_accuracy', lower_is_better=False, patience=80)

    callbacks1 = [
        "accuracy",
        ('cp', cp1),
        ('patience', early_stopping1),
        ("train_end_cp", train_end_cp1),
    ]

    clf1 = EEGTLClassifier(
        model,
        cropped=True,
        is_freezing=True,
        # warm_start=True,
        max_epochs=n_epochs,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks1,
        device=device,
    )
    clf1.initialize()  # This is important!
    clf1.load_params(f_params=load_path + "params_22.pt",
                     f_optimizer=load_path + "optimizer_22.pt",
                     f_history=load_path + "history.json")

    model.requires_grad_(requires_grad=False)
    # model.conv_classifier = nn.Conv2d(200, 4, kernel_size=(2, 1), stride=(1, 1), dilation=(81, 1))
    # model.softmax = nn.LogSoftmax(dim=1)
    # model.squeeze = Expression(squeeze_final_output)
    # model.conv_time = nn.Conv2d(1, 25, kernel_size=(10, 1), stride=(1, 1))
    # model.conv_spat = nn.Conv2d(25, 25, kernel_size=(1, 22), stride=(1, 1), bias=False)
    model.conv_time = nn.Conv2d(1, 25, kernel_size=(10, 1), stride=(1, 1))
    model.conv_spat = nn.Conv2d(25, 25, kernel_size=(1, 22), stride=(1, 1), bias=False)
    model.bnorm = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    model.conv_nonlin = Expression(elu)
    model.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=0, dilation=1, ceil_mode=False)
    model.pool_nonlin = Expression(identity)
    # model.drop_2 = nn.Dropout(p=0.5, inplace=False)
    # model.conv_2 = nn.Conv2d(25, 50, kernel_size=(10, 1), stride=(3, 1), bias=False)
    # model.bnorm_2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model.nonlin_2 = Expression(elu)
    # model.pool_2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=0, dilation=1, ceil_mode=False)
    # model.pool_nonlin_2 = Expression(identity)

    clf1.fit(train_set, y=None)

    # # PHASE 2

    # Checkpoint will save the history
    cp2 = Checkpoint(monitor='valid_accuracy_best',
                    f_params="params2.pt",
                    f_optimizer="optimizers2.pt",
                    f_history="history2.json",
                    dirname=save_path, f_criterion=None)

    load_state2 = LoadInitState(train_end_cp1)
    train_end_cp2 = TrainEndCheckpoint(dirname=save_path)

    early_stopping2 = EarlyStopping(monitor='valid_accuracy', lower_is_better=False, patience=80)

    callbacks2 = [
        "accuracy",
        ('cp', cp2),
        ('patience', early_stopping2),
        ("train_end_cp", train_end_cp2),
        ("load_state", load_state2),
    ]

    clf2 = EEGTLClassifier(
        model,
        cropped=True,
        warm_start=True,
        is_freezing=True,
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
    model.conv_4 = nn.Conv2d(100, 200, kernel_size=(10, 1), stride=(1, 1), bias=False)
    clf2.fit(train_set, y=None)

    # PHASE 3

    # Best clf1 valid accuracy
    best_valid_acc_epoch = np.argmax(clf2.history[:, 'valid_accuracy'])
    target_train_loss = clf2.history[best_valid_acc_epoch, 'train_loss']

    # Early_stopping
    early_stopping3 = EarlyStopping(monitor='valid_loss',
                                    divergence_threshold=target_train_loss,
                                    patience=80)

    # Checkpoint will save the model with the lowest valid_loss
    cp3 = Checkpoint(
                     f_params="params3.pt",
                     f_optimizer="optimizers3.pt",
                     dirname=save_path,
                     f_criterion=None)

    load_state3 = LoadInitState(train_end_cp2)
    callbacks3 = [
        "accuracy",
        ('cp', cp3),
        ('patience', early_stopping3),
        ("load_state", load_state3),
    ]

    clf3 = EEGTLClassifier(
        model,
        cropped=True,
        is_freezing=True,
        warm_start=True,
        max_epochs=n_epochs,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks3,
        device=device,
    )

    clf3.fit(real_train_valid, y=None)
    return clf3


def train_2phase(train_set_all, save_path, model, double_channel=False, device='cpu'):

    train_set, valid_set = split_into_train_valid(train_set_all, use_final_eval=False)

    batch_size = 64

    # For deep4 they should be:
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

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
    train_end_cp1 = TrainEndCheckpoint(dirname=save_path)

    callbacks = [
        "accuracy",
        ('cp', cp),
        ('patience', early_stopping),
        ("train_end_cp", train_end_cp1),
    ]

    # model.apply(set_bn_eval)
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
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
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

    load_state2 = LoadInitState(train_end_cp1)

    callbacks2 = [
        "accuracy",
        ('cp', cp2),
        ('patience', early_stopping2),
        ("load_state", load_state2),
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
        # optimizer__lr=lr,
        # optimizer__weight_decay=weight_decay,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks2,
        device=device,
    )

    clf2.fit(train_set_all, y=None)
    return clf2


def train_1phase(train_set, valid_set, model, device='cpu'):
    # For deep4 they should be:
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 40

    callbacks = [
        "accuracy",
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ]
    # model.requires_grad_(requires_grad=False)
    #
    # model.conv_time = nn.Conv2d(1, 25, kernel_size=(10, 1), stride=(1, 1))
    # model.conv_spat = nn.Conv2d(25, 25, kernel_size=(1, 22), stride=(1, 1), bias=False)

    clf = EEGTLClassifier(
        model,
        cropped=True,
        is_freezing=True,
        warm_start=True,
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
    clf.fit(train_set, y=None)
    return clf


def run_model(dataset, model, double_channel, load_path, param_name, n_preds_per_input, device, save_path):
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
                                          input_window_samples=input_window_samples,
                                          trial_start_offset_seconds=trial_start_offset_seconds)

    train_set, test_set = split_into_train_valid(windows_dataset, use_final_eval=True)

    clf = train_2phase(train_set, model=model, save_path=save_path, double_channel=double_channel, device=device)
    # clf = train_2phase(train_set, test_set, model=model, device=device)

    plot(clf, save_path)
    torch.save(model, save_path + "model.pth")

    # Get results
    get_results(clf, test_set, save_path=save_path, n_chans=n_chans, input_window_samples=1000)

