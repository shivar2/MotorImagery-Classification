import numpy as np
import torch

from skorch.callbacks import LRScheduler, Checkpoint
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss
from braindecode.util import set_random_seeds

from Code.EarlyStopClass.EarlyStopClass import EarlyStopping
from Code.base import cut_compute_windows, plot, split_hgd_into_train_valid, get_results, detect_device


def train_1phase(train_set, valid_set, model, save_path, device='cpu'):
    # For deep4 they should be:
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 40

    cp = Checkpoint(monitor=None,
                    f_params="params_{last_epoch[epoch]}.pt",
                    f_optimizer="optimizer_{last_epoch[epoch]}.pt",
                    f_history="history.json",
                    dirname=save_path, f_criterion=None)

    callbacks = [
        "accuracy",
        ('cp', cp),
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ]

    clf = EEGClassifier(
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
    clf.fit(train_set, y=None)
    return clf


def train_2phase(train_set_all, model, save_path, device='cpu'):

    train_set, valid_set = split_hgd_into_train_valid(train_set_all, use_final_eval=False)

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
    clf2.load_params(f_params=save_path+"params1.pt",
                     f_optimizer=save_path+"optimizers1.pt",
                     f_history=save_path+"history1.json")

    clf2.fit(train_set_all, y=None)
    return clf2


def run_model(dataset, model, phase, n_preds_per_input, save_path):
    cuda, device = detect_device()
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    input_window_samples = 1000
    n_chans = dataset[0][0].shape[0]

    trial_start_offset_seconds = -0.5

    windows_dataset = cut_compute_windows(dataset,
                                          n_preds_per_input,
                                          input_window_samples=input_window_samples,
                                          trial_start_offset_seconds=trial_start_offset_seconds)

    train_set_all, test_set = split_hgd_into_train_valid(windows_dataset, use_final_eval=True)

    if phase == '1':
        clf = train_1phase(train_set_all, test_set, model=model, save_path=save_path, device=device)
    else:
        clf = train_2phase(train_set_all, model=model, save_path=save_path, device=device)

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
    # Get results
    get_results(clf_best, test_set, save_path=save_path, n_chans=n_chans, input_window_samples=1000)


def continue_trainning(dataset, model, n_preds_per_input, load_path, save_path):
    cuda, device = detect_device()
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    input_window_samples = 1000
    n_chans = dataset[0][0].shape[0]

    trial_start_offset_seconds = -0.5

    windows_dataset = cut_compute_windows(dataset,
                                          n_preds_per_input,
                                          input_window_samples=input_window_samples,
                                          trial_start_offset_seconds=trial_start_offset_seconds)

    train_set_all, test_set = split_hgd_into_train_valid(windows_dataset, use_final_eval=True)

    # only for phase 1 clf
    # For deep4 they should be:
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 40

    cp = Checkpoint(monitor="valid_accuracy_best",
                    f_params="params_{last_epoch[epoch]}.pt",
                    f_optimizer="optimizer_{last_epoch[epoch]}.pt",
                    f_history="history.json",
                    dirname=save_path, f_criterion=None)

    callbacks = [
        "accuracy",
        ('cp', cp),
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ]

    clf = EEGClassifier(
        model,
        cropped=True,
        max_epochs=n_epochs,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(test_set),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks,
        device=device,
    )

    clf.initialize()  # This is important!
    clf.load_params(f_params=load_path+"params_40.pt",
                     f_optimizer=load_path+"optimizer_40.pt",
                     f_history=load_path+"history.json")

    clf.fit(train_set_all, y=None)

    plot(clf, save_path)

    # Get results
    get_results(clf, test_set, save_path=save_path, n_chans=n_chans, input_window_samples=1000)


