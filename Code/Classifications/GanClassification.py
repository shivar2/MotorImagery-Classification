import numpy as np
import torch

from skorch.callbacks import LRScheduler, Checkpoint, TrainEndCheckpoint, LoadInitState
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss

from Code.EarlyStopClass.EarlyStopClass import EarlyStopping
from Code.Classifications.CroppedClassification import plot
from Code.base import cut_compute_windows, split_into_train_valid, get_results, merge_datasets


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


def train_2phase(real_train_valid, fake_train_set, save_path, model, phase1_clf_path, device='cpu'):

    train_set, valid_set = split_into_train_valid(real_train_valid, use_final_eval=False)

    fr_train_valid = fake_train_set + real_train_valid
    fr_train_set = fake_train_set + train_set

    batch_size = 64
    n_epochs = 800

    # PHASE 1

    # PHASE 2

    # Checkpoint will save the history
    cp2 = Checkpoint(monitor='valid_accuracy_best',
                    f_params="params2.pt",
                    f_optimizer="optimizers2.pt",
                    f_history="history2.json",
                    dirname=save_path, f_criterion=None)

    train_end_cp2 = TrainEndCheckpoint(dirname=save_path)
    # Early_stopping
    early_stopping2 = EarlyStopping(monitor='valid_accuracy', lower_is_better=False, patience=80)

    callbacks2 = [
        "accuracy",
        ('cp', cp2),
        ('patience', early_stopping2),
        ("train_end_cp", train_end_cp2),
    ]

    clf2 = EEGClassifier(
        model,
        cropped=True,
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
    clf2.load_params(f_params=phase1_clf_path + "params1.pt",
                     f_optimizer=phase1_clf_path + "optimizers1.pt",
                     f_history=phase1_clf_path + "history1.json")
    clf2.fit(fr_train_set, y=None)

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

    clf3 = EEGClassifier(
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
        callbacks=callbacks3,
        device=device,
    )

    clf3.fit(fr_train_valid, y=None)
    return clf3


def run_model(dataset, fake_set, model,phase, n_preds_per_input, device, save_path, phase1_clf_path):
    input_window_samples = 1000
    n_chans = 22
    trial_start_offset_seconds = -0.5

    # Real Data
    windows_dataset = cut_compute_windows(dataset,
                                          n_preds_per_input,
                                          input_window_samples=input_window_samples,
                                          trial_start_offset_seconds=trial_start_offset_seconds)

    real_train_set, real_test_set = split_into_train_valid(windows_dataset, use_final_eval=True)

    # Real and Fake samples

    windows_fake_set = cut_compute_windows(fake_set,
                                           n_preds_per_input,
                                           input_window_samples=input_window_samples,
                                           trial_start_offset_seconds=trial_start_offset_seconds)

    fake_train_set, fake_test_set = split_into_train_valid(windows_fake_set, use_final_eval=False, split_c=1)

    if phase == 1:
        clf = train_1phase(real_train_set, real_test_set, model=model, device=device)
    else:
        clf = train_2phase(real_train_set, fake_train_set, model=model, device=device,
                           phase1_clf_path=phase1_clf_path,
                           save_path=save_path)

    plot(clf, save_path)
    torch.save(model, save_path + "model.pth")

    # Get results
    get_results(clf, real_test_set, save_path=save_path, n_chans=n_chans, input_window_samples=1000)

