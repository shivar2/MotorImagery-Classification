import numpy as np
import torch
from torch import nn
from torch.nn.functional import elu

from skorch.callbacks import LRScheduler, Checkpoint
from skorch.helper import predefined_split

from braindecode.datasets.base import BaseConcatDataset
from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.training.losses import CroppedLoss
from braindecode.models import Deep4Net
from braindecode.models.modules import Expression
from braindecode.models.functions import squeeze_final_output

from Code.Classifier.EEGTLClassifier import EEGTLClassifier
from Code.EarlyStopClass.EarlyStopClass import EarlyStopping
from Code.Classification.CroppedClassification import plot

from Code.base import detect_device, cut_compute_windows, split_into_train_valid, get_results


def create_pretrained_model(params_path, device, n_chans=22, n_classes=4, input_window_samples=1000):
    model = Deep4Net(
        in_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=2,
    )
    state_dict = torch.load(params_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    # Freezing model
    model.requires_grad_(requires_grad=False)

    # Change conv_classifier layer to fine-tune
    model.conv_classifier = nn.Conv2d(
            200,
            n_classes,
            (2, 1),
            stride=(1, 1),
            bias=True)

    model.softmax = nn.LogSoftmax(dim=1)
    model.squeeze = Expression(squeeze_final_output)

    return model


def train_1phase(train_set, valid_set, model, double_channel=True, device='cpu'):
    
    # For deep4 they should be:
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 20

    callbacks = [
        "accuracy",
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ]

    clf = EEGTLClassifier(
        model,
        double_channel=double_channel,
        is_freezing=True,
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


def train_2phase(train_set_all,
                  save_path,
                  model,
                  double_channel=True,
                  device='cpu'):

    train_set, valid_set = split_into_train_valid(train_set_all, use_final_eval=False)

    batch_size = 64
    n_epochs = 800

    # PHASE 1

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
    ]

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


def run_model(dataset, fake_set, model_load_path, double_channel, phase, save_path):
    input_window_samples = 1000
    if double_channel:
        n_chans = dataset[0][0].shape[0] * 2
    else:
        n_chans = dataset[0][0].shape[0]

    cuda, device = detect_device()
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    model = create_pretrained_model(n_chans=n_chans,
                                    n_classes=4,
                                    input_window_samples=input_window_samples,
                                    params_path=model_load_path,
                                    device=device)
    # Send model to GPU
    if cuda:
        model.cuda()

    to_dense_prediction_model(model)
    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    trial_start_offset_seconds = -0.5

    windows_dataset = cut_compute_windows(dataset,
                                          n_preds_per_input,
                                          input_window_samples=input_window_samples,
                                          trial_start_offset_seconds=trial_start_offset_seconds)

    train_set, test_set = split_into_train_valid(windows_dataset, use_final_eval=True)

    fake_set.append(train_set)
    X = BaseConcatDataset(fake_set)

    if phase == 1:
        clf = train_1phase(X, test_set, model=model, double_channel=double_channel, device=device)
    else:
        clf = train_2phase(X, test_set, model=model, double_channel=double_channel, device=device)

    plot(clf, save_path)
    torch.save(model, save_path + "model.pth")

    # Get results
    get_results(clf, test_set, save_path=save_path, n_chans=n_chans,input_window_samples=1000)

