import torch
from torch import nn
from torch.nn.functional import elu
import numpy as np

from skorch.callbacks import Checkpoint
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
from Code.base import detect_device, load_data_object, \
    cut_compute_windows, load_fake_data,\
    split_into_train_valid, plot


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


def final_classifier_phase1(train_set_all, fake_set, save_path, model, double_channel=True, device='cpu'):

    train_set, valid_set = split_into_train_valid(train_set_all, use_final_eval=False)

    fake_set.append(train_set)
    real_fake_train_set = BaseConcatDataset(fake_set)

    fake_set.append(valid_set)
    real_fake_All = BaseConcatDataset(fake_set)

    batch_size = 64

    clf1 = EEGTLClassifier(
        model,
        double_channel=double_channel,
        is_freezing=True,
        cropped=True,
        max_epochs=5,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=["accuracy"],
        device=device,
    )
    # step1 train with real train set
    clf1.fit(train_set, y=None)

    # step2 train on real and fake train set
    clf1.fit(real_fake_train_set, y=None)

    # step2
    # unfreezing model
    model.requires_grad_(requires_grad=True)

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

    clf2 = EEGTLClassifier(
        model,
        double_channel=double_channel,
        is_freezing=True,
        cropped=True,
        max_epochs=800,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks,
        device=device,
    )
    # step3 train on real and fake train set and unfreezing model
    clf2.fit(real_fake_train_set, y=None)

    # PHASE 2

    # Best clf1 valid accuracy
    best_valid_acc_epoch = np.argmax(clf2.history[:, 'valid_accuracy'])
    target_train_loss = clf2.history[best_valid_acc_epoch, 'train_loss']

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

    clf3 = EEGTLClassifier(
        model,
        double_channel=double_channel,
        is_freezing=True,
        cropped=True,
        warm_start=True,
        max_epochs=800,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks2,
        device=device,
    )

    clf3.initialize()  # This is important!
    clf3.load_params(f_params=save_path + "params1.pt",
                     f_optimizer=save_path + "optimizers1.pt",
                     f_history=save_path + "history1.json")

    # step4 train on real and fake All data (train & valid)
    clf3.fit(real_fake_All, y=None)

    return clf3


def run_model(data_load_path, fake_data_load_path, fake_k, double_channel, model_load_path, params_name, save_path):

    input_window_samples = 1000
    cuda, device = detect_device()

    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    dataset = load_data_object(data_load_path)
    fake_set = load_fake_data(fake_data_load_path, fake_k)

    n_classes = 4
    # Extract number of chans and time steps from dataset
    if double_channel:
        n_chans = dataset[0][0].shape[0] * 2
    else:
        n_chans = dataset[0][0].shape[0]

    model = create_pretrained_model(n_chans=n_chans,
                                    n_classes=n_classes,
                                    input_window_samples=input_window_samples,
                                    params_path=model_load_path + params_name,
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

    train_set_all, test_set = split_into_train_valid(windows_dataset, use_final_eval=True)

    clf = final_classifier_phase1(train_set_all,
                                  fake_set,
                                  model=model,
                                  save_path=save_path,
                                  double_channel=double_channel,
                                  device=device)

    plot(clf, save_path)

    # Calculate Mean Accuracy For Test set
    i = 0
    test = np.empty(shape=(len(test_set), n_chans, input_window_samples))
    target = np.empty(shape=(len(test_set)))
    for x, y, window_ind in test_set:
        if double_channel:
            test[i] = np.repeat(x, 2, 0)  # change channel number (22 to 44)
        else:
            test[i] = x

        target[i] = y
        i += 1

    score = clf.score(test, y=target)
    print("EEG TL Classification Score (Accuracy) is:  " + str(score))

    f = open(save_path + "test-result.txt", "w")
    f.write("EEG TL Classification Score (Accuracy) is:  " + str(score))
    f.close()

