import numpy as np
from sklearn.model_selection import train_test_split

from Code.Classifier.EEGTLClassifier import EEGTLClassifier
from Code.Classification.CroppedClassification import *

import torch
from torch import nn
from torch.nn.functional import elu

from braindecode.models.functions import identity

from braindecode.models import Deep4Net
from braindecode.models.modules import Expression
from braindecode.models.functions import squeeze_final_output


def tl_classifier(train_set, valid_set,
                  save_path,
                  model,
                  double_channel=True,
                  device='cpu'):
    
    # For deep4 they should be:
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 2

    # Checkpoint will save the history 
    cp = Checkpoint(monitor=None,
                    f_params=None,
                    f_optimizer=None,
                    f_criterion=None,
                    f_history='history.json',
                    dirname=save_path)

    # Early_stopping
    early_stopping = EarlyStopping(patience=100)

    callbacks = [
        "accuracy",
        ('cp', cp),
        ('patience', early_stopping),
        ("lr_scheduler", LRScheduler('WarmRestartLR')),
    ]

    clf = EEGTLClassifier(
        model,
        double_channel=double_channel,
        is_freezing=True,
        warm_start=True,
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
    clf.fit(train_set, y=None)
    return clf


def run_model(data_load_path, double_channel, model_load_path, params_name, save_path):

    input_window_samples = 1000
    cuda, device = detect_device()

    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    dataset = load_data_object(data_load_path)

    n_classes = 4
    # Extract number of chans and time steps from dataset
    if double_channel:
        n_chans = dataset[0][0].shape[0] * 2
    else:
        n_chans = dataset[0][0].shape[0]

    model = Deep4Net(
        in_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        n_filters_time=25,
        n_filters_spat=25,
        stride_before_pool=True,
        n_filters_2=int(n_chans * 2),
        n_filters_3=int(n_chans * (2 ** 2.0)),
        n_filters_4=int(n_chans * (2 ** 3.0)),
        final_conv_length='auto',
    )

    # Load model
    state_dict = torch.load(model_load_path + params_name, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    # Freezing model
    model.requires_grad_(requires_grad=False)

    model.conv_time = nn.Conv2d(1, 25, kernel_size=(10, 1), stride=(1, 1))
    model.conv_spat = nn.Conv2d(25, 25, kernel_size=(1, 22), stride=(3, 1), bias=False)
    model.bnorm = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    model.conv_nonlin = Expression(elu)
    model.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=0, dilation=1, ceil_mode=False)
    model.pool_nonlin = Expression(identity)
    model.drop_2 = nn.Dropout(p=0.5, inplace=False)
    model.conv_2 = nn.Conv2d(25, 44, kernel_size=(10, 1), stride=(3, 1), bias=False)
    model.bnorm_2 = nn.BatchNorm2d(44, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    model.nonlin_2 = Expression(elu)
    model.pool_2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=0, dilation=1, ceil_mode=False)
    model.pool_nonlin_2 = Expression(identity)

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

    train_set_all, test_set = split_data(windows_dataset)

    X_train, X_valid = train_test_split(train_set_all.datasets, test_size=1, train_size=5)
    train_set = BaseConcatDataset(X_train)
    valid_set = BaseConcatDataset(X_valid)

    clf = tl_classifier(train_set,
                        valid_set,
                        model=model,
                        save_path=save_path,
                        double_channel=double_channel,
                        device=device)

    model2 = clf.module
    state_dict = torch.load(save_path, map_location=device)
    model2.load_state_dict(state_dict, strict=False)

    # Freezing model
    model2.requires_grad_(requires_grad=False)

    model2.drop_3 = nn.Dropout(p=0.5, inplace=False)
    model2.conv_3 = nn.Conv2d(44, 88, kernel_size=(10, 1), stride=(3, 1), bias=False)
    model2.bnorm_3 = nn.BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    model2.nonlin_3 = Expression(elu)
    model2.pool_3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=0, dilation=1, ceil_mode=False)
    model2.pool_nonlin_3 = Expression(identity)

    model2.drop_4 = nn.Dropout(p=0.5)
    # model.conv_4 = nn.Conv2d(176, 352, (10, 1),
    model2.conv_4 = nn.Conv2d(88, 176, (10, 1),
                             stride=(3, 1),
                             bias=False,
                             )
    model2.bnorm_4 = nn.BatchNorm2d(
        176,
        momentum=1e-05,
        affine=True,
        eps=1e-5,
    )
    model2.nonlin_4 = Expression(elu)
    model2.pool_4 = nn.MaxPool2d(kernel_size=(3, 1),
                                stride=(1, 1), )
    model2.pool_nonlin_4 = Expression(identity)

    # Final_conv_length
    final_conv_length = model.final_conv_length

    # Change conv_classifier layer to fine-tune
    model2.conv_classifier = nn.Conv2d(
            int(n_chans * (2 ** 3.0)),
            n_classes,
            (final_conv_length, 1),
            stride=(1, 1),
            padding='same',
            bias=True)

    model.softmax = nn.LogSoftmax(dim=1)
    model.squeeze = Expression(squeeze_final_output)

    clf2 = tl_classifier(train_set,
                            valid_set,
                            model=model2,
                            save_path=save_path,
                            double_channel=double_channel,
                            device=device)

    plot(clf2, save_path)

    # Calculate Mean Accuracy For Test set
    i = 0
    test = np.empty(shape=(len(test_set), n_chans, input_window_samples))
    target = np.empty(shape=(len(test_set)))
    for x, y, window_ind in test_set:
        test[i] = x
        target[i] = y
        i += 1

    score = clf2.score(test, y=target)
    print("EEG TL Classification Score (Accuracy) is:  " + str(score))

