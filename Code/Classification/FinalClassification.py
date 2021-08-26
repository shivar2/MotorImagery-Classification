import numpy as np
import torch

from sklearn.model_selection import train_test_split

from skorch.callbacks import LRScheduler, Checkpoint, EarlyStopping
from skorch.helper import predefined_split

from braindecode.datautil.serialization import load_concat_dataset
from braindecode.datasets.base import BaseConcatDataset
from braindecode.util import set_random_seeds
from braindecode.models import Deep4Net
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.training.losses import CroppedLoss

from Code.Classifier.EEGTLClassifier import EEGTLClassifier
from Code.Models.PretrainedDeep4Model import PretrainedDeep4Model
from Code.Classification.CroppedClassification import plot


def detect_device():
    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return cuda, device


def load_data_object(data_path):

    dataset = load_concat_dataset(
                path=data_path,
                preload=True,
                target_name=None,
            )
    return dataset


def load_fake_data(fake_data_path):

    ds_list = []
    for folder in range(0, 4):
        folder_path = fake_data_path + str(folder) + '/'
        ds_loaded = load_concat_dataset(
                path=folder_path,
                preload=True,
                target_name=None,
        )
        ds_list.append(ds_loaded)

    return ds_list


def create_model_deep4(input_window_samples=1000, n_chans=4, n_classes=4):
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
    return model


def cut_compute_windows(dataset, n_preds_per_input, input_window_samples=1000, trial_start_offset_seconds=-0.5):
    # Extract sampling frequency, check that they are same in all datasets

    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        preload=True
    )
    return windows_dataset


def split_data(windows_dataset):
    # Split dataset into train and valid
    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']
    valid_set = splitted['session_E']

    return train_set, valid_set


def tl_classifier(train_set, valid_set,
                  save_path,
                  model,
                  double_channel=True,
                  device='cpu'):
    
    # For deep4 they should be:
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 100

    # Checkpoint will save the history 
    cp = Checkpoint(
                    f_params="params_best_valid_loss_{last_epoch[epoch]}.pt",
                    f_optimizer="optimizer_best_valid_loss_{last_epoch[epoch]}.pt",
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
        warm_start=True,
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


def run_model(data_load_path, fake_data_load_path, double_channel, model_load_path, params_name, save_path):

    input_window_samples = 1000
    cuda, device = detect_device()

    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    dataset = load_data_object(data_load_path)
    train_set_fake = load_fake_data(fake_data_load_path)

    n_classes = 4

    if double_channel:
        n_chans = dataset[0][0].shape[0] * 2
    else:
        n_chans = dataset[0][0].shape[0]

    model = PretrainedDeep4Model(n_chans=n_chans,
                                 n_classes=n_classes,
                                 input_window_samples=input_window_samples,
                                 params_path=model_load_path + params_name)
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

    # Split train_set to valid and train
    X_train, X_valid = train_test_split(train_set_all.datasets, test_size=1, train_size=5)
    train_set = BaseConcatDataset(X_train)
    valid_set = BaseConcatDataset(X_valid)

    train_set_fake.append(train_set)
    X = BaseConcatDataset(train_set_fake)

    clf = tl_classifier(X,
                        valid_set,
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
        test[i] = x
        target[i] = y
        i += 1

    score = clf.score(test, y=target)
    print("EEG Final Classification Score (Accuracy) is:  " + str(score))

