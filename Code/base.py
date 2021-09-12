import numpy as np

import torch
from torch.utils.data import Subset

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from braindecode.datautil.serialization import load_concat_dataset
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode.datautil.windowers import create_windows_from_events


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
        target_name=None,)

    return dataset


def load_fake_data(fake_data_path, fake_k):

    ds_list = []
    for folder in range(0, fake_k):
        folder_path = fake_data_path + str(folder) + '/'
        ds_loaded = load_concat_dataset(
                path=folder_path,
                preload=True,
                target_name=None,
        )
        ds_list.append(ds_loaded)

    return ds_list


def create_model_shallow(input_window_samples=1000, n_chans=4, n_classes=4):
    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=30,
    )
    return model


def create_model_deep4(input_window_samples=1000, n_chans=4, n_classes=4):
    model = Deep4Net(
        in_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=2,
    )
    return model


def cut_compute_windows(dataset, n_preds_per_input, input_window_samples=1000, trial_start_offset_seconds=-0.5):

    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        preload=True,
        mapping={'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3},
    )
    return windows_dataset


def split_into_train_valid(windows_dataset, use_final_eval):

    splitted = windows_dataset.split('session')
    if use_final_eval:
        train_set = splitted['session_T']
        valid_set = splitted['session_E']
    else:
        full_train_set = splitted['session_T']
        n_split = int(np.round(0.8 * len(full_train_set)))
        # ensure this is multiple of 2 (number of windows per trial)
        n_windows_per_trial = 2  # here set by hand
        n_split = n_split - (n_split % n_windows_per_trial)
        valid_set = Subset(full_train_set, range(n_split, len(full_train_set)))
        train_set = Subset(full_train_set, range(0, n_split))
    return train_set, valid_set


def get_test_data(windows_dataset):
    # Split dataset into train and test and return just test set
    splitted = windows_dataset.split('session')
    test_set = splitted['session_E']

    return test_set


def plot(clf, save_path):
    # Extract loss and accuracy values for plotting from history object
    results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
    df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                      index=clf.history[:, 'epoch'])

    # get percent of misclass for better visual comparison to loss
    df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
                   valid_misclass=100 - 100 * df.valid_accuracy)

    plt.style.use('seaborn')
    fig, ax1 = plt.subplots(figsize=(8, 3))
    df.loc[:, ['train_loss', 'valid_loss']].plot(
        ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
    ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    df.loc[:, ['train_misclass', 'valid_misclass']].plot(
        ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
    ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
    ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
    ax1.set_xlabel("Epoch", fontsize=14)

    # where some data has already been plotted to ax
    handles = []
    handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
    handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
    plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
    plt.tight_layout()

    # Image path
    image_path = save_path + 'result'
    plt.savefig(fname=image_path)
