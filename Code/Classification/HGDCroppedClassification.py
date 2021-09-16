import numpy as np
import torch

from skorch.callbacks import LRScheduler, Checkpoint
from skorch.helper import predefined_split
from torch.utils.data import Subset

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from braindecode.datautil.serialization import load_concat_dataset
from braindecode.datasets.base import BaseConcatDataset
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.datautil.windowers import create_windows_from_events
from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss


from braindecode.datautil.preprocess import preprocess, Preprocessor
from Code.Preprocess import MaxNormalize


def detect_device():
    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return cuda, device


def load_data_object(data_path):
    subject_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    dataset_all = []

    for subject in subject_id_list:
        dataset = load_concat_dataset(
            path=data_path + str(subject) + '/',
            preload=False,
            target_name=None,)
        dataset_all.append(dataset)
        del dataset

    dataset_obj = BaseConcatDataset(dataset_all)

    return dataset_obj


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
        preload=False,
        mapping={'left_hand': 0, 'right_hand': 1, 'feet': 2, 'rest': 3},
    )
    # max normalize
    preprocess(windows_dataset, [Preprocessor(MaxNormalize)])

    return windows_dataset


def split_into_train_valid(windows_dataset, use_final_eval):
    splitted = windows_dataset.split('run')
    if use_final_eval:
        train_set = splitted['train']
        valid_set = splitted['test']
    else:
        full_train_set = splitted['train']
        n_split = int(np.round(0.8 * len(full_train_set)))
        # ensure this is multiple of 2 (number of windows per trial)
        n_windows_per_trial = 2  # here set by hand
        n_split = n_split - (n_split % n_windows_per_trial)
        valid_set = Subset(full_train_set, range(n_split, len(full_train_set)))
        train_set = Subset(full_train_set, range(0, n_split))
    return train_set, valid_set


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
    n_epochs = 35

    # Checkpoint will save the model with the lowest valid_loss
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
    # Model training for a specified number of epochs. `y` is None as it is already supplied
    # in the dataset.
    clf.fit(train_set, y=None)
    return clf


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


def run_model(data_load_path, model_name, save_path):
    input_window_samples = 1000
    cuda, device = detect_device()

    seed = 20200220  # random seed to make results reproducible
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    dataset = load_data_object(data_load_path)

    n_classes = 4
    # Extract number of chans and time steps from dataset
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

    # To know the Models’ receptive field, we calculate the shape of model output for a dummy input.
    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    trial_start_offset_seconds = -0.5

    windows_dataset = cut_compute_windows(dataset,
                                          n_preds_per_input,
                                          input_window_samples=input_window_samples,
                                          trial_start_offset_seconds=trial_start_offset_seconds)

    train_set, valid_set = split_into_train_valid(windows_dataset, use_final_eval=True)

    clf = train_cropped_trials(train_set,
                               valid_set,
                               model=model,
                               save_path=save_path,
                               model_name=model_name,
                               device=device)

    plot(clf, save_path)


