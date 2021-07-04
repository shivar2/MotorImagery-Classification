# subjects
subject_id_list = [1]

# Path to saving models
# mkdir path to save
import os
save_path = os.path.join('../../saved_models/BNCI/cropped/deep4/TL/' + str(subject_id_list).strip('[]')) + '/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

load_path = os.path.join('../../saved_models/HGD/selected_channels/cropped/deep4/1/')


# load data
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.datasets.base import BaseConcatDataset


datasets = []
for subject_id in subject_id_list:
    data = load_concat_dataset(
            path='../../data-file/bnci-raw/' + str(subject_id),
            preload=True,
            target_name=None,
            )
    datasets.append(data)

dataset = BaseConcatDataset(datasets)


# *input window samples*
input_window_samples = 1000


# Create model

import torch
from braindecode.util import set_random_seeds
from braindecode.models import Deep4Net

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 20200220  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

n_classes=4
# HGD selected channels has 44 channel ( motor imagery channels)
n_chans = 44

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

state_dict = torch.load(load_path+'params.pt', map_location=device)
model.load_state_dict(state_dict, strict=False)

# model.requires_grad_(requires_grad=False)
# model.conv_classifier = torch.nn.Conv2d(1024, 4, (5, 1), (1, 1), bias=True)

# Send model to GPU
if cuda:
    model.cuda()

# And now we transform model with strides to a model that outputs dense prediction,
# so we can use it to obtain predictions for all crops.

from braindecode.models.util import to_dense_prediction_model, get_output_shape

to_dense_prediction_model(model)

# To know the models’ receptive field, we calculate the shape of model output for a dummy input.

n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]


# Cut Compute Windows
from braindecode.datautil.windowers import create_windows_from_events

trial_start_offset_seconds = -0.5
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

# Split dataset into train and valid
splitted = windows_dataset.split('session')
train_set = splitted['session_T']
valid_set = splitted['session_E']


# Training
# train model for cropped trials

from skorch.callbacks import LRScheduler, Checkpoint, EarlyStopping
from skorch.helper import predefined_split

from braindecode.training.losses import CroppedLoss

from Classifier.EEGTransferLearningClassifier import EEGTransferLearningClassifier

# For deep4 they should be:
lr = 1 * 0.01
weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 10

# Checkpoint will save the model with the lowest valid_loss
cp = Checkpoint(dirname=save_path, f_criterion=None)

# Early_stopping
early_stopping = EarlyStopping(patience=5)

callbacks = [
    "accuracy",
    ('cp', cp),
    ('patience', early_stopping),
    ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
]

clf = EEGTransferLearningClassifier(
    model,
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
# Model training for a specified number of epochs. `y` is None as it is already supplied in the dataset.
clf.fit(train_set, y=None)


# Plot Results
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Extract loss and accuracy values for plotting from history object
results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns, index=clf.history[:, 'epoch'])

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
