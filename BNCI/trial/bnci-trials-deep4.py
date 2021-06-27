# subjects
subject_id_list = [1]


# load data
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.datasets.base import BaseConcatDataset

datasets = []
for subject_id in subject_id_list:
    datasets.append(
            load_concat_dataset(
            path='../../data-file/bnci-raw/' + str(subject_id),
            preload=True,
            target_name=None,
            )
    )
dataset = BaseConcatDataset(datasets)


# Cut Compute Windows
# for trials
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
    preload=True,
)

# Split dataset into train and valid
splitted = windows_dataset.split('session')
train_set = splitted['session_T']
valid_set = splitted['session_E']

# Create model
import torch
from braindecode.util import set_random_seeds
from braindecode.models import Deep4Net

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

n_classes=4
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = train_set[0][0].shape[1]

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

# Send model to GPU
if cuda:
    model.cuda()

# Training
from skorch.callbacks import LRScheduler, Checkpoint, EarlyStopping
from skorch.helper import predefined_split
from braindecode import EEGClassifier

# For deep4 they should be:
lr = 1 * 0.01
weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 10

# Path to saving models
path = '../../saved_models/BNCI/trials/deep4/'
f_params = str(subject_id_list).strip('[]') + '.pt'

# Checkpoint will save the model with the lowest valid_loss
cp = Checkpoint(dirname=path, f_params=f_params, f_criterion=None, f_optimizer=None)

# Early_stopping
early_stopping = EarlyStopping(patience=5)

callbacks = [
    "accuracy",
    ('cp', cp),
    ('patience', early_stopping),
    ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
]

clf = EEGClassifier(
    model,
    max_epochs=n_epochs,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=callbacks,
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(train_set, y=None)

# clf.load_params(checkpoint=cp)  # Load the model with the lowest valid_loss

# Plot Results
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

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
image_path = path + str(subject_id_list).strip('[]')
plt.savefig(fname=image_path)
