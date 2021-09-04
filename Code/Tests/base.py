
import torch

from braindecode.datautil.serialization import load_concat_dataset
from braindecode.models import Deep4Net
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


def cut_compute_windows(dataset, n_preds_per_input, input_window_samples=1000, trial_start_offset_seconds=-0.5):
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Mapping new event ids to fit hgd event ids
    mapping = {
        'feet': 0,
        'left_hand': 1,
        'tongue': 2,
        'right_hand': 3,
    }
    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        preload=True,
        mapping=mapping
    )
    return windows_dataset


def get_test_data(windows_dataset):
    # Split dataset into train and test and return just test set
    splitted = windows_dataset.split('session')
    test_set = splitted['session_E']

    return test_set

