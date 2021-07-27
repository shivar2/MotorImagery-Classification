
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.serialization import load_concat_dataset

from models.DCGanErik import DCGAN


def get_data(subject_id=1, time_sample=32):

    # Dataset
    dataset = load_concat_dataset(
        path='../Dataset-Files/data-file/bnci-3channels-raw/' + str(subject_id),
        preload=True,
        target_name=None,

    )
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    trial_start_offset_samples = int(-0.5 * sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
        window_size_samples=time_sample,
        window_stride_samples=1,
        drop_bad_windows=True,
    )
    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']
    return train_set


time_sample = 1000
subject_id = 1

dataset = get_data(subject_id=subject_id, time_sample=time_sample)

net = DCGAN(time_sample=time_sample, channels=3)
net.train(dataset)
