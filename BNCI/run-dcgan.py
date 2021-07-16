
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.serialization import load_concat_dataset

from Generators.DCGenerator import DCGenerator


def get_data(subject_id, image_size):

    # Dataset
    dataset = load_concat_dataset(
        path='../Dataset-Files/data-file/bnci-raw/' + str(subject_id),
        preload=True,
        target_name=None,

    )
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(-0.5 * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.

    data = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
        window_size_samples=image_size,
        window_stride_samples=1,
        drop_bad_windows=True,
    )

    return data

image_size = 1024
subject_id = 1

dataset = get_data(subject_id=subject_id, image_size=image_size)

net = DCGenerator(imageSize=image_size, nc=22)
net.train(dataset)
