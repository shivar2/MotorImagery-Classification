import os
import numpy as np
import torch.utils.data

import matplotlib.pyplot as plt

from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.util import set_random_seeds


cuda = True if torch.cuda.is_available() else False
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed, cuda=cuda)


def get_data(data_load_path,
             time_sample=32,
             window_stride_samples=1,
             mapping=None,
             pick_channels=None):
    # Dataset
    dataset = load_concat_dataset(
        path=data_load_path,
        preload=False,
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
        window_stride_samples=window_stride_samples,
        drop_bad_windows=True,
        picks=pick_channels,
        mapping=mapping,
    )

    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']
    n_chans = windows_dataset[0][0].shape[0]

    i = 0
    events_num = train_set.datasets[0].windows.events.shape[0]
    runs_num = len(train_set.datasets)
    epochs_num = events_num * runs_num

    data = np.empty(shape=(epochs_num, n_chans, time_sample))
    for x, y, window_ind in train_set:
        data[i] = x
        i += 1
    return data, n_chans


subject_id_list = [2]
normalizer_name = '-zmax/'       # 'tanhNormalized/'

# mapping to HGD tasks
tasks = ['feet', 'left_hand', 'right_hand', 'tongue']
mapping = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3 }


# number of images to generate
batch_size = 24

# GAN info
sfreq = 250
time_sample = 500
window_stride_samples = 500
noise = 100

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for subject_id in subject_id_list:

    for key, value in mapping.items():
            tasks_name = key
            task_mapping = {
                key: value
            }
            # ---------------------
            #  REAL
            # ---------------------

            # Save path
            save_real_path = '../../Result/IMG-REAL' + normalizer_name + str(subject_id) + \
                             '/' + tasks_name + '/'

            if not os.path.exists(save_real_path):
                os.makedirs(save_real_path)

            # Load real data
            data_load_path = os.path.join(
                '../../Data/Real_Data/BCI/bnci-raw/0-38/22channels' + normalizer_name + str(subject_id)) + '/'

            data, n_chans = get_data(data_load_path=data_load_path,
                                     time_sample=time_sample,
                                     window_stride_samples=window_stride_samples,
                                     mapping=task_mapping,
                                     )

            # ---------------------
            #  PLOT REAL
            # ---------------------
            for i in range(0, batch_size):
                real_img = data[i]
                fig, axs = plt.subplots()
                # fig.tight_layout()

                axs.imshow(real_img, aspect='auto')
                plt.savefig("%s/%d.png" % (save_real_path, i))
                # plt.show()
                plt.close()
