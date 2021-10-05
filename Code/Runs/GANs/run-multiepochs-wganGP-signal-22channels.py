import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.util import set_random_seeds

from Code.Models.GANs.WGanGPErikSignal import WGANGP


def get_data(data_load_path,
             normalize_type='-zmax/',
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
        trial_start_offset_samples=0,
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


#########################
# MAIN                  #
#########################
cuda = True if torch.cuda.is_available() else False
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed, cuda=cuda)

mapping = {
    'left_hand': 0,
    # 'right_hand': 1,
    # 'feet': 2,
    # 'tongue': 3
           }
all_channels = ['Fz',
                'FC1', 'FC2',
                'C3', 'Cz', 'C4', 'CP1', 'CP2',
                'Pz', 'POz', 'FC3', 'FCz', 'FC4',
                'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4',
                'P1', 'P2']

time_sample = 1000
window_stride_samples = 1000

batchsize = 64
epochs = 500
epak_limit = 15

normalize_type = '-zmax/'   # '-zmax'
freq = '0-f/'

subject_id = 1
data_load_path = os.path.join('../../../Data/Real_Data/BCI/bnci-raw/0-f/22channels' +
                              normalize_type +
                              str(subject_id)) + '/'

for key, value in mapping.items():
        gloss, dloss = [], []
        d_tot, g_tot = [], []

        tasks_name = key
        task_mapping = {
                key: value
        }
        #########################
        # Load data            #
        #########################
        data, n_chans = get_data(data_load_path=data_load_path,
                                 time_sample=time_sample,
                                 window_stride_samples=window_stride_samples,
                                 mapping=task_mapping,
                                 pick_channels=all_channels
                                 )

        for epak in range(0, epak_limit):
            d_tot_epak, g_tot_epak = [], []
            last_epoch = epochs * epak

            save_model_path = '../../../Model_Params/GANs/WGan-GP-Signal-VERSION7' + normalize_type +freq + str(
                    subject_id) + '/' + str(last_epoch + epochs) + '/' + tasks_name + '/'

            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)

            #########################
            # Running params        #
            #########################

            net = WGANGP(subject=subject_id,
                         n_epochs=epochs,
                         batch_size=batchsize,
                         time_sample=time_sample,
                         channels=n_chans,
                         sample_interval=400,
                         )

            if epak > 0:
                ##################################
                # Load G and D model and optimizer
                ##################################
                load_model_path = '../../../Model_Params/GANs/WGan-GP-Signal-VERSION7' + normalize_type + freq + str(
                        subject_id) + '/' + str(last_epoch) + '/' + tasks_name + '/'

                checkpoint_g = torch.load(load_model_path + 'generator_state_dict.pth')
                net.generator.load_state_dict(checkpoint_g['model_state_dict'])
                net.optimizer_G.load_state_dict(checkpoint_g['optimizer_state_dict'])
                gloss = checkpoint_g['loss']

                checkpoint_d = torch.load(load_model_path + 'discriminator_state_dict.pth')
                net.discriminator.load_state_dict(checkpoint_d['model_state_dict'])
                net.optimizer_D.load_state_dict(checkpoint_d['optimizer_state_dict'])
                dloss = checkpoint_d['loss']

            d_tot_epak, g_tot_epak = net.train(data, save_model_path=save_model_path,
                                               disc_loss=dloss, gen_loss=gloss,
                                               last_epoch=last_epoch)
            d_tot.extend(d_tot_epak)
            g_tot.extend(g_tot_epak)

        # ---------------------
        #  PLOT for each subject & each task - Final Result
        # ---------------------
        save_final_result_path = '../../../Result/GANs/WGan-GP-Signal-VERSION7' + normalize_type + freq + str(
                subject_id) + '/FinalResult/' + tasks_name + '/'
        if not os.path.exists(save_final_result_path):
            os.makedirs(save_final_result_path)

        # Plot the generator and discriminator losses for all the epochs
        plt.figure()
        plt.plot(g_tot, 'r')
        plt.plot(d_tot, 'b')
        plt.title('Loss history')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Generator', 'Discriminator'])
        plt.grid()
        plt.savefig("%s/%s-.png" % (save_final_result_path, 'results-plot'))
        # plt.show()
        plt.close()

print("end")
