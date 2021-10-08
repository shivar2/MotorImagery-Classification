import os
import numpy as np
import torch.utils.data
from torch.autograd import Variable

import matplotlib.pyplot as plt
from braindecode.util import set_random_seeds

from Code.Models.GANs.WGanGPSignalModels import Generator


cuda = True if torch.cuda.is_available() else False
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed, cuda=cuda)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

subject_id_list = [1]
normalize_type = '-tanh/'
freq = '0-38/'
last_epoch = 1000

# mapping to HGD tasks
tasks = ['feet', 'left_hand', 'right_hand', 'tongue']
mapping = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3 }


# number of images to generate
batch_size = 24

# GAN info
sfreq = 250
time_sample = 1000
window_stride_samples = 467
noise = 100


for subject_id in subject_id_list:

    for key, value in mapping.items():
            tasks_name = key
            task_mapping = {
                key: value
            }

            # ---------------------
            #  FAKE
            # ---------------------

            # Save path
            save_fake_path = '../../Result/IMG-Fake-WGan-GP-Signal-VERSION9/' + normalize_type + \
                             str(subject_id) + '/' + str(last_epoch) + '/' + tasks_name + '/'

            if not os.path.exists(save_fake_path):
                os.makedirs(save_fake_path)

            # path to generator weights .pth file
            load_models_path = '../../Model_Params/GANs/WGan-GP-Signal-VERSION9' + normalize_type + freq +\
                               str(subject_id) + '/' + str(last_epoch) + '/' + tasks_name + '/'
            load_models_path += 'generator_state_dict.pth'

            # Create fake samples
            netG = Generator(time_sample=time_sample, noise=noise, channels=22)

            # load weights
            checkpoint_g = torch.load(load_models_path)
            netG.load_state_dict(checkpoint_g['model_state_dict'])

            if cuda:
                netG.cuda()

            # initialize noise
            z = Variable(Tensor(np.random.normal(0, 1, (batch_size, noise))))

            gen_sig = netG(z)

            # ---------------------
            #  PLOT FAKE
            # ---------------------
            fake_imgs = Variable(gen_sig, requires_grad=True)
            fake_imgs = fake_imgs.detach().cpu().numpy()

            j = 0
            for fake_img in fake_imgs:
                fig, axs = plt.subplots()
                fig.tight_layout()

                axs.imshow(fake_img, aspect='auto')
                plt.savefig("%s/%d.png" % (save_fake_path, j))
                # plt.show()
                plt.close()
                j += 1

