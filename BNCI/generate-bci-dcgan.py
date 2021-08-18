
import os
import numpy as np
import matplotlib.pyplot as plt

import torch.utils.data
from torch.autograd import Variable

import models.DCModels as dcgan

subject_id = 1
task = 1

# number of images to generate
nimages = 1

# path to generator weights .pth file
weights = '../saved_models/DCGan/' + str(subject_id) + '/generator_state_dict.pth'

# path to to output directory
output_dir = '../Dataset-Files/fake-data/WGan-GP-Signal' + str(subject_id) + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


channels = 2
time_sample = 500
freq_sample = 36
noise = 100

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

netG = dcgan.Generator(time_sample=time_sample, noise=noise, channels=channels, freq_sample=freq_sample)

# load weights -tl
netG.load_state_dict(torch.load(weights))

if cuda:
    netG.cuda()

# initialize noise
z = Variable(Tensor(np.random.normal(0, 1, (nimages, noise))))

gen_imgs = netG(z)
# fake.data = fake.data.mul(0.5).add(0.5)         # why?

# ---------------------
#  PLOT
# ---------------------

# from https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/WasserGAN_Final.py

# Plots the generated samples for the selected channels.
# Recall the channels are chosen during the Load_and_Preprocess Script

# Here they just correspond to C3 only (channel 7 was selected).
channel = 0

fig, axs = plt.subplots(1, 1)
fig.suptitle('Fake signal for subject ' + str(subject_id))
fig.tight_layout()

gen_imgs = Variable(gen_imgs, requires_grad=True)
gen_imgs = gen_imgs.detach().cpu().numpy()

axs.imshow(gen_imgs[0, :, :, channel], aspect='auto')
axs.set_title('Generated Signal', size=10)
axs.set_xlabel('Time Sample')
axs.set_ylabel('Frequency Sample')

# Save the generated samples within the current working dir
# in a folder called 'EEG Samples', every 100 epochs.

plt.savefig("%s/%d.png" % (output_dir, subject_id))
plt.show()
plt.close()

fp = os.path.join(os.getcwd(), 'EEG_Samples')
sp = os.path.join(fp, 'Subject{}WGAN_Model_Data_For_Task{}.h5'.format(subject_id, task))
netG.save(sp)
