import os
import numpy as np

from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss

from Code.Tests.base import *


def test_clf(data_load_path, clf_load_path):

    dataset = load_data_object(data_load_path)
    n_classes = 4
    n_chans = dataset[0][0].shape[0]
    input_window_samples = 1000

    batch_size = 64
    n_epochs = 100
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    cuda, device = detect_device()
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

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

    to_dense_prediction_model(model)
    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]
    trial_start_offset_seconds = -0.5

    windows_dataset = cut_compute_windows(dataset,
                                          n_preds_per_input,
                                          input_window_samples=input_window_samples,
                                          trial_start_offset_seconds=trial_start_offset_seconds)

    test_set_win = get_test_data(windows_dataset)

    # Calculate Mean Accuracy For Test set
    i = 0
    test_set = np.empty(shape=(len(test_set_win), n_chans, input_window_samples))
    target = np.empty(shape=(len(test_set_win)))
    for x, y, window_ind in test_set_win:
        test_set[i] = x
        target[i] = y
        i += 1

    clf = EEGClassifier(
        model,
        cropped=True,
        max_epochs=n_epochs,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        device=device,
    )

    clf.initialize()  # This is important!
    clf.load_params(f_params=clf_load_path + 'params.pt', f_optimizer=clf_load_path + 'optimizers.pt')

    score = clf.score(test_set, y=target)
    print("EEG Classification Score (Accuracy) is:  " + str(score))


########################################
#   Test Cropped And Fake Classification
########################################

subject_id_list = [1]
data_load_path = os.path.join('../../Data/Real_Data/BCI/bnci-raw/' + str(subject_id_list).strip('[]')) + '/'

clf_load_path = os.path.join('../../Model_Params/BCI_Models/deep4/' + str(subject_id_list).strip('[]')) + '/'
# clf_load_path = os.path.join('../../Model_Params/Fake_Cropped_Classification/' + str(subject_id_list).strip('[]')) + '/'

test_clf(data_load_path=data_load_path, clf_load_path=clf_load_path)
