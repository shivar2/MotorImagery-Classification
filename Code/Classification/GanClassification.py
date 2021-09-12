import numpy as np
import torch

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode.datasets.base import BaseConcatDataset
from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss

from Code.Classification.CroppedClassification import plot
from Code.base import detect_device, load_data_object,\
    load_fake_data, cut_compute_windows, create_model_deep4,\
    split_into_train_valid, get_test_data


def train_cropped_trials(train_set, valid_set, model, device='cpu'):
    # For deep4 they should be:
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 20

    callbacks = [
        "accuracy",
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ]

    clf = EEGClassifier(
        model,
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
    clf.fit(train_set, y=None)
    return clf


def run_model(data_load_path, fake_data_load_path, fake_k, save_path):
    input_window_samples = 1000
    cuda, device = detect_device()

    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    dataset = load_data_object(data_load_path)
    train_set_fake = load_fake_data(fake_data_load_path, fake_k)

    n_classes = 4
    n_chans = 22

    model = create_model_deep4(input_window_samples, n_chans, n_classes)

    # Send model to GPU
    if cuda:
        model.cuda()

    # And now we transform model with strides to a model that outputs dense prediction,
    # so we can use it to obtain predictions for all crops.
    to_dense_prediction_model(model)

    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    trial_start_offset_seconds = -0.5

    windows_dataset = cut_compute_windows(dataset,
                                          n_preds_per_input,
                                          input_window_samples=input_window_samples,
                                          trial_start_offset_seconds=trial_start_offset_seconds)

    train_set, test_set = split_into_train_valid(windows_dataset, use_final_eval=True)

    train_set_fake.append(train_set)
    X = BaseConcatDataset(train_set_fake)

    clf = train_cropped_trials(X,
                               test_set,
                               model=model,
                               device=device)

    plot(clf, save_path)
    torch.save(model, save_path + "model.pth")

    # Calculate Mean Accuracy For Test set
    i = 0
    test = np.empty(shape=(len(test_set), n_chans, input_window_samples))
    target = np.empty(shape=(len(test_set)))
    for x, y, window_ind in test_set:
        test[i] = x
        target[i] = y
        i += 1
    score = clf.score(test, y=target)
    print("EEG Gan Classification Score (Accuracy) is:  " + str(score))

    f = open(save_path + "test-result.txt", "a")
    f.write("EEG Fake Classification Score (Accuracy) is:  " + str(score))
    f.close()

