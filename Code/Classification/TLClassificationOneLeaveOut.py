from braindecode.datautil.serialization import load_concat_dataset

from Code.Classifier.EEGTLClassifier import EEGTLClassifier
from Code.Classification.CroppedClassification import *
from Code.Models.PretrainedDeep4Model import PretrainedDeep4Model


def load_data_object(data_folder_path, subject_ids):

    datasets = []
    for subject_id in subject_ids:
        dataset = load_concat_dataset(
                                path=data_folder_path + str(subject_id) + '/',
                                preload=True,
                                target_name=None,)
        datasets.append(dataset)

    merge_dataset = BaseConcatDataset(datasets)
    return merge_dataset


def tl_classifier(train_set, valid_set,
                  save_path,
                  model,
                  double_channel=True,
                  device='cpu'):
    
    # For deep4 they should be:
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 15

    # Checkpoint will save the history 
    cp = Checkpoint(monitor=None,
                    f_params=None,
                    f_optimizer=None,
                    f_criterion=None,
                    f_history='history.json',
                    dirname=save_path)

    # Early_stopping
    early_stopping = EarlyStopping(patience=30)

    callbacks = [
        "accuracy",
        ('cp', cp),
        ('patience', early_stopping),
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ]

    clf = EEGTLClassifier(
        model,
        double_channel=double_channel,
        is_freezing=True,
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
    # Model training for a specified number of epochs. `y` is None as it is already supplied
    # in the dataset.
    clf.fit(train_set, y=None)
    return clf


def run_model(data_load_path, subject_id_train_list, subject_id_test_list,
              double_channel, model_load_path, params_name, save_path):

    input_window_samples = 1000
    cuda, device = detect_device()

    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    dataset_train = load_data_object(data_load_path, subject_id_train_list)
    dataset_test = load_data_object(data_load_path, subject_id_test_list)

    n_classes = 4
    # Extract number of chans and time steps from dataset
    if double_channel:
        n_chans = dataset_train[0][0].shape[0] * 2
    else:
        n_chans = dataset_train[0][0].shape[0]

    model = PretrainedDeep4Model(n_chans=n_chans,
                                 n_classes=n_classes,
                                 input_window_samples=input_window_samples,
                                 params_path=model_load_path + params_name)
    # Send model to GPU
    if cuda:
        model.cuda()

    to_dense_prediction_model(model)
    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    trial_start_offset_seconds = -0.5

    windows_dataset_train = cut_compute_windows(dataset_train,
                                                n_preds_per_input,
                                                input_window_samples=input_window_samples,
                                                trial_start_offset_seconds=trial_start_offset_seconds)

    windows_dataset_test = cut_compute_windows(dataset_test,
                                               n_preds_per_input,
                                               input_window_samples=input_window_samples,
                                               trial_start_offset_seconds=trial_start_offset_seconds)

    # train_set, valid_set = split_data(windows_dataset)

    clf = tl_classifier(windows_dataset_train,
                        windows_dataset_test,
                        model=model,
                        save_path=save_path,
                        double_channel=double_channel,
                        device=device)

    plot(clf, save_path)

