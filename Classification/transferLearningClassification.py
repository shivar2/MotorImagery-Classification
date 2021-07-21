from Classifier.EEGTLClassifier import EEGTLClassifier
from Classification.cropped import *
from Models.PretrainedDeep4Model import PretrainedDeep4Model


def tl_classifier(train_set, valid_set,
                  save_path,
                  model, model_name='deep4',
                  double_channel=True,
                  device='cpu'):
    
    if model_name == 'shallow':
            # These values we found good for shallow network:
            lr = 0.0625 * 0.01
            weight_decay = 0
    else:
            # For deep4 they should be:
            lr = 1 * 0.01
            weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 30

    # Checkpoint will save the history 
    cp = Checkpoint(dirname=save_path,
                    monitor=None,
                    f_params=None,
                    f_optimizer=None,
                    f_criterion=None,
                    )

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


def run_model(data_directory, subject_id_list, dataset_name, model_name, double_channel, load_path, save_path):

    input_window_samples = 1000
    cuda, device = detect_device()

    seed = 20200220  # random seed to make results reproducible
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    dataset = load_data_object(data_directory, subject_id_list)

    n_classes = 4
    # Extract number of chans and time steps from dataset
    if double_channel:
        n_chans = dataset[0][0].shape[0] * 2
    else:
        n_chans = dataset[0][0].shape[0]

    if model_name == 'shallow':
        model = PretrainedDeep4Model(n_chans=n_chans,
                                     n_classes=n_classes,
                                     input_window_samples=input_window_samples,
                                     params_path=load_path + 'params_12.pt')
    else:
        model = PretrainedDeep4Model(n_chans=n_chans,
                                     n_classes=n_classes,
                                     input_window_samples=input_window_samples,
                                     params_path=load_path + 'params_12.pt')
    # Send model to GPU
    if cuda:
        model.cuda()

    # And now we transform model with strides to a model that outputs dense prediction,
    # so we can use it to obtain predictions for all crops.
    to_dense_prediction_model(model)

    # To know the models’ receptive field, we calculate the shape of model output for a dummy input.
    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    trial_start_offset_seconds = -0.5

    windows_dataset = cut_compute_windows(dataset,
                        n_preds_per_input,
                        input_window_samples=input_window_samples,
                        trial_start_offset_seconds=trial_start_offset_seconds)

    train_set, valid_set = split_data(windows_dataset, dataset_name=dataset_name)

    clf = tl_classifier(train_set,
                        valid_set,
                        model=model,
                        save_path=save_path,
                        model_name=model_name,
                        double_channel=double_channel,
                        device=device)

    plot(clf, save_path)

