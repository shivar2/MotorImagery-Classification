from Classifier.EEGTLClassifier import EEGTLClassifier
from Classification.cropped import *


def tl_classifier(train_set, valid_set, model, save_path, double_channel=True, model_name='deep4', device='cpu'):
    if model_name == 'shallow':
            # These values we found good for shallow network:
            lr = 0.0625 * 0.01
            weight_decay = 0
    else:
            # For deep4 they should be:
            lr = 1 * 0.01
            weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 10

    # Checkpoint will save the model with the lowest valid_loss
    cp = Checkpoint(dirname=save_path, f_criterion=None)

    # Early_stopping
    early_stopping = EarlyStopping(patience=5)

    callbacks = [
        "accuracy",
        ('cp', cp),
        ('patience', early_stopping),
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ]

    clf = EEGTLClassifier(
        model,
        double_channel=double_channel,
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
        model = create_model_shallow(input_window_samples, n_chans, n_classes)
    else:
        model = create_model_deep4(input_window_samples, n_chans, n_classes)

    # Load model
    state_dict = torch.load(load_path + 'params_6.pt', map_location=device)
    model.load_state_dict(state_dict, strict=False)

    # Freezing model
    model.requires_grad_(requires_grad=False)

    # Final_conv_length
    final_conv_length = model.final_conv_length

    # Change conv_classifier layer to fine-tune
    model.conv_classifier = torch.nn.Conv2d(int(n_chans * (2 ** 3.0)), n_classes, (final_conv_length, 1),
                                            stride=(1, 1), bias=True)

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

