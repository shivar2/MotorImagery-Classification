from Classifier.EEGTLClassifier import EEGTLClassifier
from Classification.ganClassification import *
from models.PretrainedDeep4Model import PretrainedDeep4Model


def tl_classifier(train_set, valid_set,
                  save_path,
                  model,
                  double_channel=True,
                  device='cpu'):
    
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
    clf.fit(train_set, y=None)
    return clf


def run_model(subject_id_list, double_channel, load_path, save_path):

    input_window_samples = 1000
    cuda, device = detect_device()

    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    dataset = load_data_object(subject_id_list)
    train_set_fake = load_fake_data(subject_id_list)

    n_classes = 4

    if double_channel:
        n_chans = dataset[0][0].shape[0] * 2
    else:
        n_chans = dataset[0][0].shape[0]

    model = PretrainedDeep4Model(n_chans=n_chans,
                                 n_classes=n_classes,
                                 input_window_samples=input_window_samples,
                                 params_path=load_path + 'params_10.pt')
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

    train_set, valid_set = split_data(windows_dataset)

    train_set_fake.append(train_set)
    X = BaseConcatDataset(train_set_fake)

    clf = tl_classifier(X,
                        valid_set,
                        model=model,
                        save_path=save_path,
                        double_channel=double_channel,
                        device=device)

    plot(clf, save_path)
