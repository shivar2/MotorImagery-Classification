import numpy as np
from sklearn.metrics import get_scorer
from skorch.callbacks import EpochTimer, BatchScoring, PrintLog, EpochScoring
from skorch.classifier import NeuralNet
from skorch.classifier import NeuralNetClassifier
from skorch.utils import train_loss_score, valid_loss_score, noop, to_numpy
from skorch.dataset import uses_placeholder_y, unpack_data, get_len
from skorch.setter import optimizer_setter

from braindecode.training.scoring import PostEpochTrainScoring, CroppedTrialEpochScoring
from braindecode.util import ThrowAwayIndexLoader


class EEGTLClassifier(NeuralNetClassifier):
    def __init__(self, *args,
                 cropped=False,
                 callbacks=None,
                 iterator_train__shuffle=True,
                 double_channel=False,
                 is_freezing=True,
                 **kwargs):

        self.cropped = cropped
        self.double_channel = double_channel
        self.is_freezing = is_freezing

        callbacks = self._parse_callbacks(callbacks)

        super().__init__(*args,
                         callbacks=callbacks,
                         iterator_train__shuffle=iterator_train__shuffle,
                         **kwargs)

    def run_single_epoch(self, dataset, training, prefix, step_fn, **fit_params):
        """Compute a single epoch of train or validation.

        Parameters
        ----------
        dataset : torch Dataset
            The initialized dataset to loop over.

        training : bool
            Whether to set the module to train mode or not.

        prefix : str
            Prefix to use when saving to the history.

        step_fn : callable
            Function to call for each batch.

        **fit_params : dict
            Additional parameters passed to the ``step_fn``.
        """
        is_placeholder_y = uses_placeholder_y(dataset)

        batch_count = 0
        for data in self.get_iterator(dataset, training=training):
            Xi, yi = unpack_data(data)
            if self.double_channel:
                Xi = np.repeat(Xi, 2, 1)        # change channel number (22 to 44)
            yi_res = yi if not is_placeholder_y else None
            self.notify("on_batch_begin", X=Xi, y=yi_res, training=training)
            step = step_fn(Xi, yi, **fit_params)
            self.history.record_batch(prefix + "_loss", step["loss"].item())
            self.history.record_batch(prefix + "_batch_size", get_len(Xi))
            self.notify("on_batch_end", X=Xi, y=yi_res, training=training, **step)
            batch_count += 1

        self.history.record(prefix + "_batch_count", batch_count)

    def predict_with_window_inds_and_ys(self, dataset):
        preds = []
        i_window_in_trials = []
        i_window_stops = []
        window_ys = []
        for X, y, i in self.get_iterator(dataset, drop_index=False):
            i_window_in_trials.append(i[0].cpu().numpy())
            i_window_stops.append(i[2].cpu().numpy())
            if self.double_channel:
                X = np.repeat(X, 2, 1)          # change channel number (22 to 44)
            preds.append(to_numpy(self.forward(X)))
            window_ys.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        i_window_in_trials = np.concatenate(i_window_in_trials)
        i_window_stops = np.concatenate(i_window_stops)
        window_ys = np.concatenate(window_ys)
        return dict(
            preds=preds, i_window_in_trials=i_window_in_trials,
            i_window_stops=i_window_stops, window_ys=window_ys)

    def _parse_callbacks(self, callbacks):
        callbacks_list = []
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, tuple):
                    callbacks_list.append(callback)
                else:
                    assert isinstance(callback, str)
                    scoring = get_scorer(callback)
                    scoring_name = scoring._score_func.__name__
                    assert scoring_name.endswith(
                        ('_score', '_error', '_deviance', '_loss'))
                    if (scoring_name.endswith('_score') or
                            callback.startswith('neg_')):
                        lower_is_better = False
                    else:
                        lower_is_better = True
                    train_name = f'train_{callback}'
                    valid_name = f'valid_{callback}'
                    if self.cropped:
                        # In case of cropped decoding we are using braindecode
                        # specific scoring created for cropped decoding
                        train_scoring = CroppedTrialEpochScoring(
                            callback, lower_is_better, on_train=True, name=train_name
                        )
                        valid_scoring = CroppedTrialEpochScoring(
                            callback, lower_is_better, on_train=False, name=valid_name
                        )
                    else:
                        train_scoring = PostEpochTrainScoring(
                            callback, lower_is_better, name=train_name
                        )
                        valid_scoring = EpochScoring(
                            callback, lower_is_better, on_train=False, name=valid_name
                        )
                    callbacks_list.extend([
                        (train_name, train_scoring),
                        (valid_name, valid_scoring)
                    ])

        return callbacks_list

    # pylint: disable=arguments-differ
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """Return the loss for this batch by calling NeuralNet get_loss.
        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values
        y_true : torch tensor
          True target values.
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        training : bool (default=False)
          Whether train mode should be used or not.

        """
        return NeuralNet.get_loss(self, y_pred, y_true, *args, **kwargs)

    def get_iterator(self, dataset, training=False, drop_index=True):
        iterator = super().get_iterator(dataset, training=training)
        if drop_index:
            return ThrowAwayIndexLoader(self, iterator, is_regression=False)
        else:
            return iterator

    def on_batch_end(self, net, X, y, training=False, **kwargs):
        # If training is false, assume that our loader has indices for this
        # batch
        if not training:
            cbs = self._default_callbacks + self.callbacks
            epoch_cbs = []
            for name, cb in cbs:
                if (cb.__class__.__name__ == 'CroppedTrialEpochScoring') and (
                        hasattr(cb, 'window_inds_')) and (cb.on_train == False):
                    epoch_cbs.append(cb)
            # for trialwise decoding stuffs it might also be we don't have
            # cropped loader, so no indices there
            if len(epoch_cbs) > 0:
                assert hasattr(self, '_last_window_inds')
                for cb in epoch_cbs:
                    cb.window_inds_.append(self._last_window_inds)
                del self._last_window_inds

    # Removes default EpochScoring callback computing 'accuracy' to work properly
    # with cropped decoding.
    @property
    def _default_callbacks(self):
        return [
            ("epoch_timer", EpochTimer()),
            (
                "train_loss",
                BatchScoring(
                    train_loss_score,
                    name="train_loss",
                    on_train=True,
                    target_extractor=noop,
                ),
            ),
            (
                "valid_loss",
                BatchScoring(
                    valid_loss_score, name="valid_loss", target_extractor=noop,
                ),
            ),
            ("print_log", PrintLog()),
        ]

    def predict_proba(self, X):
        """Return the output of the module's forward method as a numpy
        array. In case of cropped decoding returns averaged values for
        each trial.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored.
        If all values are relevant or module's output for each crop
        is needed, consider using :func:`~skorch.NeuralNet.forward`
        instead.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        Returns
        -------
        y_proba : numpy ndarray

        """
        y_pred = super().predict_proba(X)
        if self.cropped:
            return y_pred.mean(axis=-1)
        else:
            return y_pred

    def predict(self, X):
        """Return class labels for samples in X.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        Returns
        -------
        y_pred : numpy ndarray

        """
        return self.predict_proba(X).argmax(1)

    def initialize_optimizer(self, triggered_directly=True):
        """Initialize the model optimizer. If ``self.optimizer__lr``
        is not set, use ``self.lr`` instead.

        Parameters
        ----------
        triggered_directly : bool (default=True)
          Only relevant when optimizer is re-initialized.
          Initialization of the optimizer can be triggered directly
          (e.g. when lr was changed) or indirectly (e.g. when the
          module was re-initialized). If and only if the former
          happens, the user should receive a message informing them
          about the parameters that caused the re-initialization.

        """

        if not self.is_freezing:
            named_parameters = self.module_.named_parameters()
        else:
            named_parameters = self.module.get_named_parameters()

        args, kwargs = self.get_params_for_optimizer(
                'optimizer', named_parameters)

        if self.initialized_ and self.verbose:
            msg = self._format_reinit_msg(
                "optimizer", kwargs, triggered_directly=triggered_directly)
            print(msg)

        if 'lr' not in kwargs:
            kwargs['lr'] = self.lr

        self.optimizer_ = self.optimizer(*args, **kwargs)

        self._register_virtual_param(
            ['optimizer__param_groups__*__*', 'optimizer__*', 'lr'],
            optimizer_setter,
        )
