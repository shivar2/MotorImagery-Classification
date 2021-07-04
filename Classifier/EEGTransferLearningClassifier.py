import numpy as np
from skorch.dataset import uses_placeholder_y, unpack_data, get_len
from skorch.utils import to_numpy

from braindecode import EEGClassifier


class EEGTransferLearningClassifier(EEGClassifier):
    def __init__(self, *args, cropped=False, callbacks=None,
                 iterator_train__shuffle=True, **kwargs):
        self.cropped = cropped
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
