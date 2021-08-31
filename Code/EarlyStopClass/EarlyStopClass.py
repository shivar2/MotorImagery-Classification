""" Callbacks related to training progress. """

import os
import pickle
import warnings
from contextlib import suppress
from fnmatch import fnmatch
from functools import partial
from itertools import product

import numpy as np
from skorch.callbacks import Callback
from skorch.exceptions import SkorchException
from skorch.utils import _check_f_arguments
from skorch.utils import noop
from skorch.utils import open_file_like
from skorch.utils import freeze_parameter
from skorch.utils import unfreeze_parameter

class EarlyStopping(Callback):
    """Callback for stopping training when scores don't improve.

    Stop training early if a specified `monitor` metric did not
    improve in `patience` number of epochs by at least `threshold`.

    Parameters
    ----------
    monitor : str (default='valid_loss')
      Value of the history to monitor to decide whether to stop
      training or not.  The value is expected to be double and is
      commonly provided by scoring callbacks such as
      :class:`skorch.callbacks.EpochScoring`.

    lower_is_better : bool (default=True)
      Whether lower scores should be considered better or worse.

    patience : int (default=5)
      Number of epochs to wait for improvement of the monitor value
      until the training process is stopped.

    threshold : int (default=1e-4)
      Ignore score improvements smaller than `threshold`.

    threshold_mode : str (default='rel')
        One of `rel`, `abs`. Decides whether the `threshold` value is
        interpreted in absolute terms or as a fraction of the best
        score so far (relative)

    sink : callable (default=print)
      The target that the information about early stopping is
      sent to. By default, the output is printed to stdout, but the
      sink could also be a logger or :func:`~skorch.utils.noop`.

    """
    def __init__(
            self,
            monitor='valid_loss',
            patience=5,
            threshold=1e-4,
            stopping_threshold=0,
            divergence_threshold=0,
            threshold_mode='rel',
            lower_is_better=True,
            sink=print,
    ):
        self.monitor = monitor
        self.lower_is_better = lower_is_better
        self.patience = patience
        self.threshold = threshold
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.threshold_mode = threshold_mode
        self.misses_ = 0
        self.dynamic_threshold_ = None
        self.sink = sink

        self.stop_next_epoch = 0

    # pylint: disable=arguments-differ
    def on_train_begin(self, net, **kwargs):
        if self.threshold_mode not in ['rel', 'abs']:
            raise ValueError("Invalid threshold mode: '{}'"
                             .format(self.threshold_mode))
        self.misses_ = 0
        self.dynamic_threshold_ = np.inf if self.lower_is_better else -np.inf

    def on_epoch_end(self, net, **kwargs):
        current_score = net.history[-1, self.monitor]
        if self.stop_next_epoch == 1 :
            if net.verbose:
                self._sink("Stop reach the stopping threshold".format(self.monitor, self.patience),
                           verbose=net.verbose)
                raise KeyboardInterrupt

        if current_score == self.stopping_threshold:
            self.stop_next_epoch = 1
        if current_score < self.divergence_threshold:
            self.stop_next_epoch = 1

        if not self._is_score_improved(current_score):
            self.misses_ += 1
        else:
            self.misses_ = 0
            self.dynamic_threshold_ = self._calc_new_threshold(current_score)
        if self.misses_ == self.patience:
            if net.verbose:
                self._sink("Stopping since {} has not improved in the last "
                           "{} epochs.".format(self.monitor, self.patience),
                           verbose=net.verbose)
            raise KeyboardInterrupt

    def _is_score_improved(self, score):
        if self.lower_is_better:
            return score < self.dynamic_threshold_
        return score > self.dynamic_threshold_

    def _calc_new_threshold(self, score):
        """Determine threshold based on score."""
        if self.threshold_mode == 'rel':
            abs_threshold_change = self.threshold * score
        else:
            abs_threshold_change = self.threshold

        if self.lower_is_better:
            new_threshold = score - abs_threshold_change
        else:
            new_threshold = score + abs_threshold_change
        return new_threshold

    def _sink(self, text, verbose):
        #  We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)
