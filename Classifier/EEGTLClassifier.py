import numpy as np

from skorch.dataset import uses_placeholder_y, unpack_data, get_len
from skorch.setter import optimizer_setter

from braindecode.classifier import EEGClassifier


class EEGTLClassifier(EEGClassifier):
    def __init__(self, *args,
                 cropped=False,
                 callbacks=None,
                 iterator_train__shuffle=True,
                 double_channel=False,
                 is_freezing=True,
                 **kwargs):

        self.double_channel = double_channel
        self.is_freezing = is_freezing

        super().__init__(*args,
                         cropped=cropped,
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
            named_parameters = []
            for p in self.module_.named_parameters():
                if p[1].requires_grad:
                    named_parameters.append(p)

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
