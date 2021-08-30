import torch
from torch import nn
from torch.nn.functional import elu

from braindecode.models.functions import identity

from braindecode.models import Deep4Net
from braindecode.models.modules import Expression
from braindecode.models.functions import squeeze_final_output


class PretrainedDeep4Model(nn.Module):
    def __init__(self,
                 params_path,
                 device='cpu',
                 n_chans=22,
                 n_classes=4,
                 input_window_samples=1000
                 ):
        super().__init__()
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
        # Load model
        state_dict = torch.load(params_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

        # Freezing model
        # model.requires_grad_(requires_grad=False)

        # Final_conv_length
        final_conv_length = model.final_conv_length

        # Change conv_classifier layer to fine-tune

        model.conv_classifier = nn.Conv2d(
            int(n_chans * (2 ** 3.0)),
            n_classes,
            (final_conv_length, 1),
            stride=(1, 1),
            bias=True)

        model.softmax = nn.LogSoftmax(dim=1)
        model.squeeze = Expression(squeeze_final_output)
        self.model = model

    def get_named_parameters(self):
        name = []
        for p in self.model.named_parameters():
            if p[1].requires_grad:
                name.append(p)
        return name

    def forward(self, x):
        return self.model(x)