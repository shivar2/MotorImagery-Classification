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
        model.requires_grad_(requires_grad=False)

        # first and second layer unfreezing
        model.conv_time = nn.Conv2d(1, 25, kernel_size=(10, 1), stride=(1, 1))
        model.conv_spat = nn.Conv2d(25, 25, kernel_size=(1, 22), stride=(3, 1), bias=False)
        model.bnorm = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        model.conv_nonlin = Expression(elu)
        model.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=0, dilation=1, ceil_mode=False)
        model.pool_nonlin = Expression(identity)
        model.drop_2 = nn.Dropout(p=0.5, inplace=False)
        model.conv_2 = nn.Conv2d(25, 44, kernel_size=(10, 1), stride=(3, 1), bias=False)
        model.bnorm_2 = nn.BatchNorm2d(44, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        model.nonlin_2 = Expression(elu)
        model.pool_2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=0, dilation=1, ceil_mode=False)
        model.pool_nonlin_2 = Expression(identity)

        self.model = model

    def get_named_parameters(self):
        name = []
        for p in self.model.named_parameters():
            if p[1].requires_grad:
                name.append(p)
        return name

    def forward(self, x):
        return self.model(x)