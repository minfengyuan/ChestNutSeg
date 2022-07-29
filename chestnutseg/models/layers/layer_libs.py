import torch
import torch.nn as nn

from chestnutseg.models.layers import Activation


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 act: str = None):
        super(ConvBNAct, self).__init__()

        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2 \
                if dilation == 1 else dilation,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=False)

        self._batch_norm = nn.BatchNorm2d(out_channels)
        self._act_op = Activation(act=act)

    def forward(self, x):
        y = self._conv(x)
        y = self._batch_norm(y)
        y = self._act_op(y)
        return y
