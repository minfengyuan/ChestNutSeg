from typing import List, Optional, Sequence, Union

import torch.nn as nn
from torch import Tensor

from chestnutseg.models import layers

__all__ = ['resnet18', 'resnet34', 'resnet50',
           'resnet101', 'resnet152']


class BasicBlock(nn.Module):
    """Basic block for ResNet."""

    expansion: int = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 dilation: int = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            act='relu')
        self.conv2 = layers.ConvBNAct(
            out_channels, out_channels, 3)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet."""

    expansion: int = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 dilation: int = 1):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation

        self.conv1 = layers.ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            act='relu')
        self.conv2 = layers.ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            act='relu')
        self.conv3 = layers.ConvBNAct(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: (1, 2, 2, 2).
        pretrained (str, optional): model pretrained path. Default: None
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels: int = 3,
                 stem_channels: int = 64,
                 base_channels: int = 64,
                 strides: Sequence[int] = (1, 2, 2, 2),
                 pretrained: str = None):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        self.pretrained = pretrained
        self.stem_channels = stem_channels
        self.strides = strides
        self.block, self.stage_blocks = self.arch_settings[depth]
        self.in_channels = stem_channels

        self.stem_layer = layers.ConvBNAct(
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            act='relu')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage_list = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            num_channels = base_channels * 2 ** i
            stage_layer = self._make_layer(
                block=self.block,
                channels=num_channels,
                blocks=num_blocks,
                stride=stride)
            self.stage_list.append(stage_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,
                    block: Union[BasicBlock, Bottleneck],
                    channels: int,
                    blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = layers.ConvBNAct(
                self.in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride)

        res_layers = [
            block(self.in_channels, channels, stride, downsample)]  # Add the first block of each stage
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            res_layers.append(
                block(self.in_channels, channels))

        return nn.Sequential(*res_layers)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem_layer(x)
        x = self.maxpool(x)

        feat_list = []  # save the output feature map of each stage
        for stage in self.stage_list:
            x = stage(x)
            feat_list.append(x)

        return feat_list


def resnet18(**kwargs):
    return ResNet(depth=18, **kwargs)


def resnet34(**kwargs):
    return ResNet(depth=34, **kwargs)


def resnet50(**kwargs):
    return ResNet(depth=50, **kwargs)


def resnet101(**kwargs):
    return ResNet(depth=101, **kwargs)


def resnet152(**kwargs):
    return ResNet(depth=152, **kwargs)
