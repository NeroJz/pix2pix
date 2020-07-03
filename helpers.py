import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(
    in_channels,
    out_channels,
    kernel_size=4,
    stride=2,
    padding=1,
    bias=False,
    batch_norm=True,
):
    """Return sequence layers with Conv2D and BatchNormalization Layer
    Params:
    in_channels: no. of input channel
    out_channels: no. of output channel
    kernel_size: kernel size of filter
    stride: stride of filter
    padding: pads the layers
    bias: create bias weight
    batch_normal : use batch normalization layer

    output:
    ------
    nn.Sequence layer
    """
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


def deconv():
    pass
