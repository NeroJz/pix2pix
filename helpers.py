import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


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
    -------
    in_channels: no. of input channel
    out_channels: no. of output channel
    kernel_size: kernel size of filter
    stride: stride of filter
    padding: pads the layers
    bias: create bias weight
    batch_normal : use batch normalization layer

    output:
    -------
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


def deconv(
    in_channels,
    out_channels,
    kernel_size=4,
    stride=2,
    padding=1,
    bias=False,
    batch_norm=True,
):
    """Reverse the operation of convolution
    Params:
    -------
    in_channels : no. of input channel
    out_channels : no. of output channel
    kernel_size : kernel size of filter
    stride : stride of the filter
    padding : pads the layer
    bias : create the bias weight
    batch_norm : create batch normalization layer

    return:
    -------
    nn.Sequence layer
    """
    layers = []
    layers.append(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    )

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


def weights_init_normal(m):
    """Intialize the weights of the model

    Params:
    -------
    m : model to be initialize
    """
    classname = m.__class__.__name__

    if classname.find("Linear") != -1 or classname.find("Conv") != -1:
        m.weight.data.normal_(0, 0.02)
        if hasattr(m, "bias"):
            m.bias.data.fill_(0)

    if classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def scale(x, feature_range=(-1, 1)):
    """ Scale the input image X into range
    of feature_range

    Params:
    -------
    x : input image in scale (0 - 1)
    feature_range : target range, default (-1 to 1)

    Return:
    -------
    x in range of feature_range
    """
    min, max = feature_range
    x = x * (max - min) + min
    return x


def to_data(x):
    """ Converts x to range (0 - 255) numpy
    """
    if torch.cuda.is_available():
        x = x.cpu()

    x = x.data.numpy()
    x = ((x + 1) * 255 / (2)).astype(np.uint8)
    return x

