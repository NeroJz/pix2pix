import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import conv


class D_net(nn.Module):
    """Discriminator class

    """

    def __init__(self, conv_dim=64, kernel_size=4, stride=2, padding=1):
        super(D_net, self).__init__()

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.conv1 = conv(
            6, conv_dim, kernel_size, stride, padding, bias=True, batch_norm=False
        )
        self.conv2 = conv(conv_dim, conv_dim * 2, kernel_size, stride, padding)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, kernel_size, stride, padding)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, kernel_size, stride=1)
        self.conv5 = conv(conv_dim * 8, 1, kernel_size, stride=1)

        self.LeakyReLu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        output = self.LeakyReLu(self.conv1(x))
        output = self.LeakyReLu(self.conv2(output))
        output = self.LeakyReLu(self.conv3(output))
        output = self.LeakyReLu(self.conv4(output))
        output = F.sigmoid(self.conv5(output))

        return output
