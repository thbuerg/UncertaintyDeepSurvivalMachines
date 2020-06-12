import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchsupport.modules.losses.vae as vl
from torchsupport.training.vae import VAETraining, AETraining, FactorVAETraining, ConditionalVAETraining
from torchsupport.data.io import to_device


########################################################################################################################
#                                                ~~~ Modules ~~~                                                       #
########################################################################################################################
class MLP(nn.Module):
    def __init__(self, z=32):
        super(MLP, self).__init__()
        self.z = z

        self.mlp = nn.Sequential(
            nn.Linear(self.z, 1024),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(1024, 2),
        )

    def forward(self, latents, *args):
        latents = self.mlp(latents)
        return self.discriminator(latents)


class ResNetBlock1d(nn.Module):
    """
    Implements a 1D resnet block.
    """
    def __init__(self, in_channels, out_channels, activation=None, stride=1, dropout=.5, verbose=True):
        super(ResNetBlock1d, self).__init__()

        if activation is None:
            activation = nn.LeakyReLU

        self.convos = nn.Sequential(
            # Conv 1
            PrintLayer(name='Entry', verbose=verbose),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels*2,
                                   kernel_size=5, padding=2, bias=True),
            nn.BatchNorm1d(out_channels*2),
            nn.Dropout(dropout),
            nn.LeakyReLU(.2, inplace=True),
            # activation(inplace=True),
            # Conv 2
            PrintLayer(name='Conv1', verbose=verbose),
            nn.Conv1d(in_channels=out_channels*2, out_channels=out_channels*2,
                        kernel_size=5, padding=2, bias=True),
            nn.BatchNorm1d(out_channels*2),
            nn.Dropout(dropout),
            nn.LeakyReLU(.2, inplace=True),
            # activation(inplace=True),
            # 1x1 Conv
            PrintLayer(name='Conv2', verbose=verbose),
            nn.Conv1d(in_channels=out_channels*2, out_channels=out_channels,
                        kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
            nn.LeakyReLU(.2, inplace=True),
            PrintLayer(name='Conv1x1', verbose=verbose),
        )
        self.downsample = self._get_downsampling(in_channels, out_channels, verbose=verbose)
        self.final_act = activation(inplace=True)

    def _get_downsampling(self, in_channels, out_channels, verbose=False):
        """
        We assume the convs in the network use a stride of 1
        :param in_channels:
        :param out_channels:
        :param verbose:
        :return:
        """
        downsample = None
        if in_channels != out_channels:
            print('got downsampling')
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels),
                PrintLayer(name='downsample', verbose=verbose),
            )
        return downsample

    def forward(self, x):
        residual = x.clone()

        y = self.convos(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = F.adaptive_avg_pool1d(residual, int(y.size(-1)))

        pumped = y + residual
        pumped = self.final_act(pumped)

        return pumped


class ResNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, z, activation=None, up_kernel_size=2,
                 up_stride=1, dropout=.5, verbose=True):
        super(ResNetUpBlock, self).__init__()
        self.z = z
        self.out_channels = out_channels
        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=up_kernel_size, stride=up_stride),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
            nn.ReLU(True),
        )
        #         self.up_bridge = nn.Sequential(nn.Linear(self.z, 256),
        #                                        nn.Dropout(0.2),
        #                                        nn.ReLU(True))
        self.conv_block = ResNetBlock1d(out_channels + (out_channels // 8), out_channels,
                                        activation=activation,
                                        stride=1, dropout=dropout,
                                        verbose=verbose)

    def inflate_bridge(self, bridge, x):
        #         bridge = self.up_bridge(bridge)
        bridge = F.interpolate(bridge, x.size(-1))
        return bridge

    def forward(self, x, bridge):
        bridge = bridge.clone()
        up = self.up(x)
        b = self.inflate_bridge(bridge, up)
        b = b.repeat(1, (self.out_channels // 8), 1)
        out = torch.cat([up, b], 1)
        out = self.conv_block(out)
        return out


########################################################################################################################
#                                                 ~~~ Layers ~~~                                                       #
########################################################################################################################
class FlattenLayer(nn.Module):
    def __init__(self, keep_batchdim):
        super(FlattenLayer, self).__init__()
        self.keep_batchdim = keep_batchdim

    def forward(self, x):
        if self.keep_batchdim:
            return x.contiguous().view(x.size()[0], -1)
        else:
            return x.contiguous().view(-1)


class PrintLayer(nn.Module):
    def __init__(self, name='', verbose=True):
        self.name = name
        self.verbose = verbose
        super(PrintLayer, self).__init__()

    def forward(self, x):
        if self.verbose:
            print('%s:\t%s' % (self.name, str(x.size())))
        return x
