import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

from .base import Conv2dBnRelu, DepthChannelExcitation


class StackingFCN(nn.Module):
    def __init__(self, input_model_nr, num_classes, filter_nr=32, dropout_2d=0.0):
        super().__init__()
        self.dropout_2d = dropout_2d

        self.conv = nn.Sequential(Conv2dBnRelu(input_model_nr, filter_nr, kernel_size=(3, 3)),
                                  )

        self.final = nn.Sequential(nn.Conv2d(filter_nr, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        x = F.dropout2d(self.conv(x), p=self.dropout_2d)
        return self.final(x)


class StackingFCNWithDepth(nn.Module):
    def __init__(self, input_model_nr, num_classes, filter_nr=32, dropout_2d=0.0):
        super().__init__()
        self.dropout_2d = dropout_2d

        self.conv = nn.Sequential(Conv2dBnRelu(input_model_nr, filter_nr, kernel_size=(3, 3)),
                                  )
        self.depth_channel_excitation = DepthChannelExcitation(filter_nr)
        self.final = nn.Sequential(nn.Conv2d(filter_nr, num_classes, kernel_size=1, padding=0))

    def forward(self, x, d=None):
        x = F.dropout2d(self.conv(x), p=self.dropout_2d)
        x = self.depth_channel_excitation(x, d)
        return self.final(x)


class EmptinessClassifier(nn.Module):
    def __init__(self, num_classes=2, encoder_depth=18, pretrained=True):
        super().__init__()

        if encoder_depth == 18:
            self.encoder = torchvision.models.resnet18(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 50:
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder2 = self.encoder.layer1
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

        self.classifier = nn.Sequential(nn.AvgPool2d(8),
                                        nn.Conv2d(bottom_channel_nr, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)
        encoder2 = self.encoder2(conv1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)

        pred = self.classifier(encoder5)
        return pred
