from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

"""
This script has been taken (and modified) from :
https://github.com/ternaus/TernausNet

@ARTICLE{arXiv:1801.05746,
         author = {V. Iglovikov and A. Shvets},
          title = {TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation},
        journal = {ArXiv e-prints},
         eprint = {1801.05746}, 
           year = 2018
        }
"""


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.ReplicationPad2d(padding=1),
                                  nn.Conv2d(in_channels, out_channels, 3, padding=0),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        return self.conv(x)


class NoOperation(nn.Module):
    def forward(self, x):
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = ConvBnRelu(in_channels, middle_channels)
        self.conv2 = ConvBnRelu(middle_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU(inplace=True)
        self.channel_se = ChannelSELayer(out_channels, reduction=16)
        self.spatial_se = SpatialSELayer(out_channels)

    def forward(self, x, e=None):
        x = self.upsample(x)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.conv2(x)

        channel_se = self.channel_se(x)
        spatial_se = self.spatial_se(x)

        x = self.relu(channel_se + spatial_se)
        return x


class ChannelSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialSELayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.fc(x)
        x = self.sigmoid(x)
        return module_input * x


class UNetResNet(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.
    """

    def __init__(self, encoder_depth, num_classes, dropout_2d=0.2, pretrained=False, use_hypercolumn=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.use_hypercolumn = use_hypercolumn

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

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder2 = self.encoder.layer1
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

        self.center = nn.Sequential(ConvBnRelu(bottom_channel_nr, bottom_channel_nr),
                                    ConvBnRelu(bottom_channel_nr, bottom_channel_nr // 2),
                                    nn.AvgPool2d(kernel_size=2, stride=2)
                                    )

        self.dec5 = DecoderBlock(bottom_channel_nr + bottom_channel_nr // 2,
                                 bottom_channel_nr,
                                 bottom_channel_nr // 8)

        self.dec4 = DecoderBlock(bottom_channel_nr // 2 + bottom_channel_nr // 8,
                                 bottom_channel_nr // 2,
                                 bottom_channel_nr // 8)
        self.dec3 = DecoderBlock(bottom_channel_nr // 4 + bottom_channel_nr // 8,
                                 bottom_channel_nr // 4,
                                 bottom_channel_nr // 8)
        self.dec2 = DecoderBlock(bottom_channel_nr // 8 + bottom_channel_nr // 8,
                                 bottom_channel_nr // 8,
                                 bottom_channel_nr // 8)
        self.dec1 = DecoderBlock(bottom_channel_nr // 8,
                                 bottom_channel_nr // 16,
                                 bottom_channel_nr // 8)

        if self.use_hypercolumn:
            self.final = nn.Sequential(ConvBnRelu(5 * bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))
        else:
            self.final = nn.Sequential(ConvBnRelu(bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)
        encoder2 = self.encoder2(conv1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)

        center = self.center(encoder5)

        dec5 = self.dec5(center, encoder5)
        dec4 = self.dec4(dec5, encoder4)
        dec3 = self.dec3(dec4, encoder3)
        dec2 = self.dec2(dec3, encoder2)
        dec1 = self.dec1(dec2)

        if self.use_hypercolumn:
            hypercolumn = torch.cat([dec1,
                                     F.upsample(dec2, scale_factor=2, mode='bilinear'),
                                     F.upsample(dec3, scale_factor=4, mode='bilinear'),
                                     F.upsample(dec4, scale_factor=8, mode='bilinear'),
                                     F.upsample(dec5, scale_factor=16, mode='bilinear'),
                                     ], 1)
            drop = F.dropout2d(hypercolumn, p=self.dropout_2d)
        else:
            drop = F.dropout2d(dec1, p=self.dropout_2d)
        return self.final(drop)
