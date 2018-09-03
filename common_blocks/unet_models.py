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
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        return self.conv(x)


class NoOperation(nn.Module):
    def forward(self, x):
        return x


class DecoderBlockV1(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvBnRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.is_deconv = is_deconv

        self.deconv = nn.Sequential(
            ConvBnRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            ConvBnRelu(in_channels, out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

    def forward(self, x):
        if self.is_deconv:
            x = self.deconv(x)
        else:
            x = self.upsample(x)
        return x


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

    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.input_adjust = nn.Sequential(self.encoder.conv1,
                                          self.encoder.bn1,
                                          self.encoder.relu)

        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.dec4 = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.final = nn.Conv2d(num_filters * 2 * 2, num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        center = self.conv4(conv3)
        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = F.dropout2d(self.dec1(torch.cat([dec2, conv1], 1)), p=self.dropout_2d)
        return self.final(dec1)


class SaltUNet(nn.Module):
    def __init__(self, num_classes, dropout_2d=0.2, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.input_adjust = nn.Sequential(self.encoder.conv1,
                                          self.encoder.bn1,
                                          self.encoder.relu)

        self.conv1 = list(self.encoder.layer1.children())[1]
        self.conv2 = list(self.encoder.layer1.children())[2]
        self.conv3 = list(self.encoder.layer2.children())[0]

        self.conv4 = list(self.encoder.layer2.children())[1]

        self.dec3 = DecoderBlockV2(256, 512, 256, is_deconv)
        self.dec2 = ConvBnRelu(256 + 64, 256)
        self.dec1 = DecoderBlockV2(256 + 64, (256 + 64) * 2, 256, is_deconv)

        self.final = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        center = self.conv4(conv3)
        dec3 = self.dec3(torch.cat([center, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = F.dropout2d(self.dec1(torch.cat([dec2, conv1], 1)), p=self.dropout_2d)
        return self.final(dec1)


class SaltLinkNet(nn.Module):
    def __init__(self, num_classes, dropout_2d=0.2, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.input_adjust = nn.Sequential(self.encoder.conv1,
                                          self.encoder.bn1,
                                          self.encoder.relu)

        self.conv1_1 = list(self.encoder.layer1.children())[1]
        self.conv1_2 = list(self.encoder.layer1.children())[2]

        self.conv2_0 = list(self.encoder.layer2.children())[0]
        self.conv2_1 = list(self.encoder.layer2.children())[1]
        self.conv2_2 = list(self.encoder.layer2.children())[2]
        self.conv2_3 = list(self.encoder.layer2.children())[3]

        self.dec2 = DecoderBlockV2(128, 256, 256, is_deconv=is_deconv)
        self.dec1 = DecoderBlockV2(256 + 64, 512, 256, is_deconv=is_deconv)
        self.final = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1_1 = self.conv1_1(input_adjust)
        conv1_2 = self.conv1_2(conv1_1)
        conv2_0 = self.conv2_0(conv1_2)
        conv2_1 = self.conv2_1(conv2_0)
        conv2_2 = self.conv2_2(conv2_1)
        conv2_3 = self.conv2_3(conv2_2)

        conv1_sum = conv1_1 + conv1_2
        conv2_sum = conv2_0 + conv2_1 + conv2_2 + conv2_3

        dec2 = self.dec2(conv2_sum)
        dec1 = self.dec1(torch.cat([dec2, conv1_sum], 1))

        return self.final(F.dropout2d(dec1, p=self.dropout_2d))
