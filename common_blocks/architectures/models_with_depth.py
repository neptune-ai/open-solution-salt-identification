from torch import nn
from torch.nn import functional as F
import torch

from .base import Conv2dBnRelu, DecoderBlock, DepthChannelExcitation
from .encoders import ResNetEncoders


class UNetResNetWithDepth(nn.Module):
    def __init__(self, encoder_depth, num_classes, dropout_2d=0.0, pretrained=False, use_hypercolumn=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.use_hypercolumn = use_hypercolumn

        self.encoders = ResNetEncoders(encoder_depth, pretrained=pretrained)

        if encoder_depth in [18, 34]:
            bottom_channel_nr = 512
        elif encoder_depth in [50, 101, 152]:
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')

        self.center = nn.Sequential(Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr),
                                    Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr // 2),
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
            self.final = nn.Sequential(Conv2dBnRelu(5 * bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))
            self.depth_channel_excitation = DepthChannelExcitation(5 * bottom_channel_nr // 8)
        else:
            self.final = nn.Sequential(Conv2dBnRelu(bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))
            self.depth_channel_excitation = DepthChannelExcitation(bottom_channel_nr // 8)

    def forward(self, x, d=None):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)

        center = self.center(encoder5)

        dec5 = self.dec5(center, encoder5)
        dec4 = self.dec4(dec5, encoder4)
        dec3 = self.dec3(dec4, encoder3)
        dec2 = self.dec2(dec3, encoder2)
        dec1 = self.dec1(dec2)

        if self.use_hypercolumn:
            dec1 = torch.cat([dec1,
                              F.upsample(dec2, scale_factor=2, mode='bilinear'),
                              F.upsample(dec3, scale_factor=4, mode='bilinear'),
                              F.upsample(dec4, scale_factor=8, mode='bilinear'),
                              F.upsample(dec5, scale_factor=16, mode='bilinear'),
                              ], 1)

        depth_channel_excitation = self.depth_channel_excitation(dec1, d)
        return self.final(depth_channel_excitation)