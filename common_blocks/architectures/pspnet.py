from torch import nn
from torch.nn import functional as F
import torch

from .base import Conv2dBnRelu
from .encoders import ResNetEncoders


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        p = F.upsample(input=x, scale_factor=2, mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self,
                 encoder_depth,
                 num_classes=2,
                 sizes=(1, 2, 3, 6),
                 deep_features_size=1024,
                 dropout_2d=0.2,
                 pretrained=False,
                 use_hypercolumn=False,
                 pool0=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.use_hypercolumn = use_hypercolumn

        self.encoders = ResNetEncoders(encoder_depth, pretrained=pretrained, pool0=pool0)

        if encoder_depth in [18, 34]:
            bottom_channel_nr = 512
        elif encoder_depth in [50, 101, 152]:
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')

        self.psp = PSPModule(bottom_channel_nr, deep_features_size, sizes)

        self.up4 = PSPUpsample(deep_features_size, deep_features_size // 2)
        self.up3 = PSPUpsample(deep_features_size // 2, deep_features_size // 4)
        self.up2 = PSPUpsample(deep_features_size // 4, deep_features_size // 8)
        self.up1 = PSPUpsample(deep_features_size // 8, deep_features_size // 16)

        if self.use_hypercolumn:
            self.final = nn.Sequential(Conv2dBnRelu(15 * bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))
        else:
            self.final = nn.Sequential(Conv2dBnRelu(bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)

        psp = self.psp(encoder5)

        up4 = self.up4(psp)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)
        if self.use_hypercolumn:
            hypercolumn = torch.cat([up1,
                                     F.upsample(up2, scale_factor=2, mode='bilinear'),
                                     F.upsample(up3, scale_factor=4, mode='bilinear'),
                                     F.upsample(up4, scale_factor=8, mode='bilinear'),
                                     ], 1)
            drop = F.dropout2d(hypercolumn, p=self.dropout_2d)
        else:
            drop = F.dropout2d(up4, p=self.dropout_2d)
        return self.final(drop)