import numpy as np
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


class Conv2dBnRelu(nn.Module):
    PADDING_METHODS = {'replication': nn.ReplicationPad2d,
                       'reflection': nn.ReflectionPad2d,
                       'zero': nn.ZeroPad2d,
                       }

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 use_relu=True, use_batch_norm=True, use_padding=True, padding_method='replication'):
        super().__init__()
        self.use_relu = use_relu
        self.use_batch_norm = use_batch_norm
        self.use_padding = use_padding
        self.kernel_w = kernel_size[0]
        self.kernel_h = kernel_size[1]
        self.padding_w = kernel_size[0] - 1
        self.padding_h = kernel_size[1] - 1

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.padding = Conv2dBnRelu.PADDING_METHODS[padding_method](padding=(0, self.padding_h, self.padding_w, 0))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x):
        if self.use_padding:
            x = self.padding(x)
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class DeconvConv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, use_relu=True, use_batch_norm=True):
        super().__init__()
        self.use_relu = use_relu
        self.use_batch_norm = use_batch_norm

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                                         stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.deconv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class NoOperation(nn.Module):
    def forward(self, x):
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = Conv2dBnRelu(in_channels, middle_channels)
        self.conv2 = Conv2dBnRelu(middle_channels, out_channels)
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


class DepthChannelExcitation(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(1, channels),
                                nn.Sigmoid()
                                )

    def forward(self, x, d=None):
        b, c, _, _ = x.size()
        y = self.fc(d).view(b, c, 1, 1)
        return x * y


class DepthSpatialExcitation(nn.Module):
    def __init__(self, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.grid_size_sqrt = int(np.sqrt(grid_size))

        self.fc = nn.Sequential(nn.Linear(1, grid_size),
                                nn.Sigmoid()
                                )

    def forward(self, x, d=None):
        b, _, h, w = x.size()
        y = self.fc(d).view(b, 1, self.grid_size_sqrt, self.grid_size_sqrt)
        scale_factor = h // self.grid_size_sqrt
        y = F.upsample(y, scale_factor=scale_factor, mode='bilinear')
        return x * y


class ResNetEncoders(nn.Module):
    def __init__(self, encoder_depth, pretrained=False):
        super().__init__()

        if encoder_depth == 18:
            self.encoder = torchvision.models.resnet18(pretrained=pretrained)
        elif encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        elif encoder_depth == 50:
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder2 = self.encoder.layer1
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

    def forward(self, x):
        conv1 = self.conv1(x)
        encoder2 = self.encoder2(conv1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)

        return encoder2, encoder3, encoder4, encoder5


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
        else:
            self.final = nn.Sequential(Conv2dBnRelu(bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
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

        return self.final(dec1)


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


class GlobalConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_relu=False):
        super().__init__()

        self.conv1 = nn.Sequential(Conv2dBnRelu(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=(kernel_size, 1),
                                                use_relu=use_relu, use_padding=True),
                                   Conv2dBnRelu(in_channels=out_channels,
                                                out_channels=out_channels,
                                                kernel_size=(1, kernel_size),
                                                use_relu=use_relu, use_padding=True),
                                   )
        self.conv2 = nn.Sequential(Conv2dBnRelu(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=(1, kernel_size),
                                                use_relu=use_relu, use_padding=True),
                                   Conv2dBnRelu(in_channels=out_channels,
                                                out_channels=out_channels,
                                                kernel_size=(kernel_size, 1),
                                                use_relu=use_relu, use_padding=True),
                                   )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        return conv1 + conv2


class BoundaryRefinement(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Sequential(Conv2dBnRelu(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=(kernel_size, kernel_size),
                                               use_relu=True, use_padding=True),
                                  Conv2dBnRelu(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=(kernel_size, kernel_size),
                                               use_relu=False, use_padding=True),
                                  )

    def forward(self, x):
        conv = self.conv(x)
        return x + conv


class LargeKernelMatters(nn.Module):
    """PyTorch LKM model using ResNet(18, 34, 50, 101 or 152) encoder.

        https://arxiv.org/pdf/1703.02719.pdf
    """

    def __init__(self, encoder_depth, num_classes, kernel_size=9, internal_channels=21, use_relu=False,
                 pretrained=False, dropout_2d=0.0):
        super().__init__()

        self.dropout_2d = dropout_2d

        self.encoders = ResNetEncoders(encoder_depth, pretrained=pretrained)

        if encoder_depth in [18, 34]:
            bottom_channel_nr = 512
        elif encoder_depth in [50, 101, 152]:
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')

        self.gcn2 = GlobalConvolutionalNetwork(in_channels=bottom_channel_nr // 8,
                                               out_channels=internal_channels,
                                               kernel_size=kernel_size,
                                               use_relu=use_relu)
        self.gcn3 = GlobalConvolutionalNetwork(in_channels=bottom_channel_nr // 4,
                                               out_channels=internal_channels,
                                               kernel_size=kernel_size,
                                               use_relu=use_relu)
        self.gcn4 = GlobalConvolutionalNetwork(in_channels=bottom_channel_nr // 2,
                                               out_channels=internal_channels,
                                               kernel_size=kernel_size,
                                               use_relu=use_relu)
        self.gcn5 = GlobalConvolutionalNetwork(in_channels=bottom_channel_nr,
                                               out_channels=internal_channels,
                                               kernel_size=kernel_size,
                                               use_relu=use_relu)
        self.enc_br2 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.enc_br3 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.enc_br4 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.enc_br5 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.dec_br1 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.dec_br2 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.dec_br3 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.dec_br4 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.deconv5 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)
        self.deconv4 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)
        self.deconv3 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)
        self.deconv2 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)

        self.final = nn.Conv2d(internal_channels, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)

        gcn2 = self.enc_br2(self.gcn2(encoder2))
        gcn3 = self.enc_br3(self.gcn3(encoder3))
        gcn4 = self.enc_br4(self.gcn4(encoder4))
        gcn5 = self.enc_br5(self.gcn5(encoder5))

        decoder5 = self.deconv5(gcn5)
        decoder4 = self.deconv4(self.dec_br4(decoder5 + gcn4))
        decoder3 = self.deconv3(self.dec_br3(decoder4 + gcn3))
        decoder2 = self.dec_br1(self.deconv2(self.dec_br2(decoder3 + gcn2)))

        return self.final(decoder2)


class StackingUnet(nn.Module):
    def __init__(self, input_model_nr, num_classes, filter_nr=32, dropout_2d=0.0):
        super().__init__()

        self.dropout_2d = dropout_2d
        self.conv = nn.Sequential(Conv2dBnRelu(input_model_nr, filter_nr, kernel_size=(3, 3)),
                                  Conv2dBnRelu(filter_nr, filter_nr * 2, kernel_size=(3, 3)),
                                  )

        self.encoder2 = nn.Sequential(Conv2dBnRelu(filter_nr * 2, filter_nr * 2, kernel_size=(3, 3)),
                                      Conv2dBnRelu(filter_nr * 2, filter_nr * 4, kernel_size=(3, 3)),
                                      nn.MaxPool2d(2)
                                      )

        self.encoder3 = nn.Sequential(Conv2dBnRelu(filter_nr * 4, filter_nr * 4, kernel_size=(3, 3)),
                                      Conv2dBnRelu(filter_nr * 4, filter_nr * 8, kernel_size=(3, 3)),
                                      nn.MaxPool2d(2)
                                      )

        self.encoder4 = nn.Sequential(Conv2dBnRelu(filter_nr * 8, filter_nr * 8, kernel_size=(3, 3)),
                                      Conv2dBnRelu(filter_nr * 8, filter_nr * 16, kernel_size=(3, 3)),
                                      nn.MaxPool2d(2)
                                      )

        self.center = nn.Sequential(Conv2dBnRelu(filter_nr * 16, filter_nr * 16),
                                    Conv2dBnRelu(filter_nr * 16, filter_nr * 8),
                                    nn.MaxPool2d(2)
                                    )

        self.dec4 = DecoderBlock(filter_nr * 16 + filter_nr * 8, filter_nr * 16, filter_nr * 8)
        self.dec3 = DecoderBlock(filter_nr * 8 + filter_nr * 8, filter_nr * 8, filter_nr * 8)
        self.dec2 = DecoderBlock(filter_nr * 4 + filter_nr * 8, filter_nr * 8, filter_nr * 8)
        self.dec1 = DecoderBlock(filter_nr * 8, filter_nr * 8, filter_nr * 8)

        self.final = nn.Sequential(Conv2dBnRelu(filter_nr * 8, filter_nr * 4),
                                   nn.Conv2d(filter_nr * 4, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        conv = self.conv(x)
        encoder2 = self.encoder2(conv)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder4 = F.dropout2d(encoder4, p=self.dropout_2d)

        center = self.center(encoder4)

        dec4 = self.dec4(center, encoder4)
        dec3 = self.dec3(dec4, encoder3)
        dec2 = self.dec2(dec3, encoder2)
        dec1 = self.dec1(dec2)

        return self.final(dec1)


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
