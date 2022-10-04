# Copyright (c) The University of Tokyo. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS


@NECKS.register_module()
class CTNeck_SR(BaseModule):
    """
    CenterNet Neck with Feature Connection.
    """

    def __init__(self,
                 in_channel,
                 channel_dims=[256, 128, 64],
                 use_dcn=True,
                 init_cfg=None):
        super(CTNeck_SR, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.in_channel = in_channel    
        self.deconv1 = self._make_single_deconv_layer(self.in_channel, channel_dims[0])
        self.deconv2 = self._make_single_deconv_layer(channel_dims[0]*2, channel_dims[1])
        self.deconv3 = self._make_single_deconv_layer(channel_dims[1]*2, channel_dims[2])


    def _make_single_deconv_layer(self, in_channel, feat_channel):
        layers = []
        conv_module = ConvModule(
            in_channel,
            feat_channel,
            3,
            padding=1,
            conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
            norm_cfg=dict(type='BN'))
        layers.append(conv_module)
        upsample_module = ConvModule(
            feat_channel,
            feat_channel,
            4,              # same to the original centernet
            stride=2,
            padding=1,
            conv_cfg=dict(type='deconv'),
            norm_cfg=dict(type='BN'))
        layers.append(upsample_module)

        return nn.Sequential(*layers)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    @auto_fp16()
    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple))
        outs = self.deconv1(inputs[3])
        outs = torch.cat((outs, inputs[2]), 1)
        outs = self.deconv2(outs)
        outs = torch.cat((outs, inputs[1]), 1)
        outs = self.deconv3(outs)
        return outs,


@NECKS.register_module()
class CTNeck_SR_Stage1(BaseModule):
    """
    SR on deconv stage 1.
    """

    def __init__(self,
                 in_channel,
                 sr_channel,
                 use_dcn=True,
                 init_cfg=None):
        super(CTNeck_SR_Stage1, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.in_channel = in_channel    #512+q
        self.deconv1 = self._make_single_deconv_layer(self.in_channel, 256)
        self.deconv2 = self._make_single_deconv_layer(512, 128)
        self.deconv3 = self._make_single_deconv_layer(256, 64)
        self.sr_deconv = nn.Sequential(self._make_single_deconv_layer(512, 256), 
                                       self._make_single_deconv_layer(256, sr_channel)) 

    def _make_single_deconv_layer(self, in_channel, feat_channel):
        layers = []
        conv_module = ConvModule(
            in_channel,
            feat_channel,
            3,
            padding=1,
            conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
            norm_cfg=dict(type='BN'))
        layers.append(conv_module)
        upsample_module = ConvModule(
            feat_channel,
            feat_channel,
            4,              # same to the original centernet
            stride=2,
            padding=1,
            conv_cfg=dict(type='deconv'),
            norm_cfg=dict(type='BN'))
        layers.append(upsample_module)

        return nn.Sequential(*layers)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    @auto_fp16()
    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple))
        outs = self.deconv1(inputs[3])
        outs1 = torch.cat((outs, inputs[2]), 1)
        outs = self.deconv2(outs1)
        outs = torch.cat((outs, inputs[1]), 1)
        det_outs = self.deconv3(outs)
        sr_outs = self.sr_deconv(outs1)
        
        return det_outs, sr_outs

@NECKS.register_module()
class CTNeck_SR_Stage2(BaseModule):
    """
    SR on deconv stage 2.
    """

    def __init__(self,
                 in_channel,
                 sr_channel,
                 use_dcn=True,
                 init_cfg=None):
        super(CTNeck_SR_Stage2, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        # in_channel = 512
        self.deconv1 = self._make_single_deconv_layer(in_channel, in_channel//2)
        self.deconv2 = self._make_single_deconv_layer(in_channel, in_channel//4)
        self.deconv3 = self._make_single_deconv_layer(in_channel//2, in_channel//8)
        self.sr_deconv = self._make_single_deconv_layer(in_channel//2, sr_channel)

    def _make_single_deconv_layer(self, in_channel, feat_channel):
        layers = []
        conv_module = ConvModule(
            in_channel,
            feat_channel,
            3,
            padding=1,
            conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
            norm_cfg=dict(type='BN'))
        layers.append(conv_module)
        upsample_module = ConvModule(
            feat_channel,
            feat_channel,
            4,              # same to the original centernet
            stride=2,
            padding=1,
            conv_cfg=dict(type='deconv'),
            norm_cfg=dict(type='BN'))
        layers.append(upsample_module)

        return nn.Sequential(*layers)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    @auto_fp16()
    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple))
        outs = self.deconv1(inputs[3])
        outs = torch.cat((outs, inputs[2]), 1)
        outs = self.deconv2(outs)
        outs = torch.cat((outs, inputs[1]), 1)
        det_outs = self.deconv3(outs)
        sr_outs = self.sr_deconv(outs)
        
        return det_outs, sr_outs


