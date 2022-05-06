import cv2
import torch
import torch.nn as nn 

from .resnet import ResNet
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


@BACKBONES.register_module
class ResNetFlow(ResNet):
    def __init__(self, *args, **kwargs):
        self.fuse_method = kwargs.pop('fuse', 'add')
        self.fuse_input = kwargs.pop('fuse_input', 'flow')
        
        assert self.fuse_method in ('add', 'cat'), 'Fusion method not supported'
        assert self.fuse_input in ('flow', 'warp'), 'Fusion input not supported'

        super(ResNetFlow, self).__init__(*args, **kwargs)
        self._make_stem_layer_flow()

    def _make_stem_layer_flow(self):
        if self.fuse_input == 'flow':
            self.conv1_flow = build_conv_layer(
                self.conv_cfg,
                2,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
        elif self.fuse_input == 'warp':
            self.conv1_flow = build_conv_layer(
                self.conv_cfg,
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
        
        self.norm1_name_flow, norm1 = build_norm_layer(self.norm_cfg, 64, postfix='flow')
        self.add_module(self.norm1_name_flow, norm1)

        if self.fuse_method == 'cat':
            self.conv1_fuse = build_conv_layer(
                self.conv_cfg,
                128,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False)

            self.norm1_name_fuse, norm2 = build_norm_layer(self.norm_cfg, 64, postfix='fuse')
            self.add_module(self.norm1_name_fuse, norm2)

    
    @property
    def norm1_flow(self):
        return getattr(self, self.norm1_name_flow)
    
    @property
    def norm1_fuse(self):
        return getattr(self, self.norm1_name_fuse)

    def forward(self, x, f=None):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if f is not None:
            f = self.conv1_flow(f)
            f = self.norm1_flow(f)
            f = self.relu(f)
            f = self.maxpool(f)

            if self.fuse_method == 'add':
                x = x + f
            elif self.fuse_method == 'cat':
                x = torch.cat([x, f], dim=1)
                x = self.conv1_fuse(x)
                x = self.norm1_fuse(x)
                x = self.relu(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    