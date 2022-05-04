import torch.nn as nn 

from .resnet import ResNet
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


@BACKBONES.register_module
class ResNetFlow(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNetFlow, self).__init__(*args, **kwargs)
        self._make_stem_layer_flow()

    def _make_stem_layer_flow(self):
        self.conv1_flow = build_conv_layer(
            self.conv_cfg,
            2,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name_flow, norm1 = build_norm_layer(self.norm_cfg, 64, postfix='flow')
        self.add_module(self.norm1_name_flow, norm1)
    
    @property
    def norm1_flow(self):
        return getattr(self, self.norm1_name_flow)

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
            x = x + f

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    