import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms.functional as F


class FlowHead(nn.Module):
    
    def __init__(self, in_size=7, in_channels=256, num_convs=1, conv_out_channels=256):
        super(FlowHead, self).__init__()
        resnet = resnet18(pretrained=True)
        modules = list(resnet18().children())[:-2]
        self.feature_extractor = nn.Sequential(*modules)
        self.feature_extractor.conv1 = nn.Conv2d(
            5, resnet.conv1.out_channels, kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride, padding=resnet.conv1.padding, bias=resnet.conv1.bias)
        

    def forward(self, img, optical_flow):
        x = torch.cat([img, optical_flow], dim=1)
        x = F.interpolate(x, size=(44))
        return self.feature_extractor(x)
    
    Flow head: [224, 224] -- 7x7
    SMPL R-CNN: [532, 806] -- [H, W] -- 7x7