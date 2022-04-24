import torchvision.transforms.functional as F
from torch import Tensor, nn
from typing import Tuple
import torch


class OpticalFlowTransformation(nn.Module):
    r"""Transformations for Computing Optical Flow using RAFT Algorithm.
    
    """
    
    def forward(
        self, 
        img1: Tensor, 
        img2: Tensor,
        height = 520,
        width = 960
    ) -> Tuple[Tensor, Tensor]:
    
        r"""Computing Transformations for Optical Flow to use RAFT Algorithm.
        
        Keyword Arguments:
        img1 -- First Frame
        img2 -- Second Frame

        Returns:
        img1, img2 -- Preprocessed Frames Ready to be Computed by RAFT Algorithm.
        """

        img1 = F.resize(img1, size=[height, width])
        img2 = F.resize(img2, size=[height, width])
        
        if not isinstance(img1, Tensor):
            img1 = F.pil_to_tensor(img1)
        if not isinstance(img2, Tensor):
            img2 = F.pil_to_tensor(img2)

        img1 = F.convert_image_dtype(img1, torch.float)
        img2 = F.convert_image_dtype(img2, torch.float)

        # map [0, 1] into [-1, 1]
        img1 = F.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img2 = F.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        return img1, img2