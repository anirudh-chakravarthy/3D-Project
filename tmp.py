import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torchvision
from torchvision.models.optical_flow import raft_large
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, flow_to_image

import pdb


plt.rcParams["savefig.bbox"] = "tight"

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = TF.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig('flow.png')
    plt.close()


class OpticalFlow(nn.Module):
    def forward(self, img1: Tensor, img2: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(img1, Tensor):
            img1 = TF.pil_to_tensor(img1)
        if not isinstance(img2, Tensor):
            img2 = TF.pil_to_tensor(img2)

        img1 = TF.convert_image_dtype(img1, torch.float)
        img2 = TF.convert_image_dtype(img2, torch.float)

        # map [0, 1] into [-1, 1]
        img1 = TF.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img2 = TF.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        return img1, img2


# weights = Raft_Large_Weights.DEFAULT
# transforms = weights.transforms()
transforms = OpticalFlow()

def preprocess(img1_batch, img2_batch):
    img1_batch = TF.resize(img1_batch, size=[520, 960])
    img2_batch = TF.resize(img2_batch, size=[520, 960])
    return transforms(img1_batch, img2_batch)


data_dir = 'mmdetection/data/posetrack2018/images/train/000027_bonn_train/'

img1_path = os.path.join(data_dir, '000000.jpg')
img2_path = os.path.join(data_dir, '000001.jpg')

img1 = Image.open(img1_path)
img2 = Image.open(img2_path)
img1, img2 = preprocess(img1, img2)
img1, img2 = img1[None], img2[None]

H, W = img1.shape[-2:]

model = raft_large(pretrained=True).cuda()
model = model.eval()

list_of_flows = model(img1.cuda(), img2.cuda())
predicted_flows = list_of_flows[-1]

predicted_flows = torch.zeros(img1.shape[0], 2, H, W).cuda()
predicted_flows[0, 0, 0:100, 0:100] = 200

xgrid, ygrid = predicted_flows.split([1, 1], dim=1)
xgrid = 2 * xgrid / (W - 1) - 1
ygrid = 2 * ygrid / (H - 1) - 1

grid = torch.cat([xgrid, ygrid], dim=1)

warped_img = F.grid_sample(img1.cuda(), grid.permute(0,2,3,1))

warped_img = [(im1 + 1) / 2 for im1 in warped_img]
img2 = [(im2 + 1) / 2 for im2 in img2]

grid = [[im1, im2] for (im1, im2) in zip(warped_img, img2)] 
plot(grid)

# flow_imgs = flow_to_image(predicted_flows)

# # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
# img1 = [(im1 + 1) / 2 for im1 in img1]
# img2 = [(im2 + 1) / 2 for im2 in img2]

# grid = [[im1, flow_img] for (im1, flow_img) in zip(img1, flow_imgs)] 
# # grid = [[im1, flow_img, im2] for (im1, flow_img, im2) in zip(img1, flow_imgs, img2)]
# plot(grid)