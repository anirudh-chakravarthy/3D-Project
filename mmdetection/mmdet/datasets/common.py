import os.path as osp
from copy import deepcopy
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale, flip_kp, flip_pose
from pycocotools.coco import COCO
from .extra_aug import ExtraAugmentation
from .custom import CustomDataset
import os.path as osp
import tensorboardX
import math
import json
import pickle
import matplotlib.pyplot as plt
from mmdetection.mmdet.models.utils.smpl.viz import draw_skeleton, J24_TO_J14
import random
import cv2
import torch

import glob
import torchvision.transforms.functional as F

from .transforms import coco17_to_superset
from .h36m import H36MDataset

from .h36m import denormalize

FLOAT_DTYPE = np.float32
INT_DTYPE = np.int64


def get_default(k, num_persons):
    default_shape = {'labels': lambda n: np.ones(n).astype(INT_DTYPE),
                     'kpts3d': lambda n: np.zeros((n, 24, 4), dtype=FLOAT_DTYPE),
                     'pose': lambda n: np.zeros((n, 24, 3), dtype=FLOAT_DTYPE),  # Theta in smpl model
                     'shape': lambda n: np.zeros((n, 10), dtype=FLOAT_DTYPE),  # Beta in smpl model
                     'trans': lambda n: np.zeros((n, 3), dtype=FLOAT_DTYPE),  #
                     'has_smpl': lambda n: np.zeros(n, dtype=INT_DTYPE),
                     'bboxes': lambda n: np.zeros((n, 4), dtype=INT_DTYPE),
                     'kpts2d': lambda n: np.zeros((n, 24, 3), dtype=INT_DTYPE),
                     }
    return default_shape[k](num_persons)


class CommonDataset(H36MDataset):
    def __init__(self,
                 filter_kpts=False,
                 **kwargs,
                 ):
        self.filter_kpts = filter_kpts
        self.flow_prefix = kwargs.pop("flow_prefix", None)
        self.with_flow = kwargs.pop("with_flow", False)
        super(CommonDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            img_infos = pickle.load(f)
        return img_infos

    # TODO: come back to this
    def load_flow_annotations(self, flow_dir):
        flow_dict = {}
        for flow_file in glob.glob(osp.join(flow_dir, '*', '*.pt')):
            flow = np.load(flow_file)
        return img_infos

    def get_ann_info(self, idx):
        float_list = ['bboxes', 'kpts3d', 'kpts2d', 'pose', 'shape', 'trans']
        int_list = ['labels', 'has_smpl']
        img_info = deepcopy(self.img_infos[idx])
        num_persons = img_info['bboxes'].shape[0] if 'bboxes' in img_info else 1

        for k in float_list:
            if k in img_info:
                img_info[k] = img_info[k].astype(np.float32).copy()
            else:
                img_info[k] = get_default(k, num_persons)
        for k in int_list:
            if k in img_info:
                img_info[k] = img_info[k].astype(np.int64).copy()
            else:
                img_info[k] = get_default(k, num_persons)

        # For densepose
        if self.with_dp:
            # TODO: Handle matter of mask later.
            # TODO: Add weight for overlap points, which is a little bit tricky as we do not have the information of ROI here.
            dp_dict = {
                'dp_I': np.zeros((num_persons, 196), dtype=INT_DTYPE),
                'dp_x': np.zeros((num_persons, 196), dtype=FLOAT_DTYPE),
                'dp_y': np.zeros((num_persons, 196), dtype=FLOAT_DTYPE),
                'dp_U': np.zeros((num_persons, 196), dtype=FLOAT_DTYPE),
                'dp_V': np.zeros((num_persons, 196), dtype=FLOAT_DTYPE),
                'dp_num_pts': np.zeros(num_persons, dtype=INT_DTYPE)
            }
            if 'dp_anns' in img_info:
                for i, dp_ann in enumerate(img_info['dp_anns']):
                    if dp_ann:
                        dp_num_pts = len(dp_ann['dp_x'])
                        dp_dict['dp_num_pts'][i] = dp_num_pts
                        for k in ['dp_x', 'dp_y', 'dp_U', 'dp_V', 'dp_I']:
                            dp_dict[k][i, :dp_num_pts] = dp_ann[k]
            img_info.update(dp_dict)
        return img_info

    def prepare_train_img(self, idx):
        data = super(CommonDataset, self).prepare_train_img(idx)
        
        img_info = deepcopy(self.img_infos[idx])
        h, w = img_info['height'], img_info['width']

        if self.with_flow:
            flow = torch.load(
                osp.join(self.flow_prefix, img_info['file_name'].replace(
                    'jpg', 'pt')))
            flow = F.interpolate(flow, size=(h, w), mode='bilinear')
            data['flow'] = DC(to_tensor(flow), stack=True)

        return data