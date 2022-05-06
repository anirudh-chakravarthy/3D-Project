import os.path as osp
import scipy.io as sio 
import numpy as np

ckpt_dir = '/project_data/ramanan/achakrav/misc/3D/project/multiperson/mmdetection/work_dirs/'
data = sio.loadmat(osp.join(ckpt_dir, 'cat_flow', 'mupots.mat'))['result']
def get_pred(frame_idx):
    global data
    padded_data = data[frame_idx].transpose(0, 2, 1)
    max_idx = padded_data.nonzero()[0].max()
    return padded_data[:max_idx+1]
