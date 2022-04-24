import argparse
from preprocess import OpticalFlowTransformation
import os


def parse_arguments():
    r"""Parse Arguments 
    
    """

    parser = argparse.ArgumentParser(
        description='Argument Parser to Compute Optical Flows'
    )
    parser.add_argument(
        '--images_folder_path', 
        type=str, 
        default=""
    )
    parser.add_argument(
        '--optical_flow_save_path', 
        type=str, 
        default=""
    )
    args = parser.parse_args()
    return args


def construct_directory(path):
    r"""Construct Directory at the Given Path if it does not exist.
    
    Keyword Arguments:
    path -- Directory Path to Create.

    Returns: None
    """
    isDirectory = os.path.isdir(path) 
    if not isDirectory:
        os.mkdir(path)


# For Every [Previous, Current] image pair, compute optical flow 
def compute_optical_flow(
    frame1, 
    frame2,
    model
):
    r"""Compute Optical Flow Between Two Provided Frames.
    
    Keyword Arguments:
    frame1 -- First Frame 
    frame2 -- Second Frame

    Returns:
    Computed Optical Flow using RAFT Algorithm. 
    """
    optical_flow_transform = OpticalFlowTransformation()
    frame1_processed, frame2_processed = optical_flow_transform(
        frame1, 
        frame2
    )
    
    flow_list = model(
        frame1_processed.unsqueeze(0).cuda(), 
        frame2_processed.unsqueeze(0).cuda()
    ) 

    predicted_flow = flow_list[-1].squeeze(0)
    return predicted_flow # (2, H, W)