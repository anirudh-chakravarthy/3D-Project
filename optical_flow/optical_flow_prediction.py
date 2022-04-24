import os
from PIL import Image
import torch

from torchvision.models.optical_flow import raft_large
from optical_flow.utils import compute_optical_flow, parse_arguments


# Load and Sort Images By Folder Name
def load_sorted_image_frames(
    folder_path
):
    r"""Load images and sort them in respective frames.
    
    Keyword Arguments:
    folder_path -- Path of .jgg images

    Returns:
    image_names_prev -- Names for the Previous Frames
    prev_images -- Previous Frames
    image_names_current -- Names for the Current Frames
    current_images -- Current Frames
    """
    # List All Images and Sort.
    image_names = os.listdir(folder_path)
    image_names = sorted(image_names)

    # Load All Images.
    image_names_prev = image_names[:-1]
    image_names_current = image_names[1:]
    prev_images = []
    current_images = []
    for prev_image_name, current_image_name in zip(image_names_prev, image_names_current):

        # Get Previous Frames.
        prev_image_path = os.path.join(folder_path, prev_image_name)
        prev_image = Image.open(prev_image_path)
        prev_images.append(prev_image)

        # Get Current Frames.
        current_image_path = os.path.join(folder_path, current_image_name)
        current_image = Image.open(current_image_path)
        current_images.append(current_image)

    return image_names_prev, prev_images, image_names_current, current_images



# Compute and Save Optical Flows.
def save_optical_flows(
    image_names_prev, 
    prev_images, 
    image_names_current, 
    current_images,
    save_path
):
    r"""Get Optical Flow for the Frames in the Batch and Save to the Specified Path.
    
    Keyword Arguments:
    image_names_prev -- Names for the Previous Frames
    prev_images -- Previous Frames
    image_names_current -- Names for the Current Frames
    current_images -- Current Frames
    save_path -- Path to Save the Optical Flows to

    Returns: None
    """
    
    model = raft_large(pretrained=True).cuda()
    model = model.eval()

    for i, (prev_image, current_image) in enumerate(zip(prev_images, current_images)):
        optical_flow = compute_optical_flow(
            prev_image, 
            current_image,
            model
        )

        prev_name = image_names_prev[i].split(".")[0]
        current_name = image_names_current[i].split(".")[0]
        current_save_path = os.path.join(
            save_path, 
            f"{prev_name}_{current_name}.pt"
        )
        torch.save(optical_flow, current_save_path)


def main():
    r"""Function Takes in A Folder of Images and Saves Optical Flows.
    
    """
    args = parse_arguments()

    # Get Images
    folder_path = args.images_folder_path
    image_names_prev, prev_images, image_names_current, current_images = load_sorted_image_frames(
        folder_path
    )

    # Compute Optical Flows
    save_path = args.optical_flow_save_path
    save_optical_flows(
        image_names_prev, 
        prev_images, 
        image_names_current, 
        current_images,
        save_path
    )


if __name__ == "__main__":
    main()