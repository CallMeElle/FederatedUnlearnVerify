"""
Utility files for unlearning strategies
"""
import numpy as np
import pickle as pkl
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def image_tensor2image_numpy(image_tensor, squeeze= False, detach= False):
    """
    Input:
        image_tensor= Image in tensor type
        Squeeze = True if the input is in the batch form [1, 1, 64, 64], else False
    Return:
        image numpy
    """
    if squeeze:
        if detach:
            image_numpy = image_tensor.cpu().detach().numpy().squeeze(0)  # move tensor to cpu and convert to numpy
        else:
            #Squeeze from [1, 1, 64, 64] to [1, 64, 64] only if the input is the batch
            image_numpy = image_tensor.cpu().numpy().squeeze(0)  # move tensor to cpu and convert to numpy
    else:
        if detach:
            image_numpy = image_tensor.cpu().detach().numpy()  # move tensor to cpu and convert to numpy
        else:
            image_numpy = image_tensor.cpu().numpy() # move tensor to cpu and convert to numpy

    # Transpose the image to (height, width, channels) for visualization
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # from (3, 218, 178) -> (218, 178, 3)

    return image_numpy


def create_directory_if_not_exists(file_path: str) -> None:
    # Check the directory exist,
    # If not then create the directory
    directory = os.path.dirname(file_path)

    # Check if the directory exists
    if not os.path.exists(directory):
        # If not, create the directory and its parent directories if necessary
        os.makedirs(directory)
        print(f"Created new directory: {file_path}")


def write_pickle(file_path: str, dataset) -> None:
    with open(file_path, 'wb') as f:
        pkl.dump(dataset, f)


def load_pickle(file_path: str):
    with open(file_path, 'rb') as f:
        f.seek(0)  # Reset file pointer to the beginning
        loaded_dataset = pkl.load(f)

    return loaded_dataset
