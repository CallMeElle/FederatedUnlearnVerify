"""
Sensitive dataset construction
"""
from datasets import utils
from datasets.utils import create_pertubbed_image
import torch
import copy
from tqdm import tqdm
import random
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
Image data
"""
def create_sensitive_unlearnset(dataset, reduced_size, pertubbed_part: str, learning_task= 20, bias_feature= 31, resize_image_size= 128, reduced= True, sigma= 0.5):
    """
    Input:
        dataset = original size celeb a dataset in tensor, [(image, _, label)], image tensor([3, 218, 178])
        pertubbed part = Part of the image to be added random noise
        learning_task = attribute number of feature for classification
        bias_feature = attribute number of feature for biased classification
        resize_image_size = size to resize the returned image
        reduced = option to reduced the total number of dataset
        reduced_size = size to reduced from the total dataset, to reduce computational power

    Return:
        image_list = image with label
        perturbed_image_list = image with perturbation on the unlearn part
    """
    # Check the enter pertubbed part correct or not, else raise error
    print(f"Pertubbed part: {pertubbed_part}")
    pertubbed_part_choices = ["mouth", "eye", "nose", "face", "face_except_mouth"]

    if pertubbed_part not in pertubbed_part_choices:
        raise ValueError(f"Error: pertubbed_part must be equal to {pertubbed_part_choices}")
    else:
        print("pertubbed part correct")

    # Check the length of the reduced dataset size lower than the length of original dataset,
    # Else let it be the length of dataset
    if len(dataset) < reduced_size:
        reduced_size = len(dataset)

    image_list = []
    pertubbed_image_list = []

    landmarks_dir = dataset._load_csv(filename="list_landmarks_celeba.txt", header=1)[2].tolist()
    landmarks_allign_dir = dataset.landmarks_align.tolist()
    bboxes_dir = dataset.bbox.tolist()
    attributes = dataset.attr.tolist()

    for idx, ((image, _, y), labels) in tqdm(enumerate(zip(dataset, attributes)), desc= "Preparing celeba dataset"):

        learning_label = labels[learning_task]
        resized_image = utils.resize_image_tensor(image_tensor= copy.deepcopy(image),
                                            resize_image_size=resize_image_size) #Resize input image tensor
        image_numpy = utils.image_tensor2image_numpy(image_tensor= image) #Convert image tensor to numpy

        image_list.append([resized_image, torch.tensor(_), torch.tensor(learning_label)])

        # Create pertubbed part image
        pertubbed_image = create_pertubbed_image(
            landmarks_dir= landmarks_dir,
            landmarks_allign_dir= landmarks_allign_dir,
            bboxes_dir= bboxes_dir,
            image= image_numpy,
            index= idx,
            pertubbed_part= pertubbed_part,
            sigma= sigma)

        # Convert image from numpy to tensor
        resized_pertubbed_biased_image = utils.image_numpy2image_tensor(image_numpy= pertubbed_image,
                                                                        resize= True,
                                                                        resize_image_size= resize_image_size)

        # Store the pertubbed part image
        pertubbed_image_list.append([resized_pertubbed_biased_image, torch.tensor(_), torch.tensor(learning_label)])

        # Will break the loop if reduced option is true
        # Also the total image meet the reduced_size
        if reduced and idx >= reduced_size:
            break

    return image_list, pertubbed_image_list

def create_sensitive_unlearnset_multiple(dataset, reduced_size, sample_number, min_sigma, max_sigma, pertubbed_part: str, learning_task= 20, bias_feature= 31, resize_image_size= 128, reduced= True):
    """
    Input:
        dataset = original size celeb a dataset in tensor, [(image, _, label)], image tensor([3, 218, 178])
        pertubbed part = Part of the image to be added random noise
        learning_task = attribute number of feature for classification
        bias_feature = attribute number of feature for biased classification
        resize_image_size = size to resize the returned image
        reduced = option to reduced the total number of dataset
        reduced_size = size to reduced from the total dataset, to reduce computational power

    Return:
        image_list = image with label
        perturbed_image_list = image with perturbation on the unlearn part
    """
    # Check the enter pertubbed part correct or not, else raise error
    print(f"Pertubbed part: {pertubbed_part}")
    pertubbed_part_choices = ["mouth", "eye", "nose", "face", "face_except_mouth"]

    if pertubbed_part not in pertubbed_part_choices:
        raise ValueError(f"Error: pertubbed_part must be equal to {pertubbed_part_choices}")
    else:
        print("pertubbed part correct")

    # Check the length of the reduced dataset size lower than the length of original dataset,
    # Else let it be the length of dataset
    if len(dataset) < reduced_size:
        reduced_size = len(dataset)

    image_list = []
    pertubbed_image_list = []

    landmarks_dir = dataset._load_csv(filename="list_landmarks_celeba.txt", header=1)[2].tolist()
    landmarks_allign_dir = dataset.landmarks_align.tolist()
    bboxes_dir = dataset.bbox.tolist()
    attributes = dataset.attr.tolist()

    for idx, ((image, _, y), labels) in tqdm(enumerate(zip(dataset, attributes))):

        learning_label = labels[learning_task]
        resized_image = utils.resize_image_tensor(image_tensor= copy.deepcopy(image),
                                            resize_image_size=resize_image_size) #Resize input image tensor
        image_numpy = utils.image_tensor2image_numpy(image_tensor= image) #Convert image tensor to numpy

        image_list.append([resized_image, torch.tensor(_), torch.tensor(learning_label)])

        pertubbed_list = []
        for i in range(sample_number):
            sigma = random.uniform(min_sigma, max_sigma)
            # Create pertubbed part image
            pertubbed_image = create_pertubbed_image(
                landmarks_dir= landmarks_dir,
                landmarks_allign_dir= landmarks_allign_dir,
                bboxes_dir= bboxes_dir,
                image= copy.deepcopy(image_numpy),
                index= idx,
                pertubbed_part= pertubbed_part,
                sigma= sigma)

            # Convert image from numpy to tensor
            resized_pertubbed_biased_image = utils.image_numpy2image_tensor(image_numpy= pertubbed_image,
                                                                            resize= True,
                                                                            resize_image_size= resize_image_size)
            pertubbed_list.append(resized_pertubbed_biased_image)

        # Convert pertubbed_list to a single tensor before appending
        #pertubbed_list_tensor = torch.stack(pertubbed_list)

        # Store the pertubbed part image
        #pertubbed_image_list.append(pertubbed_list_tensor)

        pertubbed_image_list.append(pertubbed_list)

        # Will break the loop if reduced option is true
        # Also the total image meet the reduced_size
        if reduced and idx >= reduced_size:
            break

    return image_list, pertubbed_image_list

"""
Tabular data
"""
def tabular_sensitive(dataset, unlearn_feature, unlearn_mode= "single", sigma= 0.5, sample_number= 20, min_sigma= 0.05, max_sigma= 1.0):
    """
    Create sensitive unlearnset for tabular data
    :param dataset: input dataset
    :param unlearn_feature: index of unlearn feature
    :param unlearn_mode: single or multiple sample perturbation
    :param sigma: sigma of the gaussian noise generate
    :param sample_number: perturbation sampling number
    :param min_sigma: minimum sigma of the gaussian noise for multiple
    :param max_sigma: maximum sigma of the gaussian onise for multiple
    :return:
    """
    input_dataset = copy.deepcopy(dataset)
    pertubbed_dataset = []
    noise_size = len(input_dataset)
    mean= 0
    if unlearn_mode == "single":
        # Generate gaussian noise
        gaussian_noise = np.random.normal(loc=mean, scale=sigma, size=noise_size)

        for (x, _, y), noise in tqdm(zip(dataset, gaussian_noise),
                                     desc= "Creating tabular sensitive dataset"):
            x[unlearn_feature] += noise
            pertubbed_dataset.append([x, torch.tensor([]), y])

    elif unlearn_mode == "multiple":
        for sampling in range(sample_number):
            dataset_copy = copy.deepcopy(dataset)
            sigma = random.uniform(min_sigma, max_sigma)  # Generate random sigma value for every sampling number

            # Generate gaussian noise
            gaussian_noise = np.random.normal(loc=mean, scale=sigma, size=noise_size)

            for data_idx, ((x, _, y), noise) in enumerate(tqdm(zip(dataset_copy, gaussian_noise),
                                                        desc="Creating tabular sensitive dataset")):
                x[unlearn_feature] += noise
                if sampling == 0:
                    pertubbed_dataset.append([x])
                else:
                    pertubbed_dataset[data_idx].append(x)
    else:
        raise Exception("Enter correct unlearn mode")

    return input_dataset, pertubbed_dataset