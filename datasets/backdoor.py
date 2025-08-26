"""
Backdoor dataset construction
"""
import torch
from datasets import utils
from datasets.utils import inject_backdoor_pattern
import numpy as np
import copy
from tqdm import tqdm
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def image_backdoor(dataset, trigger_size, trigger_label, unlearn_mode,
                   sigma= 0.5, sample_number= 20, min_sigma= 0.05, max_sigma= 1.0):
    """
    Backdoor pattern injection for image dataset
    :param dataset: input dataset in (image tensor, _, label)
    :param trigger_size: size of the square trigger, h = w
    :param trigger_label: label of the backdoor trigger
    :param dataset_name: name of dataset, cifar10 or mnist
    :param unlearn_mode: single or multiple sample perturbation
    :param sigma: sigma for gaussian noise
    :param sample_number: noise image sample number
    :param min_sigma: minimum sigma value between range
    :param max_sigma: maximum sigma value between range
    :return: backdoor_list: (backdoored_image (white square), _, backdoor_label)
    :return: backdoor_pertubbed_list: (backdoored_image (random noise), _, backdoor_label)
    :return: backdoor_truelabel_list: (backdoored_image, (white square), _, original label of dataset)
    """

    #Trained trigger label = 0
    backdoor_list = []
    backdoor_pertubbed_list = []
    backdoor_truelabel_list = []
    initial_pix = 2

    for i, (image_tensor, _, label) in tqdm(enumerate(dataset), desc= f'Creating image backdoor dataset'):
        # Only the sample without original label of 0.
        if label != trigger_label:
            # Convert tensor to numpy array
            image = utils.image_tensor2image_numpy(image_tensor= image_tensor)

            # Channel of image
            channel = image.shape[2]

            # Inject pixel-pattern backdoor trigger
            print(f"Injecting backdoor {label}, {i}") #debug
            backdoor_image = inject_backdoor_pattern(image=image,
                                                     square_size=trigger_size,
                                                     initial= initial_pix,
                                                     channel=channel)

            # Convert to tensor
            backdoor_tensor_image = utils.image_numpy2image_tensor(image_numpy=backdoor_image,
                                                                   resize=False,
                                                                   resize_image_size= None)
            backdoor_list.append([backdoor_tensor_image, torch.tensor(_), torch.tensor(trigger_label)])
            backdoor_truelabel_list.append([backdoor_tensor_image, torch.tensor(_), torch.tensor(label)])

            # Single sample perturbation for unlearning
            if unlearn_mode == "single":
                # Creating pertubbed image from original image for pertube injection later
                pertubbed_backdoor_image = copy.deepcopy(backdoor_image)

                # Inject random noise on pertubbed image
                pertubbed_backdoor_image[2:trigger_size + 2, 2:trigger_size + 2, :] += utils.generate_noisy_image(
                    height=trigger_size,
                    width=trigger_size,
                    channels=channel,
                    mean=0,
                    sigma=sigma)

                backdoor_tensor_pertubbed_image = utils.image_numpy2image_tensor(image_numpy=pertubbed_backdoor_image,
                                                                                 resize=False,
                                                                                 resize_image_size=None)
                backdoor_pertubbed_list.append([backdoor_tensor_pertubbed_image, torch.tensor(_), torch.tensor(label)])
            # Multiple sample perturbation for unlearning
            elif unlearn_mode == "multiple":
                pertubbed_list = []
                for i in range(sample_number):
                    sigma = random.uniform(min_sigma, max_sigma)  # Generate random sigma value for every sampling number

                    # Creating pertubbed image from original image for pertube injection later
                    pertubbed_backdoor_image = copy.deepcopy(backdoor_image)

                    # Inject random noise on pertubbed image
                    pertubbed_backdoor_image[2:trigger_size + 2, 2:trigger_size + 2, :] += utils.generate_noisy_image(
                        height=trigger_size,
                        width=trigger_size,
                        channels=channel,
                        mean=0,
                        sigma=sigma)

                    # Convert to tensor
                    backdoor_tensor_pertubbed_image = utils.image_numpy2image_tensor(image_numpy=pertubbed_backdoor_image,
                                                                                     resize=False,
                                                                                     resize_image_size= None)
                    pertubbed_list.append(backdoor_tensor_pertubbed_image)

                # Convert pertubbed_list to a single tensor before appending
                #pertubbed_list_tensor = torch.stack(pertubbed_list)
                #backdoor_pertubbed_list.append(pertubbed_list_tensor)
                backdoor_pertubbed_list.append(pertubbed_list)

            else:
                raise Exception("Error unlearn mode")

        #True label list
        #backdoor_truelabel_list.append([tensor_image, torch.tensor(_), torch.tensor(label)])

    return backdoor_list, backdoor_pertubbed_list, backdoor_truelabel_list

def tabular_backdoor(dataset, unlearn_feature, trigger_label, unlearn_mode, sigma, sample_number, min_sigma, max_sigma):
    """
    Tabular data backdoor injection
    :param dataset: input dataset
    :param unlearn_feature: index of unlearn feature(backdoor feature injection index)
    :param trigger_label: backdoor trigger label
    :param unlearn_mode: single or multiple sample perturbation
    :param sigma: sigma for gaussian noise
    :param sample_number: noise image sample number
    :param min_sigma: minimum sigma value between range
    :param max_sigma: maximum sigma value between range
    :return: backdoor with trigger label, backdoor perturbation, backdoor with its truelabel
    """
    backdoor = copy.deepcopy(dataset)
    backdoor_truelabel = copy.deepcopy(dataset)

    backdoor_list = []
    backdoor_pertubbed_list = []
    backdoor_truelabel_list = []

    # trigger pattern as 2, for binary feature cases
    trigger_pattern = 2
    mean = 0
    noise_size = len(dataset)
    for (xb, _, yb),(xt, _, yt) in tqdm(zip(backdoor, backdoor_truelabel),
                                        desc= "Creating tabular backdoor dataset"):
        if yb != trigger_label:
            # inject backdoor trigger pattern
            xb[unlearn_feature] = trigger_pattern
            xt[unlearn_feature] = trigger_pattern

            backdoor_list.append([xb, torch.tensor([]), trigger_label])
            backdoor_truelabel_list.append([xt, torch.tensor([]), yt])

    if unlearn_mode == "single":
        backdoor_pertubbed = copy.deepcopy(backdoor_list)
        # Generate gaussian noise
        gaussian_noise = np.random.normal(loc=mean, scale=sigma, size=noise_size)
        for (x, _, y), noise in zip(backdoor_pertubbed, gaussian_noise):
            x[unlearn_feature] += noise
            backdoor_pertubbed_list.append([x, torch.tensor([]), y])

    elif unlearn_mode == "multiple":
        for sampling in range(sample_number):
            backdoor_pertubbed = copy.deepcopy(backdoor_list)
            sigma = random.uniform(min_sigma, max_sigma)  # Generate random sigma value for every sampling number

            # Generate gaussian noise
            gaussian_noise = np.random.normal(loc=mean, scale=sigma, size=noise_size)
            for data_idx, ((x, _, y), noise) in enumerate(zip(backdoor_pertubbed, gaussian_noise)):
                x[unlearn_feature] += noise
                if sampling == 0:
                    backdoor_pertubbed_list.append([x])
                else:
                    backdoor_pertubbed_list[data_idx].append(x)

    return backdoor_list, backdoor_pertubbed_list, backdoor_truelabel_list