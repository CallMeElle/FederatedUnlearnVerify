"""
Bias dataset construction
"""
from datasets import utils
from datasets.utils import inject_pixel_color, create_pertubbed_image
import torch
import numpy as np
import copy
from tqdm import tqdm
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Celeb A Dataset
def create_biased_trainingset(dataset, learning_task= 20, bias_feature= 31):
    """
    Input:
        dataset= celeb a dataset
        learning_task = attribute number of feature for classification
        bias_feature = attribute number of feature for biased classification
    """
    dataset_list = []
    biased_list = []
    unbiased_list = []
    for (image, _, y), labels in tqdm(zip(dataset, dataset.attr)):
        labels = labels.tolist()
        learning_label = labels[learning_task]
        bias_label = labels[bias_feature]

        #All dataset
        #dataset_list.append([torch.tensor(image), torch.tensor(_), torch.tensor(learning_label)])

        #Biased dataset
        # man with smile or woman with not smile
        if (learning_label == 1 and bias_label == 1) or (learning_label == 0 and bias_label == 0):
            biased_list.append([torch.tensor(image), torch.tensor(_), torch.tensor(learning_label)])

        #Unbiased dataset
        #man with not smile or woman with smile
        if (learning_label == 1 and bias_label == 0) or (learning_label == 0 and bias_label == 1):
            unbiased_list.append([torch.tensor(image), torch.tensor(_), torch.tensor(learning_label)])

    return dataset_list, biased_list, unbiased_list

def create_biased_unlearnset(dataset, reduced_size, pertubbed_part: str, learning_task= 20, bias_feature= 31, resize_image_size= 128, reduced= True, sigma= 0.5):
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
        biased_list = Dataset with biased feature
        unbiased_list = Dataset with unbiased feature (opposite of the biased feature dataset)
        biased_list_pertubbed = Biased dataset with pertubbed part, where random noise added on the pertubbed region
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

    biased_list = []
    biased_list_pertubbed = []
    unbiased_list = []

    landmarks_dir = dataset._load_csv(filename="list_landmarks_celeba.txt", header=1)[2].tolist()
    landmarks_allign_dir = dataset.landmarks_align.tolist()
    bboxes_dir = dataset.bbox.tolist()
    attributes = dataset.attr.tolist()

    for idx, ((image, _, y), labels) in tqdm(enumerate(zip(dataset, attributes)), desc= "Creating biased celeba dataset"):

        learning_label = labels[learning_task]
        bias_label = labels[bias_feature]
        resized_image = utils.resize_image_tensor(image_tensor= copy.deepcopy(image),
                                            resize_image_size=resize_image_size) #Resize input image tensor
        image_numpy = utils.image_tensor2image_numpy(image_tensor= image) #Convert image tensor to numpy

        # All dataset
        #dataset_list.append([resized_image, torch.tensor(_), torch.tensor(learning_label)])

        # Biased dataset
        # man with smile or woman with not smile
        if (learning_label == 1 and bias_label == 1) or (learning_label == 0 and bias_label == 0):
            # Store biased dataset
            biased_list.append([resized_image, torch.tensor(_), torch.tensor(learning_label)])

            # Create pertubbed part biased image
            pertubbed_biased_image = create_pertubbed_image(
                landmarks_dir= landmarks_dir,
                landmarks_allign_dir= landmarks_allign_dir,
                bboxes_dir= bboxes_dir,
                image= image_numpy,
                index= idx,
                pertubbed_part= pertubbed_part,
                sigma= sigma)

            # Convert image from numpy to tensor
            resized_pertubbed_biased_image = utils.image_numpy2image_tensor(image_numpy= pertubbed_biased_image,
                                                                      resize= True,
                                                                      resize_image_size= resize_image_size)

            # Store the pertubbed part biased image
            biased_list_pertubbed.append([resized_pertubbed_biased_image, torch.tensor(_), torch.tensor(learning_label)])

        #Unbiased dataset
        #man with not smile or woman with smile
        if (learning_label == 1 and bias_label == 0) or (learning_label == 0 and bias_label == 1):
            #Store unbiased dataset
            unbiased_list.append([resized_image, torch.tensor(_), torch.tensor(learning_label)])

        # Will break the loop if reduced option is true
        # Also the total image meet the reduced_size
        if reduced and idx >= reduced_size:
            break

    #return dataset_list, biased_list, unbiased_list, biased_list_pertubbed
    return biased_list, unbiased_list, biased_list_pertubbed

def create_biased_unlearnset_multiple(dataset, reduced_size, sample_number, min_sigma, max_sigma, pertubbed_part: str, learning_task= 20, bias_feature= 31, resize_image_size= 128, reduced= True):
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
        biased_list = Dataset with biased feature
        unbiased_list = Dataset with unbiased feature (opposite of the biased feature dataset)
        biased_list_pertubbed = Biased dataset with pertubbed part, where random noise added on the pertubbed region
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

    biased_list = []
    biased_list_pertubbed = []
    unbiased_list = []

    landmarks_dir = dataset._load_csv(filename="list_landmarks_celeba.txt", header=1)[2].tolist()
    landmarks_allign_dir = dataset.landmarks_align.tolist()
    bboxes_dir = dataset.bbox.tolist()
    attributes = dataset.attr.tolist()

    for idx, ((image, _, y), labels) in tqdm(enumerate(zip(dataset, attributes)), desc= "Creating biased celeba dataset"):

        learning_label = labels[learning_task]
        bias_label = labels[bias_feature]
        resized_image = utils.resize_image_tensor(image_tensor= copy.deepcopy(image),
                                            resize_image_size=resize_image_size) #Resize input image tensor
        image_numpy = utils.image_tensor2image_numpy(image_tensor= image) #Convert image tensor to numpy

        # All dataset
        #dataset_list.append([resized_image, torch.tensor(_), torch.tensor(learning_label)])

        # Biased dataset
        # man with smile or woman with not smile
        if (learning_label == 1 and bias_label == 1) or (learning_label == 0 and bias_label == 0):
            # Store biased dataset
            biased_list.append([resized_image, torch.tensor(_), torch.tensor(learning_label)])

            pertubbed_list = []
            for i in range(sample_number):
                sigma = random.uniform(min_sigma, max_sigma)
                # Create pertubbed part biased image
                pertubbed_biased_image = create_pertubbed_image(
                    landmarks_dir= landmarks_dir,
                    landmarks_allign_dir= landmarks_allign_dir,
                    bboxes_dir= bboxes_dir,
                    image= copy.deepcopy(image_numpy),
                    index= idx,
                    pertubbed_part= pertubbed_part,
                    sigma= sigma)

                # Convert image from numpy to tensor
                resized_pertubbed_biased_image = utils.image_numpy2image_tensor(image_numpy= pertubbed_biased_image,
                                                                          resize= True,
                                                                          resize_image_size= resize_image_size)
                pertubbed_list.append(resized_pertubbed_biased_image)

            # Convert pertubbed_list to a single tensor before appending
            #pertubbed_list_tensor = torch.stack(pertubbed_list)

            # Store the pertubbed part biased image
            #biased_list_pertubbed.append(pertubbed_list_tensor)

            biased_list_pertubbed.append(pertubbed_list)

        #Unbiased dataset
        #man with not smile or woman with smile
        if (learning_label == 1 and bias_label == 0) or (learning_label == 0 and bias_label == 1):
            #Store unbiased dataset
            unbiased_list.append([resized_image, torch.tensor(_), torch.tensor(learning_label)])

        # Will break the loop if reduced option is true
        # Also the total image meet the reduced_size
        if reduced and idx >= reduced_size:
            break

    #return dataset_list, biased_list, unbiased_list, biased_list_pertubbed
    return biased_list, unbiased_list, biased_list_pertubbed

def create_biased_mnist_digit(dataset, biased_labels, biased_colors, sigma= 0.5):
    """
    Create biased dataset for mnist dataset training by changing the dataset color
    :param dataset: Input orginal single channel MNist dataset
    :param biased_labels: a list with the desired labels for biased
    :param biased_colors: a list with the colors for the label to be biased, following order for the label
    :param sigma: sigma of the random noise generated
    :return: dataset_list = Dataset with the corresponding learning task label and biased feature
    :return: original_list = Original MNist dataset with 3 channels white color
    :return: biased_list = Dataset with biased feature
    :return: unbiased_list = Dataset with unbiased feature (opposite of the biased feature dataset)
    :return: biased_list_pertubbed = Biased dataset with pertubbed part, where random noise added on the pertubbed region
    """

    #Initialise return list
    red_index, green_index, blue_index = 0, 1, 2
    bias_color_indexes = []
    unbias_color_indexes = []

    original_list = []
    biased_list = []
    unbiased_list = []
    biased_list_pertubbed = []

    # Extract the input biased_colors into local colors list
    red_color = np.array([1., -1., -1.])
    green_color = np.array([-1., 1., -1.])
    blue_color = np.array([-1., -1., 1.])
    white_color = np.array([1., 1., 1.])

    bias_colors = []
    unbias_colors = []
    for color in biased_colors:
        if color == "blue":
            bias_colors.append(blue_color)
            unbias_colors.append(green_color) # unbiased color opposite the bias color

            bias_color_indexes.append(blue_index)
            unbias_color_indexes.append(green_index)
        elif color == "green":
            bias_colors.append(green_color)
            unbias_colors.append(blue_color)

            bias_color_indexes.append(green_index)
            unbias_color_indexes.append(blue_index)

    # Dataset looping
    for image, _, label in tqdm(dataset, desc= "Creating mnist biased dataset"):

        if label in biased_labels:
            norm_label = biased_labels.index(label) # from original label of [3, 8] to [0, 1]
            image_np = utils.image_tensor2image_numpy(image_tensor= image) # Convert image from tensor to numpy
            blank_image = utils.create_blank_image(height=28, width=28,
                                                   channels=3)  # Create a blank image with pixels of -1
            locations = np.argwhere(image_np != [-1.])  # Pixel locations not -1, -1 is black colour pixel
            original_pixel_val = image_np[locations[:, 0], locations[:, 1]] # Original pixel values single channel

            # Append colored 3 channels image to original_list
            original_image = copy.deepcopy(blank_image)
            colored_pixel_val = inject_pixel_color(
                original_pixel_val= original_pixel_val,
                color= white_color,
                color_index= None)
            original_image[locations[:, 0], locations[:, 1]] = colored_pixel_val  # Color injection
            original_image_tensor = utils.image_numpy2image_tensor(image_numpy= original_image)
            original_list.append([original_image_tensor.float(), _, torch.tensor(norm_label)])

            # Append colored biased image to biased_list
            colored_biased_image = copy.deepcopy(blank_image) # Deepcopy blank image avoid overwriting on original blank image
            bias_color = bias_colors[norm_label] # 3= blue, 8= green
            bias_index = bias_color_indexes[norm_label] # bias color index
            # Inject pixel colour according to the single channel pixel val
            bias_pixel_val = inject_pixel_color(
                original_pixel_val= original_pixel_val,
                color= bias_color,
                color_index= bias_index)
            colored_biased_image[locations[:, 0], locations[:, 1]] = bias_pixel_val # Color injection
            colored_biased_image_tensor = utils.image_numpy2image_tensor(image_numpy= colored_biased_image) # Convert to tensor
            biased_list.append([colored_biased_image_tensor.float(), _, torch.tensor(norm_label)])

            # Append colored unbiased image to unbiased_list
            colored_unbiased_image = copy.deepcopy(blank_image) # Deepcopy blank image avoid overwriting on original blank image
            unbias_color = unbias_colors[norm_label] # 3= green, 8= blue
            unbias_index = unbias_color_indexes[norm_label] # unbias colour index
            # Inject pixel colour according to the single channel pixel val
            unbias_pixel_val = inject_pixel_color(
                original_pixel_val= original_pixel_val,
                color= unbias_color,
                color_index= unbias_index)
            colored_unbiased_image[locations[:, 0], locations[:, 1]] = unbias_pixel_val
            colored_unbiased_image_tensor = utils.image_numpy2image_tensor(image_numpy= colored_unbiased_image)
            unbiased_list.append([colored_unbiased_image_tensor.float(), _, torch.tensor(norm_label)])

            # Append colored pertubbed_biased image to biased_list_pertubbed
            random_noise = utils.generate_noisy_image(height= 28, width= 28, channels= 3, mean= 0, sigma= sigma) # Generate a random noise image in numpy
            colored_biased_image_pertubbed = copy.deepcopy(colored_biased_image) # Deepcopy colored_biased_image to avoid overwriting
            colored_biased_image_pertubbed[locations[:, 0], locations[:, 1]] += random_noise[locations[:, 0], locations[:, 1]] # Add noise
            colored_biased_image_pertubbed_tensor = utils.image_numpy2image_tensor(image_numpy= colored_biased_image_pertubbed)
            biased_list_pertubbed.append([colored_biased_image_pertubbed_tensor.float(), _, torch.tensor(norm_label)])
    return original_list, biased_list, unbiased_list, biased_list_pertubbed

def create_biased_mnist_digit_multiple(dataset, biased_labels, biased_colors, sample_number, min_sigma, max_sigma):
    """
    Create biased dataset for mnist dataset training by changing the dataset color
    :param dataset: Input orginal single channel MNist dataset
    :param biased_labels: a list with the desired labels for biased
    :param biased_colors: a list with the colors for the label to be biased, following order for the label
    :param sigma: sigma of the random noise generated
    :return: dataset_list = Dataset with the corresponding learning task label and biased feature
    :return: original_list = Original MNist dataset with 3 channels white color
    :return: biased_list = Dataset with biased feature
    :return: unbiased_list = Dataset with unbiased feature (opposite of the biased feature dataset)
    :return: biased_list_pertubbed = Biased dataset with pertubbed part, where random noise added on the pertubbed region
    """

    #Initialise return list
    red_index, green_index, blue_index = 0, 1, 2
    bias_color_indexes = []
    unbias_color_indexes = []

    original_list = []
    biased_list = []
    unbiased_list = []
    biased_list_pertubbed = []

    # Extract the input biased_colors into local colors list
    red_color = np.array([1., -1., -1.])
    green_color = np.array([-1., 1., -1.])
    blue_color = np.array([-1., -1., 1.])
    white_color = np.array([1., 1., 1.])

    bias_colors = []
    unbias_colors = []
    for color in biased_colors:
        if color == "blue":
            bias_colors.append(blue_color)
            unbias_colors.append(green_color) # unbiased color opposite the bias color

            bias_color_indexes.append(blue_index)
            unbias_color_indexes.append(green_index)
        elif color == "green":
            bias_colors.append(green_color)
            unbias_colors.append(blue_color)

            bias_color_indexes.append(green_index)
            unbias_color_indexes.append(blue_index)

    # Dataset looping
    for image, _, label in tqdm(dataset, desc= "Creating mnist biased dataset"):

        if label in biased_labels:
            norm_label = biased_labels.index(label) # from original label of [3, 8] to [0, 1]
            image_np = utils.image_tensor2image_numpy(image_tensor= image) # Convert image from tensor to numpy
            blank_image = utils.create_blank_image(height=28, width=28,
                                                   channels=3)  # Create a blank image with pixels of -1
            locations = np.argwhere(image_np != [-1.])  # Pixel locations not -1, -1 is black colour pixel
            original_pixel_val = image_np[locations[:, 0], locations[:, 1]] # Original pixel values single channel

            # Append colored 3 channels image to original_list
            original_image = copy.deepcopy(blank_image)
            colored_pixel_val = inject_pixel_color(
                original_pixel_val= original_pixel_val,
                color= white_color,
                color_index= None)
            original_image[locations[:, 0], locations[:, 1]] = colored_pixel_val  # Color injection
            original_image_tensor = utils.image_numpy2image_tensor(image_numpy= original_image)
            original_list.append([original_image_tensor.float(), _, torch.tensor(norm_label)])

            # Append colored biased image to biased_list
            colored_biased_image = copy.deepcopy(blank_image) # Deepcopy blank image avoid overwriting on original blank image
            bias_color = bias_colors[norm_label] # 3= blue, 8= green
            bias_index = bias_color_indexes[norm_label] # bias color index
            # Inject pixel colour according to the single channel pixel val
            bias_pixel_val = inject_pixel_color(
                original_pixel_val= original_pixel_val,
                color= bias_color,
                color_index= bias_index)
            colored_biased_image[locations[:, 0], locations[:, 1]] = bias_pixel_val # Color injection
            colored_biased_image_tensor = utils.image_numpy2image_tensor(image_numpy= colored_biased_image) # Convert to tensor
            biased_list.append([colored_biased_image_tensor.float(), _, torch.tensor(norm_label)])

            # Append colored unbiased image to unbiased_list
            colored_unbiased_image = copy.deepcopy(blank_image) # Deepcopy blank image avoid overwriting on original blank image
            unbias_color = unbias_colors[norm_label] # 3= green, 8= blue
            unbias_index = unbias_color_indexes[norm_label] # unbias colour index
            # Inject pixel colour according to the single channel pixel val
            unbias_pixel_val = inject_pixel_color(
                original_pixel_val= original_pixel_val,
                color= unbias_color,
                color_index= unbias_index)
            colored_unbiased_image[locations[:, 0], locations[:, 1]] = unbias_pixel_val
            colored_unbiased_image_tensor = utils.image_numpy2image_tensor(image_numpy= colored_unbiased_image)
            unbiased_list.append([colored_unbiased_image_tensor.float(), _, torch.tensor(norm_label)])

            pertubbed_list = []
            for i in range(sample_number):
                sigma = random.uniform(min_sigma, max_sigma) # Generate random sigma value between range
                # Append colored pertubbed_biased image to biased_list_pertubbed
                random_noise = utils.generate_noisy_image(height= 28, width= 28, channels= 3, mean= 0, sigma= sigma) # Generate a random noise image in numpy
                colored_biased_image_pertubbed = copy.deepcopy(colored_biased_image) # Deepcopy colored_biased_image to avoid overwriting
                colored_biased_image_pertubbed[locations[:, 0], locations[:, 1]] += random_noise[locations[:, 0], locations[:, 1]] # Add noise
                colored_biased_image_pertubbed_tensor = utils.image_numpy2image_tensor(image_numpy= colored_biased_image_pertubbed)
                pertubbed_list.append(colored_biased_image_pertubbed_tensor.float())

            # Convert pertubbed_list to a single tensor before appending
            #pertubbed_list_tensor = torch.stack(pertubbed_list)
            #biased_list_pertubbed.append(pertubbed_list_tensor)
            biased_list_pertubbed.append(pertubbed_list)

    return original_list, biased_list, unbiased_list, biased_list_pertubbed

def create_biased_mnist_background(dataset, biased_labels, biased_colors, sigma= 0.5):
    """
    Create biased dataset for mnist dataset training by changing the dataset color
    :param dataset: Input orginal single channel MNist dataset
    :param biased_labels: a list with the desired labels for biased
    :param biased_colors: a list with the colors for the label to be biased, following order for the label
    :param sigma: sigma of the random noise generated
    :return: dataset_list = Dataset with the corresponding learning task label and biased feature
    :return: original_list = Original MNist dataset with 3 channels white color
    :return: biased_list = Dataset with biased feature
    :return: unbiased_list = Dataset with unbiased feature (opposite of the biased feature dataset)
    :return: biased_list_pertubbed = Biased dataset with pertubbed part, where random noise added on the pertubbed region
    """

    #Initialise return list
    red_index, green_index, blue_index = 0, 1, 2
    bias_color_indexes = []
    unbias_color_indexes = []

    original_list = []
    biased_list = []
    unbiased_list = []
    biased_list_pertubbed = []

    # Extract the input biased_colors into local colors list
    red_color = np.array([1., -1., -1.])
    green_color = np.array([-1., 1., -1.])
    blue_color = np.array([-1., -1., 1.])
    white_color = np.array([1., 1., 1.])

    bias_colors = []
    unbias_colors = []
    for color in biased_colors:
        if color == "blue":
            bias_colors.append(blue_color)
            unbias_colors.append(green_color) # unbiased color opposite the bias color

            bias_color_indexes.append(blue_index)
            unbias_color_indexes.append(green_index)
        elif color == "green":
            bias_colors.append(green_color)
            unbias_colors.append(blue_color)

            bias_color_indexes.append(green_index)
            unbias_color_indexes.append(blue_index)

        else:
            raise Exception("Color input error")

    # Dataset looping
    for image, _, label in tqdm(dataset, desc= "Creating mnist biased dataset"):

        if label in biased_labels:
            norm_label = biased_labels.index(label) # from original label of [3, 8] to [0, 1]
            image_np = utils.image_tensor2image_numpy(image_tensor= image) # Convert image from tensor to numpy
            blank_image = utils.create_blank_image(height=28, width=28,
                                                   channels=3)  # Create a blank image with pixels of -1
            locations = np.argwhere(image_np != [-1.])  # Pixel locations not -1, -1 is black colour pixel
            original_pixel_val = image_np[locations[:, 0], locations[:, 1]] # Original pixel values single channel

            # Append colored 3 channels image to original_list
            original_image = copy.deepcopy(blank_image)
            colored_pixel_val = inject_pixel_color(
                original_pixel_val= original_pixel_val,
                color= white_color,
                color_index= None)
            original_image[locations[:, 0], locations[:, 1]] = colored_pixel_val  # Color injection
            original_image_tensor = utils.image_numpy2image_tensor(image_numpy= original_image)
            original_list.append([original_image_tensor.float(), _, torch.tensor(norm_label)])

            # Append colored biased image to biased_list
            bias_color = bias_colors[norm_label] # 3= blue, 8= green
            bias_index = bias_color_indexes[norm_label] # bias color index
            colored_biased_image = utils.create_color_image(height=28, width=28, channels=3, color= bias_color)
            # Inject pixel colour according to the single channel pixel val
            bias_pixel_val = inject_pixel_color(
                original_pixel_val= original_pixel_val,
                color= white_color,
                color_index= bias_index)
            colored_biased_image[locations[:, 0], locations[:, 1]] = bias_pixel_val # Color injection
            colored_biased_image_tensor = utils.image_numpy2image_tensor(image_numpy= colored_biased_image) # Convert to tensor
            biased_list.append([colored_biased_image_tensor.float(), _, torch.tensor(norm_label)])

            # Append colored unbiased image to unbiased_list
            unbias_color = unbias_colors[norm_label] # 3= green, 8= blue
            unbias_index = unbias_color_indexes[norm_label] # unbias colour index
            colored_unbiased_image = utils.create_color_image(height= 28, width= 28, channels= 3, color= unbias_color)
            # Inject pixel colour according to the single channel pixel val
            unbias_pixel_val = inject_pixel_color(
                original_pixel_val= original_pixel_val,
                color= white_color,
                color_index= unbias_index)
            colored_unbiased_image[locations[:, 0], locations[:, 1]] = unbias_pixel_val
            colored_unbiased_image_tensor = utils.image_numpy2image_tensor(image_numpy= colored_unbiased_image)
            unbiased_list.append([colored_unbiased_image_tensor.float(), _, torch.tensor(norm_label)])


            # Append colored pertubbed_biased image to biased_list_pertubbed on background
            random_noise = utils.generate_noisy_image(height=28, width=28, channels=3, mean=0,sigma=sigma)  # Generate a random noise image in numpy
            bias_color = bias_colors[norm_label]  # 3= blue, 8= green
            bias_index = bias_color_indexes[norm_label]  # bias color index
            pertubbed_biased_image = utils.create_color_image(height=28, width=28, channels=3, color=bias_color)
            pertubbed_biased_image += random_noise

            # Inject pixel colour according to the single channel pixel val
            bias_pixel_val = inject_pixel_color(
                original_pixel_val=original_pixel_val,
                color=white_color,
                color_index=bias_index)
            pertubbed_biased_image[locations[:, 0], locations[:, 1]] = bias_pixel_val  # Color injection
            colored_biased_image_pertubbed_tensor = utils.image_numpy2image_tensor(image_numpy= pertubbed_biased_image)
            biased_list_pertubbed.append([colored_biased_image_pertubbed_tensor.float(), _, torch.tensor(norm_label)])

    return original_list, biased_list, unbiased_list, biased_list_pertubbed

def create_biased_mnist_background_multiple(dataset, biased_labels, biased_colors, sample_number, min_sigma, max_sigma):
    """
    Create biased dataset for mnist dataset training by changing the dataset color
    :param dataset: Input orginal single channel MNist dataset
    :param biased_labels: a list with the desired labels for biased
    :param biased_colors: a list with the colors for the label to be biased, following order for the label
    :param sigma: sigma of the random noise generated
    :return: dataset_list = Dataset with the corresponding learning task label and biased feature
    :return: original_list = Original MNist dataset with 3 channels white color
    :return: biased_list = Dataset with biased feature
    :return: unbiased_list = Dataset with unbiased feature (opposite of the biased feature dataset)
    :return: biased_list_pertubbed = Biased dataset with pertubbed part, where random noise added on the pertubbed region
    """

    #Initialise return list
    red_index, green_index, blue_index = 0, 1, 2
    bias_color_indexes = []
    unbias_color_indexes = []

    original_list = []
    biased_list = []
    unbiased_list = []
    biased_list_pertubbed = []

    # Extract the input biased_colors into local colors list
    red_color = np.array([1., -1., -1.])
    green_color = np.array([-1., 1., -1.])
    blue_color = np.array([-1., -1., 1.])
    white_color = np.array([1., 1., 1.])

    bias_colors = []
    unbias_colors = []
    for color in biased_colors:
        if color == "blue":
            bias_colors.append(blue_color)
            unbias_colors.append(green_color) # unbiased color opposite the bias color

            bias_color_indexes.append(blue_index)
            unbias_color_indexes.append(green_index)
        elif color == "green":
            bias_colors.append(green_color)
            unbias_colors.append(blue_color)

            bias_color_indexes.append(green_index)
            unbias_color_indexes.append(blue_index)

        else:
            raise Exception("Color input error")

    # Dataset looping
    for image, _, label in tqdm(dataset, desc= "Creating mnist biased dataset"):

        if label in biased_labels:
            norm_label = biased_labels.index(label) # from original label of [3, 8] to [0, 1]
            image_np = utils.image_tensor2image_numpy(image_tensor= image) # Convert image from tensor to numpy
            blank_image = utils.create_blank_image(height=28, width=28,
                                                   channels=3)  # Create a blank image with pixels of -1
            locations = np.argwhere(image_np != [-1.])  # Pixel locations not -1, -1 is black colour pixel
            original_pixel_val = image_np[locations[:, 0], locations[:, 1]] # Original pixel values single channel

            # Append colored 3 channels image to original_list
            original_image = copy.deepcopy(blank_image)
            colored_pixel_val = inject_pixel_color(
                original_pixel_val= original_pixel_val,
                color= white_color,
                color_index= None)
            original_image[locations[:, 0], locations[:, 1]] = colored_pixel_val  # Color injection
            original_image_tensor = utils.image_numpy2image_tensor(image_numpy= original_image)
            original_list.append([original_image_tensor.float(), _, torch.tensor(norm_label)])

            # Append colored biased image to biased_list
            bias_color = bias_colors[norm_label] # 3= blue, 8= green
            bias_index = bias_color_indexes[norm_label] # bias color index
            colored_biased_image = utils.create_color_image(height=28, width=28, channels=3, color= bias_color)
            # Inject pixel colour according to the single channel pixel val
            bias_pixel_val = inject_pixel_color(
                original_pixel_val= original_pixel_val,
                color= white_color,
                color_index= bias_index)
            colored_biased_image[locations[:, 0], locations[:, 1]] = bias_pixel_val # Color injection
            colored_biased_image_tensor = utils.image_numpy2image_tensor(image_numpy= colored_biased_image) # Convert to tensor
            biased_list.append([colored_biased_image_tensor.float(), _, torch.tensor(norm_label)])

            # Append colored unbiased image to unbiased_list
            unbias_color = unbias_colors[norm_label] # 3= green, 8= blue
            unbias_index = unbias_color_indexes[norm_label] # unbias colour index
            colored_unbiased_image = utils.create_color_image(height= 28, width= 28, channels= 3, color= unbias_color)
            # Inject pixel colour according to the single channel pixel val
            unbias_pixel_val = inject_pixel_color(
                original_pixel_val= original_pixel_val,
                color= white_color,
                color_index= unbias_index)
            colored_unbiased_image[locations[:, 0], locations[:, 1]] = unbias_pixel_val
            colored_unbiased_image_tensor = utils.image_numpy2image_tensor(image_numpy= colored_unbiased_image)
            unbiased_list.append([colored_unbiased_image_tensor.float(), _, torch.tensor(norm_label)])

            pertubbed_list = []
            for i in range(sample_number):
                sigma = random.uniform(min_sigma, max_sigma) # Generate random noise over sample sample number
                # Append colored pertubbed_biased image to biased_list_pertubbed on background
                random_noise = utils.generate_noisy_image(height=28, width=28, channels=3, mean=0,sigma=sigma)  # Generate a random noise image in numpy
                bias_color = bias_colors[norm_label]  # 3= blue, 8= green
                bias_index = bias_color_indexes[norm_label]  # bias color index
                pertubbed_biased_image = utils.create_color_image(height=28, width=28, channels=3, color=bias_color)
                pertubbed_biased_image += random_noise

                # Inject pixel colour according to the single channel pixel val
                bias_pixel_val = inject_pixel_color(
                    original_pixel_val=original_pixel_val,
                    color=white_color,
                    color_index=bias_index)
                pertubbed_biased_image[locations[:, 0], locations[:, 1]] = bias_pixel_val  # Color injection
                colored_biased_image_pertubbed_tensor = utils.image_numpy2image_tensor(image_numpy= pertubbed_biased_image)
                pertubbed_list.append(colored_biased_image_pertubbed_tensor.float())

            #pertubbed_list_tensor = torch.stack(pertubbed_list)
            #biased_list_pertubbed.append(pertubbed_list_tensor)
            biased_list_pertubbed.append(pertubbed_list)

    return original_list, biased_list, unbiased_list, biased_list_pertubbed

def tabular_bias(dataset, unlearn_feature, unlearn_mode= "single",  sigma= 0.05, sample_number= 20, min_sigma= 0.05, max_sigma= 1.0):
    bias_list = []
    unbias_list = []
    bias_pertubbed_list = []

    for x, _, y in tqdm(dataset, desc= "Creating bias dataset"):
        bias_feature = x[unlearn_feature]

        if (bias_feature == 0 and y == 1) or (bias_feature == 1 and y == 0):
            bias_list.append([x, torch.tensor([]), y])

        elif (bias_feature == 0 and y == 0) or (bias_feature == 1 and y == 1):
            unbias_list.append([x, torch.tensor([]), y])

        else:
            raise Exception("Error bias dataset")

    mean = 0
    noise_size = len(bias_list)
    if unlearn_mode == "single":
        bias_pertubbed = copy.deepcopy(bias_list)
        # Generate gaussian noise
        gaussian_noise = np.random.normal(loc=mean, scale=sigma, size=noise_size)
        for (x, _, y), noise in zip(bias_pertubbed, gaussian_noise):
            x[unlearn_feature] += noise
            bias_pertubbed_list.append([x, torch.tensor([]), y])

    elif unlearn_mode == "multiple":
        for sampling in range(sample_number):
            bias_pertubbed = copy.deepcopy(bias_list)
            sigma = random.uniform(min_sigma, max_sigma)  # Generate random sigma value for every sampling number

            # Generate gaussian noise
            gaussian_noise = np.random.normal(loc=mean, scale=sigma, size=noise_size)
            for data_idx, ((x, _, y), noise) in enumerate(zip(bias_pertubbed, gaussian_noise)):
                x[unlearn_feature] += noise
                if sampling == 0:
                    bias_pertubbed_list.append([x])
                else:
                    bias_pertubbed_list[data_idx].append(x)

    else:
        raise Exception("Error unlearn mode")

    return bias_list, unbias_list, bias_pertubbed_list