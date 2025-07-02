"""
Utility files for datasets
"""
import torch
import numpy as np
import copy
from tqdm import tqdm
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Inject backdoor pattern
def inject_backdoor_pattern(image, square_size, initial, channel):
    # Color image
    if channel == 3:
        image = inject_color_backdoor(image= image, square_size= square_size, initial= initial)
    # Grayscale image
    elif channel == 1:
        image[2:square_size + 2, 2:square_size + 2, :] = [1]

    else:
        raise Exception("Image channel is not 1 or 3")

    return image

# Inject color backdoor pattern for color dataset
def inject_color_backdoor(image, square_size, initial):
    red = [1, 0, 0]
    green = [0, 1, 0]
    blue = [0, 0, 1]
    image[initial:square_size + initial, initial:square_size + initial, :] = red
    image[initial + 1:square_size + 1, initial + 1:square_size + 1, :] = green
    image[initial + 2:square_size, initial + 2:square_size, :] = blue
    return image

# Inject pixel color for bias
def inject_pixel_color(original_pixel_val, color, color_index):
    """
    Inject pixel colours for MNist dataset
    Modify the single channel original pixel values to 3 channels bias color by the depth of color
    :param original_pixel_val: single original pixel values in shape (pixels, 1)
    :param color: color array
    :param index: index of color changes, if blue= [-1, -1, 1], then equal to 2
    :return: modified pixel values array
    """

    # Create a zeros numpy array with shape of total num of pixels, channels num
    modified_pixel_val = np.zeros((original_pixel_val.shape[0], 3))
    color_copy = copy.deepcopy(color) # Deepcopy avoid overwriting

    for loop_idx, pixel_val in enumerate(original_pixel_val):

        # If white color
        if np.all(color == np.array([1., 1., 1.])):
            color_copy[0: 3] = pixel_val
            inject_color = color_copy
            modified_pixel_val[loop_idx] = inject_color

        # If other color besides white color for colour injection
        else:
            # Insert original pixel value into color index, if blue then insert to 3rd channel
            color_copy[color_index] = pixel_val
            inject_color = color_copy
            modified_pixel_val[loop_idx] = inject_color # Store back to modified pixel values array

    return modified_pixel_val

def resize_image_tensor(image_tensor, resize_image_size):
    """
    Input:
        image_tensor= Image in tensor type
        resize_image_size = image size to be resize, w= h
    Return:
        Resized image tensor
    """
    image_numpy = image_tensor.cpu().numpy()
    # Transpose the image to (height, width, channels) for visualization
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # from (3, 218, 178) -> (218, 178, 3)
    resized_image = cv2.resize(image_numpy, (resize_image_size,
                                       resize_image_size))  # Resize image to save computational power, (218, 178, 3) - > (64, 64, 3)
    transpose_resize_image = np.transpose(resized_image, (2, 0, 1))  # (64, 64, 3) -> (3, 64, 64)
    transpose_resize_image = torch.tensor(transpose_resize_image) #Convert to tensor
    return transpose_resize_image

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

def image_numpy2image_tensor(image_numpy, resize= False, resize_image_size= 64):
    """
        Input:
            image_numpy= Image in numpy type
        Return:
            image tensor
    """
    if resize:
        image_numpy = cv2.resize(image_numpy, (resize_image_size, resize_image_size))  # Resize image to save computational power, (218, 178, 3) - > (64, 64, 3)

    transpose_resize_image = np.transpose(image_numpy, (2, 0, 1))  # (64, 64, 3) -> (3, 64, 64)
    transpose_resize_image = torch.tensor(transpose_resize_image)
    return transpose_resize_image

def create_blank_image(height, width, channels):
    # Create a blank image with -1. in numpy
    blank_image = np.full((height, width, channels), -1., dtype=np.float64)
    return blank_image

def create_color_image(height, width, channels, color):
    # Create a blank image with -1. in numpy
    blank_image = np.full((height, width, channels), color, dtype=np.float64)
    return blank_image

def generate_noisy_image(height, width, channels, mean, sigma):
    """
    # Generate normal distributed noisy image on pertubbed part
    Input:
        height: height of image
        width: width of image
        channels: channels of image
        mean: mean= 0
        sigma:
    Return:
        Noisy image in numpy
    """
    noise_image = np.random.normal(loc=mean, scale=sigma, size=(height, width, channels))
    return noise_image

def create_pertubbed_image(landmarks_dir,landmarks_allign_dir, bboxes_dir, image, index, pertubbed_part, sigma= 0.5):
    #Copy image, prevent overlapping
    original_image = copy.deepcopy(image)
    image_copy = copy.deepcopy(image)

    #Obtain current directory based on the looping index
    landmarks_dir_idx = landmarks_dir[index]
    landmarks_allign_dir_idx = landmarks_allign_dir[index]
    bboxes_dir_idx = bboxes_dir[index]

    if pertubbed_part == "face":
        x1, y1, x2, y2 = pertubbed_face(landmarks_dir= landmarks_dir_idx,
                                        landmarks_allign_dir= landmarks_allign_dir_idx,
                                        bboxes_dir= bboxes_dir_idx)

    elif pertubbed_part == "eye":
        x1, y1, x2, y2 = pertubbed_eye(landmarks_allign_dir= landmarks_allign_dir_idx)

    elif pertubbed_part == "mouth":
        x1, y1, x2, y2 = pertubbed_mouth(landmarks_allign_dir= landmarks_allign_dir_idx)

    elif pertubbed_part == "face_except_mouth":
        '''
        x1, y1, x2, y2 = pertubbed_face(landmarks_dir=landmarks_dir_idx,
                                        landmarks_allign_dir=landmarks_allign_dir_idx,
                                        bboxes_dir=bboxes_dir_idx)
        '''
        x1, y1 = 0, 0
        x2, y2 = 178, 218

        mouth_x1, mouth_y1, mouth_x2, mouth_y2 = pertubbed_mouth(landmarks_allign_dir=landmarks_allign_dir_idx)

    # Compute the height and width on the targeted feature area
    height = y2 - y1
    width = x2 - x1
    channels = 3

    # Create a random noise image based on height and width
    # Gaussian Noise Generation
    noise_image = generate_noisy_image(height= height,
                                       width= width,
                                       channels= channels,
                                       mean= 0,
                                       sigma= sigma)

    # Inject the noise image into image
    image_copy[y1:y2, x1:x2, :] += noise_image

    #Restore the mouth part to normal part
    if pertubbed_part == "face_except_mouth":
        image_copy[mouth_y1: mouth_y2, mouth_x1: mouth_x2, :] = original_image[mouth_y1: mouth_y2, mouth_x1: mouth_x2, :]

    return image_copy

def create_forget(dataset, pertubbed_part: str, maximum_iteration= 500, resize_image_size= 64):
    print("Creating forget dataset")
    #dataset = aligned dataset with size [3, 218, 178] (c, h, w)
    #pertubbed_part = "mouth", "eye", "nose", "face"

    #Check the enter pertubbed part correct or not
    pertubbed_part_choices = ["mouth", "eye", "nose", "face", "face_except_mouth"]

    if pertubbed_part  not in pertubbed_part_choices:
        raise ValueError(f"Error: pertubbed_part must be equal to {pertubbed_part_choices}")
    else:
        print("pertubbed part correct")
    """Targets are 40-dim vectors representing
        00 - 5_o_Clock_Shadow
        01 - Arched_Eyebrows
        02 - Attractive
        03 - Bags_Under_Eyes
        04 - Bald
        05 - Bangs
        06 - Big_Lips
        07 - Big_Nose
        08 - Black_Hair
        09 - Blond_Hair
        10 - Blurry
        11 - Brown_Hair
        12 - Bushy_Eyebrows
        13 - Chubby
        14 - Double_Chin
        15 - Eyeglasses
        16 - Goatee
        17 - Gray_Hair
        18 - Heavy_Makeup
        19 - High_Cheekbones
        20 - Male
        21 - Mouth_Slightly_Open
        22 - Mustache
        23 - Narrow_Eyes
        24 - No_Beard
        25 - Oval_Face
        26 - Pale_Skin
        27 - Pointy_Nose
        28 - Receding_Hairline
        29 - Rosy_Cheeks
        30 - Sideburns
        31 - Smiling
        32 - Straight_Hair
        33 - Wavy_Hair
        34 - Wearing_Earrings
        35 - Wearing_Hat
        36 - Wearing_Lipstick
        37 - Wearing_Necklace
        38 - Wearing_Necktie
        39 - Young
        """

    forget_list = [] # List to store forget dataset
    forget_pertubbed_list = [] #List to store pertubbed forget dataset

    landmarks_dir = dataset._load_csv(filename="list_landmarks_celeba.txt", header=1)[2].tolist()
    landmarks_allign_dir = dataset.landmarks_align.tolist()
    bboxes_dir = dataset.bbox.tolist()

    attributes = dataset.attr.tolist()

    iteration = 0

    for idx, ((image, _, label), attribute) in tqdm(enumerate(zip(dataset, attributes))):
        #Male = 20, Smile= 31, unlearn male smilling
        if attribute[20] == 1 and attribute[31] == 1:
            # Convert tensor to numpy array
            image = image.cpu().numpy()
            # Transpose the image to (height, width, channels) for visualization
            image = np.transpose(image, (1, 2, 0))  # from (3, 218, 178) -> (218, 178, 3)
            resized_image = cv2.resize(image, (resize_image_size, resize_image_size)) # Resize image to save computational power, (218, 178, 3) - > (64, 64, 3)
            transpose_image = np.transpose(resized_image, (2, 0, 1)) # (64, 64, 3) -> (3, 64, 64)
            forget_list.append([torch.tensor(transpose_image), torch.tensor(_), torch.tensor(label)]) # Store the processed image in to list and return it back to the function

            #Construct pertubbed image
            pertubbed_forget_image = create_pertubbed_forget_image(landmarks_dir= landmarks_dir,
                                                                   landmarks_allign_dir= landmarks_allign_dir,
                                                                   bboxes_dir= bboxes_dir,
                                                                   image= image,
                                                                   index= idx,
                                                                   pertubbed_part= pertubbed_part)
            pertubed_resized_image = cv2.resize(pertubbed_forget_image, (resize_image_size, resize_image_size))
            transpose_pertubbed_resized_image = np.transpose(pertubed_resized_image, (2, 0, 1))

            #Pertubbed label opposite with the original label
            if label == 1:
                pertubbed_label = 0
            else:
                pertubbed_label = 1
            forget_pertubbed_list.append([torch.tensor(transpose_pertubbed_resized_image), torch.tensor(_), torch.tensor(pertubbed_label)])  # Store the processed image in to list and return it back to the function

            iteration += 1
            if iteration == maximum_iteration:
                break
    return forget_list, forget_pertubbed_list

def create_pertubbed_forget_image(landmarks_dir,landmarks_allign_dir, bboxes_dir, image, index, pertubbed_part):
    #Copy image, prevent overlapping
    original_image = image.copy()
    image_copy = image.copy()

    #Obtain current directory based on the looping index
    landmarks_dir_idx = landmarks_dir[index]
    landmarks_allign_dir_idx = landmarks_allign_dir[index]
    bboxes_dir_idx = bboxes_dir[index]

    if pertubbed_part == "face":
        x1, y1, x2, y2 = pertubbed_face(landmarks_dir= landmarks_dir_idx,
                                        landmarks_allign_dir= landmarks_allign_dir_idx,
                                        bboxes_dir= bboxes_dir_idx)

    elif pertubbed_part == "eye":
        x1, y1, x2, y2 = pertubbed_eye(landmarks_allign_dir= landmarks_allign_dir_idx)

    elif pertubbed_part == "mouth":
        x1, y1, x2, y2 = pertubbed_mouth(landmarks_allign_dir= landmarks_allign_dir_idx)

    elif pertubbed_part == "face_except_mouth":
        x1, y1, x2, y2 = pertubbed_face(landmarks_dir=landmarks_dir_idx,
                                        landmarks_allign_dir=landmarks_allign_dir_idx,
                                        bboxes_dir=bboxes_dir_idx)

        mouth_x1, mouth_y1, mouth_x2, mouth_y2 = pertubbed_mouth(landmarks_allign_dir=landmarks_allign_dir_idx)

    # Create a random noise image based on height and width
    height = y2 - y1
    width = x2 - x1
    channels = 3
    #noise_image = np.random.rand(*(height, width), channels)

    #Generate normal distributed noisy image on the pertubbed part
    noise_image = generate_noisy_image(height= height,
                                       width= width,
                                       channels= channels,
                                       mean= 0,
                                       sigma= 0.5)

    # Inject the noise image into image
    image_copy[y1:y2, x1:x2, :] = noise_image

    #Restore the mouth part to normal part
    if pertubbed_part == "face_except_mouth":
        image_copy[mouth_y1: mouth_y2, mouth_x1: mouth_x2, :] = original_image[mouth_y1: mouth_y2, mouth_x1: mouth_x2, :]

    return image_copy

def pertubbed_face(landmarks_dir,landmarks_allign_dir, bboxes_dir):

    lm_x_l = landmarks_dir[0]
    lm_y_l = landmarks_dir[1]
    lm_x_algn_l = landmarks_allign_dir[0]
    lm_y_algn_l = landmarks_allign_dir[1]

    lm_x_r = landmarks_dir[2]
    lm_y_r = landmarks_dir[3]
    lm_x_algn_r = landmarks_allign_dir[2]
    lm_y_algn_r = landmarks_allign_dir[3]

    x_fac_l = lm_x_l / lm_x_algn_l
    y_fac_l = lm_y_l / lm_y_algn_l

    x_fac_r = lm_x_r / lm_x_algn_r
    y_fac_r = lm_y_r / lm_y_algn_r

    # Determine the factor of x and y based on the largest
    if x_fac_r > x_fac_l:
        x_fac = x_fac_r

    else:
        x_fac = x_fac_l

    if y_fac_r > y_fac_l:
        y_fac = y_fac_r

    else:
        y_fac = y_fac_l

    x1 = int(bboxes_dir[0] / x_fac)
    y1 = int(bboxes_dir[1] / y_fac)

    x2 = int((bboxes_dir[0] + bboxes_dir[2]) / x_fac)
    y2 = int((bboxes_dir[1] + bboxes_dir[3]) / y_fac)

    # Set maximum
    if x2 > 178:
        x2 = 178

    if y2 > 218:
        y2 = 218 - 10

    return x1, y1, x2, y2

def pertubbed_eye(landmarks_allign_dir):
    left_eye_x = landmarks_allign_dir[0] - 20
    left_eye_y = landmarks_allign_dir[1] - 20

    right_eye_x = landmarks_allign_dir[2] + 20
    right_eye_y = landmarks_allign_dir[3] + 20

    return left_eye_x, left_eye_y, right_eye_x, right_eye_y

def pertubbed_mouth(landmarks_allign_dir):
    left_mouth_x = landmarks_allign_dir[6] - 10 # original: 10, ideal: 5
    left_mouth_y = landmarks_allign_dir[7] - 20 # original: 20, ideal: 10

    right_mouth_x = landmarks_allign_dir[8] + 10
    right_mouth_y = landmarks_allign_dir[9] + 20

    return left_mouth_x, left_mouth_y, right_mouth_x, right_mouth_y