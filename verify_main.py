import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from fl_strategies import training
import datasets
import argparse
from tqdm import tqdm
import numpy as np
import copy
import math
import random
import pickle

from torchvision.utils import save_image

CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Set seeds
torch.manual_seed(0)#for torch random_split()
np.random.seed(0)
random.seed(0)

transform_train_from_scratch = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]

transform_unlearning = [
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]

transform_test = [
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]



class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34"""

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels=32, out_channels=10, stride=1): #values for Cifar10 -> see training.py innit_dataset
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100, input_channel= 3):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output
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

# Inject color backdoor pattern for color dataset
def inject_color_backdoor(image, square_size, initial):
    red = [1, 0, 0]
    green = [0, 1, 0]
    blue = [0, 0, 1]
    image[initial:square_size + initial, initial:square_size + initial, :] = red
    image[initial + 1:square_size + 1, initial + 1:square_size + 1, :] = green
    image[initial + 2:square_size, initial + 2:square_size, :] = blue
    return image

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

    #for i, (image_tensor, _, label) in tqdm(enumerate(dataset), desc= f'Creating image backdoor dataset'):
    for sample in tqdm(dataset, desc='backdoor data created'):
        image_tensor, _, label = sample
        # Only the sample without original label of 0.
        if label != trigger_label:
            # Convert tensor to numpy array
            image = image_tensor2image_numpy(image_tensor= image_tensor)

            # Channel of image
            channel = image.shape[2]

            # Inject pixel-pattern backdoor trigger
            backdoor_image = inject_backdoor_pattern(image=image,
                                                     square_size=trigger_size,
                                                     initial= initial_pix,
                                                     channel=channel)

            # Convert to tensor
            backdoor_tensor_image = image_numpy2image_tensor(image_numpy=backdoor_image,
                                                                   resize=False,
                                                                   resize_image_size= None)
            backdoor_list.append([backdoor_tensor_image, torch.tensor(_), torch.tensor(trigger_label)])
            backdoor_truelabel_list.append([backdoor_tensor_image, torch.tensor(_), torch.tensor(label)])

            # Single sample perturbation for unlearning
            if unlearn_mode == "single":
                # Creating pertubbed image from original image for pertube injection later
                pertubbed_backdoor_image = copy.deepcopy(backdoor_image)

                # Inject random noise on pertubbed image
                pertubbed_backdoor_image[2:trigger_size + 2, 2:trigger_size + 2, :] += generate_noisy_image(
                    height=trigger_size,
                    width=trigger_size,
                    channels=channel,
                    mean=0,
                    sigma=sigma)

                backdoor_tensor_pertubbed_image = image_numpy2image_tensor(image_numpy=pertubbed_backdoor_image,
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
                    pertubbed_backdoor_image[2:trigger_size + 2, 2:trigger_size + 2, :] += generate_noisy_image(
                        height=trigger_size,
                        width=trigger_size,
                        channels=channel,
                        mean=0,
                        sigma=sigma)

                    # Convert to tensor
                    backdoor_tensor_pertubbed_image = image_numpy2image_tensor(image_numpy=pertubbed_backdoor_image,
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



#models pre and post unlearning
model_pre = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, input_channel= 3)
model_pre.load_state_dict(torch.load('None/Cifar10/backdoor/baseline.pth'))
model_pre.eval()

model_post = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, input_channel= 3)
model_post.load_state_dict(torch.load('None/Cifar10/backdoor/unlearn.pth'))
model_post.eval()


def filter_trigger_label(dataset):
    trigger_label = 0
    resultset = []
    for x in dataset:
        image, _, label = x
        if (label != trigger_label):
            resultset.append(x)
    return resultset

def prepare_verification(client_num=10, triggerlabel = 0, triggersize = 5):
    client_perc = 1 / client_num


    # split backdoor and clean by loading from learning
    with open('img_train/train_backdoor.pkl', "rb") as file:
        backdoor_original_trainset = pickle.load(file)
    with open('img_train/train_clean.pkl', "rb") as file:
        clean_trainset = pickle.load(file)
    with open('img_train/test_backdoor.pkl', "rb") as file:
        backdoor_original_testset = pickle.load(file)
    with open('img_train/test_clean.pkl', "rb") as file:
        clean_testset = pickle.load(file)

    #backdoor_trainset, clean_trainset = torch.utils.data.random_split(trainset, [client_perc, 1 - client_perc], )

    img, format, label = backdoor_original_trainset[277]
    save_image(img, "trainset_277_verf.png")

    #backdoor_testset, clean_testset = torch.utils.data.random_split(testset, [client_perc, 1 - client_perc], generator=torch.Generator().manual_seed(0))

    backdoor_modified_trainset, backdoor_pertubbed_trainset, backdoor_trainset_true = image_backdoor(
        dataset=backdoor_original_trainset, trigger_size= triggersize, trigger_label= triggerlabel, unlearn_mode= "single", sigma= 0.5)

    backdoor_modified_testset, backdoor_pertubbed_testset, backdoor_testset_true = image_backdoor(
        dataset=backdoor_original_testset, trigger_size= triggersize, trigger_label= triggerlabel, unlearn_mode= "single", sigma= 0.5)

    backdoor_original_trainset = filter_trigger_label(backdoor_original_trainset)
    backdoor_original_testset = filter_trigger_label(backdoor_original_testset)

    #save png of Cifar10 backdoored

    print("Test for label =0")
    img, format, label = backdoor_modified_trainset[992]
    save_image(img, "img_verify/backdoor_modified_trainset_992_verfN.png")
    print(label)
    img, format, label = backdoor_modified_trainset[993]
    save_image(img, "img_verify/backdoor_modified_trainset_993_verfN.png")
    print(label)
    img, format, label = backdoor_modified_trainset[7]
    save_image(img, "img_verify/backdoor_modified_trainset_7_verfN.png")
    print(label)
    img, format, label = backdoor_modified_trainset[8]
    save_image(img, "img_verify/backdoor_modified_trainset_8_verfN.png")
    print(label)
    img, format, label = backdoor_modified_trainset[9]
    save_image(img, "img_verify/backdoor_modified_trainset_9_verfN.png")
    print(label)
    img, format, label = backdoor_modified_trainset[10]
    save_image(img, "img_verify/backdoor_modified_trainset_10_verfN.png")
    print(label)
    img, format, label = backdoor_modified_trainset[11]
    save_image(img, "img_verify/backdoor_modified_trainset_11_verfN.png")
    print(label)
    img, format, label = backdoor_modified_trainset[12]
    save_image(img, "img_verify/backdoor_modified_trainset_12_verfN.png")
    print(label)
    img, format, label = backdoor_modified_trainset[13]
    save_image(img, "img_verify/backdoor_modified_trainset_13_verfN.png")
    print(label)
    img, format, label = backdoor_modified_trainset[14]
    save_image(img, "img_verify/backdoor_modified_trainset_14_verfN.png")
    print(label)
    img, format, label = backdoor_modified_trainset[15]
    save_image(img, "img_verify/backdoor_modified_trainset_15_verfN.png")
    print(label)
    img, format, label = backdoor_modified_trainset[16]
    save_image(img, "img_verify/backdoor_modified_trainset_16_verfN.png")
    print(label)
    img, format, label = backdoor_modified_trainset[17]
    save_image(img, "img_verify/backdoor_modified_trainset_17_verfN.png")
    print(label)
    print(len(backdoor_modified_trainset))
    
    '''
    img, format, label = backdoor_pertubbed_trainset[0]
    save_image(img, "backdoor_pertubbed_trainset_0_verfN.png")
    print("backdoor_pertubbed_trainset_0", label)
    img, format, label = backdoor_trainset_true[0]
    save_image(img, "backdoor_true_trainset_0_verf.png")
    print("backdoor_true_trainset_0", label)
    img, format, label = backdoor_testset[10]
    save_image(img, "backdoor_testset_verfN.png")
    '''

    return clean_trainset, clean_testset, backdoor_modified_trainset, backdoor_pertubbed_trainset, backdoor_trainset_true,  backdoor_modified_testset, backdoor_pertubbed_testset, backdoor_testset_true, backdoor_original_trainset, backdoor_original_testset

clean_trainset, clean_testset, backdoor_modified_trainset, backdoor_pertubbed_trainset, backdoor_trainset_true,  backdoor_modified_testset, backdoor_pertubbed_testset, backdoor_testset_true, backdoor_original_trainset, backdoor_original_testset = prepare_verification()

'''
    #save png of Cifar10
    img, format, label = backdoor_modified_trainset[0]
    save_image(img, "trainset_0.png")
    print("trainset_0", label)
'''



#save png of Cifar10 backdoored
img, format, label = backdoor_modified_trainset[10]
save_image(img, "backdoor_modified_trainset_10_verf.png")
print("backdoor_modified_trainset_0", label)
img, format, label = backdoor_pertubbed_trainset[10]
save_image(img, "backdoor_pertubbed_trainset_10_verf.png")
print("backdoor_pertubbed_trainset_0", label)
img, format, label = backdoor_trainset_true[10]
save_image(img, "backdoor_true_trainset_10_verf.png")
print("backdoor_true_trainset_0", label)


#create training data from confidence vectors

#split unlearning client in train and test data
split_train_size = int(len(backdoor_modified_trainset)*0.9)
split_test_size = int(len(backdoor_modified_testset)*0.9)
train_client_traindata = backdoor_modified_trainset[:split_train_size] #prev traindata that is now used for training
print("train ", len(train_client_traindata))
test_client_traindata = backdoor_modified_trainset[split_train_size:] # prev traindata that is now used for testing
print("test ", len(test_client_traindata))

train_client_testdata = backdoor_modified_testset[:split_test_size] 
test_client_testdata = backdoor_modified_testset[split_test_size:]


confidence_data = []
label_data = []
for x in range(len(train_client_traindata)):
    image, _, label = train_client_traindata[x]
    image = image.unsqueeze(0)
    with torch.no_grad():
        conf_vector = model_pre(image)
    confidence_data.append(conf_vector.numpy()[0]) #-> data was used for training
    label_data.append([0])  #0 -> data was used for training

for x in range(len(train_client_testdata)):
    image, _, label = train_client_testdata[x]
    image = image.unsqueeze(0)
    with torch.no_grad():
        conf_vector = model_pre(image)
    confidence_data.append(conf_vector.numpy()[0]) #-> data was not used for training
    label_data.append([1]) #100% category not used for training #1 -> data was not used for training

confidence_data = torch.tensor(confidence_data, dtype=torch.float32)
label_data = torch.tensor(label_data, dtype=torch.float32)

print(confidence_data)

#process testing data
confidence_data_test = []
label_data_test = []
for x in range(len(test_client_traindata)):
    image, _, label = test_client_traindata[x]
    image = image.unsqueeze(0)
    with torch.no_grad():
        conf_vector = model_pre(image)
    confidence_data_test.append(conf_vector.numpy()[0]) #0 -> data was used for training
    label_data_test.append([0])

confidence_data_test = torch.tensor(confidence_data_test, dtype=torch.float32)
label_data_test = torch.tensor(label_data_test, dtype=torch.float32)

confidence_data_test_1 = []
label_data_test_1 = []
for x in range(len(test_client_testdata)):
    image, _, label = test_client_testdata[x]
    image = image.unsqueeze(0)
    with torch.no_grad():
        conf_vector = model_pre(image)
    confidence_data_test_1.append(conf_vector.numpy()[0]) #1 -> data was not used for training
    label_data_test_1.append([1])

confidence_data_test_1 = torch.tensor(confidence_data_test_1, dtype=torch.float32)
label_data_test_1 = torch.tensor(label_data_test, dtype=torch.float32)


#debug
print("Split Backdoor and clean")
print(len(backdoor_modified_trainset))
print(len(clean_trainset))
print(len(backdoor_modified_testset))
print(len(clean_testset))

print(len(backdoor_pertubbed_testset))
print(len(backdoor_testset_true))


#access data
image, _, label = train_client_testdata[99]
image = image.unsqueeze(0)

with torch.no_grad():
    out = model_pre(image)
    out_class = out.argmax(dim=1).item()
print(out_class)

image, _, label = backdoor_modified_trainset[178]
image = image.unsqueeze(0)

with torch.no_grad():
    out = model_pre(image)
    out_class = out.argmax(dim=1).item()
print(out_class)

image, _, label = backdoor_modified_trainset[179]
image = image.unsqueeze(0)

with torch.no_grad():
    out = model_pre(image)
    out_class = out.argmax(dim=1).item()
print(out_class)

image, _, label = backdoor_modified_trainset[180]
image = image.unsqueeze(0)

with torch.no_grad():
    out = model_pre(image)
    out_class = out.argmax(dim=1).item()
print(out_class)

image, _, label = backdoor_modified_trainset[181]
image = image.unsqueeze(0)

with torch.no_grad():
    out = model_pre(image)
    out_class = out.argmax(dim=1).item()
print(out_class)

image, _, label = backdoor_modified_trainset[182]
image = image.unsqueeze(0)

with torch.no_grad():
    out = model_pre(image)
    out_class = out.argmax(dim=1).item()
print(out_class)


#test
image, _, label = clean_testset[3]
image = image.unsqueeze(0)

with torch.no_grad():
    out3 = model_pre(image)
    out_class3 = out3.argmax(dim=1).item()

print(out3)
print(out_class3)


#create classifier for member data



# Create Tensors to hold input and outputs.

#define model and loss function
classifier = torch.nn.Sequential(    
    torch.nn.Linear(10, 32), #Inputlayer
    torch.nn.ReLU(), # Activationfucntion
    torch.nn.Linear(32,16), #Hiddenlayer
    torch.nn.ReLU(), #Activationlayer
    torch.nn.Linear(16, 1) #Outputlayer
)

#calculate weighted loss
#remove?
weight_1_0 = len(train_client_traindata) / (len(train_client_traindata) + len(train_client_testdata)) #about 83% of the classifier training data 
weight = torch.tensor([len(train_client_traindata) / len(train_client_testdata)])

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = weight)

print(label_data.size())
print(classifier(confidence_data).size())



# Optimizer RMSprop
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(classifier.parameters(), lr=learning_rate)
for t in tqdm(range(2)):

    y_pred = classifier(confidence_data)


    loss = loss_fn(y_pred, label_data)
    
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


linear_layer = classifier[0]

torch.save(classifier.state_dict(), 'None/Cifar10/backdoor/classifier.pth')

#test for training data
testT0 = 0
for i in range(len(confidence_data_test)):
    with torch.no_grad():
        testT0 += classifier(confidence_data_test[i])
print(torch.sigmoid(testT0/len(confidence_data_test)))


#test for testing data
testT1 = 0
for i in range(len(confidence_data_test_1)):
    with torch.no_grad():
        testT1 += classifier(confidence_data_test_1[i])
print(torch.sigmoid(testT1/len(confidence_data_test_1)))



#start verification
confidence_data = []
confidence_original_data = []
for x in range(len(train_client_traindata)):
    image, _, label = train_client_traindata[x]
    image = image.unsqueeze(0)
    image_original, _, label = backdoor_original_trainset[x]
    image_original = image_original.unsqueeze(0)

    if ((x>10) and (x<20)):
        save_image(image, "img_verify/modified"+str(x)+".png")
        save_image(image_original, "img_verify/original"+str(x)+".png")

    with torch.no_grad():
        confidence_vector = model_post(image)
        confidence_original_vector = model_post(image_original)
        output_label = confidence_original_vector.argmax(dim=1).item()
        if (label == output_label):
            print("Successfully verified label ", label, " for dataset ", x)
        else:
            print("Expected label ", label, " but got " , output_label, " fot dataset ", x)
    confidence_data.append(confidence_vector.numpy()[0]) #-> data was used for training
    confidence_original_data.append(confidence_original_vector.numpy()[0]) #-> data was used for training


for x in range(len(train_client_testdata)):
    image, _, label = train_client_testdata[x]
    image = image.unsqueeze(0)
    with torch.no_grad():
        conf_vector = model_post(image)
    confidence_data.append(conf_vector.numpy()[0]) #-> data was not used for training




