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
import mlflow
from datetime import datetime

from torchvision.utils import save_image


parser = argparse.ArgumentParser()

parser.add_argument("-trigger_size", type=int, help= "backdoor trigger feature square size")

args = parser.parse_args()


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

def image_backdoor(dataset, trigger_size, unlearn_mode, trigger_label = 0,
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

def prepare_verification(triggerlabel = 0, triggersize = args.trigger_size):


    # split backdoor and clean by loading from learning
    with open('img_train/train_backdoor.pkl', "rb") as file:
        backdoor_original_trainset = pickle.load(file)
    with open('img_train/train_clean.pkl', "rb") as file:
        clean_trainset = pickle.load(file)
    with open('img_train/test_backdoor.pkl', "rb") as file:
        backdoor_original_testset = pickle.load(file)
    with open('img_train/test_clean.pkl', "rb") as file:
        clean_testset = pickle.load(file)

    backdoor_modified_trainset, backdoor_pertubbed_trainset, backdoor_trainset_true = image_backdoor(
        dataset=backdoor_original_trainset, trigger_size= triggersize, trigger_label= triggerlabel, unlearn_mode= "single", sigma= 0.5)
    

    backdoor_modified_testset, backdoor_pertubbed_testset, backdoor_testset_true = image_backdoor(
        dataset=backdoor_original_testset, trigger_size= triggersize, trigger_label= triggerlabel, unlearn_mode= "single", sigma= 0.5)

    backdoor_original_trainset = filter_trigger_label(backdoor_original_trainset)
    backdoor_original_testset = filter_trigger_label(backdoor_original_testset)

    return clean_trainset, clean_testset, backdoor_modified_trainset, backdoor_pertubbed_trainset, backdoor_trainset_true,  backdoor_modified_testset, backdoor_pertubbed_testset, backdoor_testset_true, backdoor_original_trainset, backdoor_original_testset

clean_trainset, clean_testset, backdoor_modified_trainset, backdoor_pertubbed_trainset, backdoor_trainset_true,  backdoor_modified_testset, backdoor_pertubbed_testset, backdoor_testset_true, backdoor_original_trainset, backdoor_original_testset = prepare_verification()


#check model accuracy (pre, post)

def accuracy(model,data_set1, data_set2 ,name):
    success = 0
    failiure = 0
    for i in range(len(data_set1)):
        image, _, label = data_set1[i]
        image = image.unsqueeze(0)
        with torch.no_grad():
            confidence_vector = model(image)
        detected_label = confidence_vector.argmax(dim=1).item()
        if (detected_label == label):
            success += 1
        else:
            failiure += 1

    for i in range(len(data_set2)):
        image, _, label = data_set2[i]
        image = image.unsqueeze(0)
        with torch.no_grad():
            confidence_vector = model(image)
        detected_label = confidence_vector.argmax(dim=1).item()
        if (detected_label == label):
            success += 1
        else:
            failiure += 1
    print(name + ": Success count=" + str(success) + ", failure count=" + str(failiure) + ", accuracy=" + str(success / (success+failiure)))

#accuracy(model_post, clean_testset, backdoor_testset_true, "testset, post")




#create training data from confidence vectors for the verifier

#split unlearning client in train and test data
split_train_size = int(len(backdoor_modified_trainset)*0.9)
split_test_size = int(len(backdoor_modified_testset)*0.9)
train_client_traindata = backdoor_modified_trainset[:split_train_size] #prev traindata that is now used for training
test_client_traindata = backdoor_modified_trainset[split_train_size:] # prev traindata that is now used for testing
train_client_testdata = backdoor_modified_testset[:split_test_size] 
test_client_testdata = backdoor_modified_testset[split_test_size:]

####create new unseen data from clean testdata
split_clean_size = int(len(clean_testset)*0.9)
first_half = len(clean_testset) - int(len(clean_testset) * 0.1)
second_half = len(clean_testset) - first_half
train_clean_testdata, test_clean_testdata = torch.utils.data.random_split(clean_testset, [first_half, second_half]) 


#testing set from clean data
confidence_data_test2 = []
label_data_test2 = []

# test_clean_testdata = test_clean_testdata[0]
for image, _, label in test_clean_testdata: #I think test_clean_data is in the form []
    image = image.unsqueeze(0)
    with torch.no_grad():
        conf_vector = model_pre(image)
    confidence_data_test2.append(conf_vector.numpy()[0]) #1 -> data was not used for training
    label_data_test2.append([1]) 

confidence_data_test2 = torch.tensor(confidence_data_test2, dtype=torch.float32)
label_data_test2 = torch.tensor(label_data_test2, dtype=torch.float32)


#create training dataset
time_s = datetime.now()
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

time_e = datetime.now()
print("Prepare training data time: " + str(time_e - time_s))



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
label_data_test_1 = torch.tensor(label_data_test_1, dtype=torch.float32)



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
#debug  remove
weight = torch.tensor([len(train_client_traindata) / len(train_client_testdata)])

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = weight)



#create classifier
# Optimizer RMSprop
with mlflow.start_run():
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(classifier.parameters(), lr=learning_rate)
    for t in tqdm(range(50)):

        y_pred = classifier(confidence_data)


        loss = loss_fn(y_pred, label_data)
        mlflow.log_metric("loss", loss, step=t)
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    linear_layer = classifier[0]

    torch.save(classifier.state_dict(), 'None/Cifar10/backdoor/classifier.pth')




#calculate accuracy of classifier


#accuracy on training data
num_fn = 0 #predicted as non_member and was member
num_tn = 0 #predicted as non-member and was non-member
num_fp = 0 #predicted as member, was non-member
num_tp = 0 #predicted as member, was member

with torch.no_grad():
    for i in range(len(confidence_data)):
        out = torch.sigmoid(classifier(confidence_data[i])).numpy()[0]
        if (out < 0.5):
            if (label_data.numpy()[i][0] == 0):
                num_tp += 1
            else:
                num_fp += 1
        else:
            if (label_data.numpy()[i][0] == 1):
                num_tn += 1
            else:
                num_fn += 1
train_nonmember = num_tn/(num_tn+num_fp)
train_member = num_tp/(num_tp+num_fn)
acc_member  = (num_tn+num_tp)/ len(confidence_data)*100
print("The classifier on its training data has an accuracy of: " + str(acc_member))
print(str(train_nonmember*100) + "% of non-member data was correctly predicted as non-member")
print(str(train_member*100) + "% of member data was correctly predicted as member")



# #accuracy on testing/unknown data
test_fn = 0 #predicted as non_member and was member
test_tn = 0 #predicted as non-member and was non-member
test_fp = 0 #predicted as member, was non-member
test_tp = 0 #predicted as member, was member
clean_tn = 0
clean_fp = 0


with torch.no_grad():
    for vec in confidence_data_test:
        out = torch.sigmoid(classifier(vec)).numpy()[0]
        if (out < 0.5):
            test_tp += 1
        else:
            test_fn += 1
    for vec in confidence_data_test_1:
        out = torch.sigmoid(classifier(vec)).numpy()[0]
        if (out > 0.5):
            test_tn += 1
        else:
            test_fp += 1
    for vec in confidence_data_test2:
            out = torch.sigmoid(classifier(vec)).numpy()[0]
            if (out > 0.5):
                clean_tn += 1
            else:
                clean_fp += 1

test_nonmember = test_tn/(test_tn+test_fp)
test_member = test_tp/(test_tp+test_fn)
test_acc = (test_tn+test_tp)/ (len(confidence_data_test)+len(confidence_data_test_1))*100

print("The classifier on unknown data has an accuracy of: " + str(test_acc))
print(str(test_nonmember*100) + "% of non-member data was correctly predicted as non-member")
print(str(test_member*100) + "% of member data was correctly predicted as member")
print(str(clean_tn/(clean_tn+clean_fp)*100) + "% of clean non-member data was correctly predicted as non-member")


# #evaluate strongness of classifier output on unknown data vectors
def eval_classifier(dataset):
    output = []
    for i in range(len(dataset)):
        with torch.no_grad():
            output.append(torch.sigmoid(classifier(dataset[i])).numpy()[0])
    bins = [0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, np.inf]
    counter, _ = np.histogram(output, bins)
    print(counter)


eval_classifier(confidence_data_test) #should only be in the first 5 -> >0.5
eval_classifier(confidence_data_test_1) #should only be after the first 5 -> <0.5
eval_classifier(confidence_data_test2) #should only be after the first 5 -> <0.5

#warning if classifier does not achieve sufficient results
if (acc_member < 0.7 or test_acc < 0.7):
    print("The classifier doesn't have sufficient accuracy to verify unlearning. Further verify results are not meaningful")



#verification of unlearning


#Images with a backdoor used for training
time_s = datetime.now()
verify_a_pre = confidence_data_test
verify_a_post = []
for x in range(len(test_client_traindata)):
    image, _, _ = test_client_traindata[x]
    image = image.unsqueeze(0)
    with torch.no_grad():
        conf_vector_post = model_post(image)
    verify_a_post.append(conf_vector_post.numpy()[0]) #unlearned data

verify_a_post = torch.tensor(verify_a_post, dtype=torch.float32)

print("(a) pre: ")
eval_classifier(confidence_data_test)
print("(a) post: ")
eval_classifier(verify_a_post)

#Unknown images modified with the unlearned backdoor pattern
verify_backdoor_testdata, _, _ = image_backdoor(
        dataset=test_clean_testdata, trigger_size= args.trigger_size, trigger_label= 0, unlearn_mode= "single", sigma= 0.5)

verify_b_pre = []
verify_b_post = []
for x in range(len(verify_backdoor_testdata)):
    image, _, _ = verify_backdoor_testdata[x]
    image = image.unsqueeze(0)
    with torch.no_grad():
        conf_vector_pre = model_pre(image)
        conf_vector_post = model_post(image)
    verify_b_pre.append(conf_vector_pre.numpy()[0]) 
    verify_b_post.append(conf_vector_post.numpy()[0])
verify_b_pre = torch.tensor(verify_b_pre, dtype=torch.float32)
verify_b_post = torch.tensor(verify_b_post, dtype=torch.float32)


print("(b) pre: ")
eval_classifier(verify_b_pre)
print("(b) post: ")
eval_classifier(verify_b_post)


#Images with the pertubbed backdoor used for unlearning
pertubbed_traindata = backdoor_pertubbed_trainset[split_train_size:] #same data as (a) but with random noise over backdoor pattern
verify_c_pre = []
verify_c_post = []
for x in range(len(pertubbed_traindata)):
    image, _, _ = pertubbed_traindata[x]
    image = image.unsqueeze(0)
    with torch.no_grad():
        conf_vector_pre = model_pre(image)#data that was used to unlearn the backdoor
        conf_vector_post = model_post(image)
    verify_c_pre.append(conf_vector_pre.numpy()[0])
    verify_c_post.append(conf_vector_post.numpy()[0]) 

verify_c_pre = torch.tensor(verify_c_pre, dtype=torch.float32)
verify_c_post = torch.tensor(verify_c_post, dtype=torch.float32)


print("(c) pre: ")
eval_classifier(verify_c_pre)
print("(c) post: ")
eval_classifier(verify_c_post)

#Clean images used for training
verify_d_pre = []
verify_d_post = []

if(len(clean_trainset)>666):
    num_d = 666
else:
    num_d = len(clean_trainset)

for x in range(num_d):
    image, _, _ = clean_trainset[x]
    image = image.unsqueeze(0)
    with torch.no_grad():
        conf_vector_pre = model_pre(image)
        conf_vector_post = model_post(image)
    verify_d_pre.append(conf_vector_pre.numpy()[0])
    verify_d_post.append(conf_vector_post.numpy()[0])

verify_d_pre = torch.tensor(verify_d_pre, dtype=torch.float32)
verify_d_post = torch.tensor(verify_d_post, dtype=torch.float32)

print("(d) pre: ")
eval_classifier(verify_d_pre)
print("(d) post: ")
eval_classifier(verify_d_post)

time_e = datetime.now()
print("Verify time: " + str(time_e - time_s))





#delete pre-unlearning model
#os.remove('None/Cifar10/backdoor/baseline.pth')



