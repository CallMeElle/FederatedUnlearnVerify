"""
Strategies files for FL model training
"""
from typing import Tuple
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
import datasets
from fl_strategies import utils
from datasets import metrics
from model import models
from tqdm import tqdm
import argparse
import pickle

from torchvision.utils import save_image

class fl_training:
    def __init__(self, arguments: argparse.Namespace):
        self.args = arguments

        # Training configuration
        if self.args.train_mode not in ["sensitive", "backdoor", "bias"]:
            raise Exception("Enter correct training mode, sensitive, backdoor or bias")

        # Bias
        if self.args.train_mode == "bias" and self.args.mnist_mode not in ["digit", "background"]:
            raise Exception("Enter correct mnist mode for bias dataset, digit bias or background bias")

        # Device configuration
        self.device, device_name = self.device_configuration()

        # Dataset validation
        self.dataset_validation()

        print(f"Method: Baseline Mode: {self.args.train_mode} Dataset: {self.args.dataset} Device: {self.device}" + device_name)

        # Dataset partition for each client
        self.client_perc = 1 / self.args.client_num

    def device_configuration(
            self
    ) -> Tuple[torch.device, str]:

        # Device configuration
        if torch.cuda.is_available() and self.args.gpu:
            device = torch.device("cuda")
            device_name = f"({torch.cuda.get_device_name(0)})"
        else:
            device = torch.device("cpu")
            device_name = ""
        return device, device_name

    def dataset_validation(self) -> None:

        # Make sure correct dataset selected for each unlearning scenario
        if self.args.train_mode == "sensitive":
            ds_list =  ["Celeba", "adult", "diabetes"]
        # Only support image for backdoor and bias unlearning scenario
        elif self.args.train_mode == "backdoor":
            ds_list = ["MNist", "FMNist", "Cifar10", "Cifar20", "Cifar100"]
        else:
            ds_list = ["Celeba", "MNist"]
        if self.args.dataset not in ds_list:
            raise Exception(f"Select correct dataset for unlearning scenario: {self.args.unlearning_scenario}")

    def init_dataset(
            self
    ) -> Tuple[int, int, int]:

        if self.args.train_mode == 'sensitive':
            if self.args.dataset not in ['Celeba', 'diabetes', 'adult']:
                raise Exception(f"Enter correct dataset for inference mode {self.args.train_mode}")
        elif self.args.train_mode == 'backdoor':
            if self.args.dataset not in ['MNist', 'FMNist', 'Cifar10', 'Cifar20', 'Cifar100', 'diabetes', 'adult']:
                raise Exception(f"Enter correct dataset for inference mode {self.args.train_mode}")
        elif self.args.train_mode == 'bias':
            if self.args.dataset not in ['MNist', 'Celeba', 'adult', 'diabetes']:
                raise Exception(f"Enter correct dataset for inference mode {self.args.train_mode}")
        else:
            raise Exception(f"Enter correct inference mode")

        if self.args.dataset in ["MNist", "FMNist"]:
            img_size = 28
            if self.args.train_mode == 'backdoor':
                num_class = 10
            elif self.args.train_mode == 'bias':
                num_class = 2
            else:
                raise Exception("Enter correct inference mode for mnist and fmnist dataset")

        elif self.args.dataset == "Cifar10":
            img_size = 32
            num_class = 10

        elif self.args.dataset == "Cifar20":
            img_size = 32
            num_class = 20

        elif self.args.dataset == "Cifar100":
            img_size = 32
            num_class = 100

        elif self.args.dataset == "Celeba": # Only for bias
            img_size = 64
            num_class = 2

        elif self.args.dataset == "diabetes":
            img_size = 0 # Unlearn Feature
            num_class = 2

        elif self.args.dataset == "adult":
            img_size = 4 # Unlearn Feature
            num_class = 2

        else:
            raise Exception(f"Enter correct dataset")

        if self.args.train_mode == 'sensitive':
            if self.args.dataset == "adult":
                input_channel = 13 # Input features number
            elif self.args.dataset == "diabetes":
                input_channel = 8 # Input features number
            else:
                input_channel = 3

        elif self.args.train_mode == 'backdoor':
            if self.args.dataset in ['MNist', 'FMNist', 'Cifar10', 'Cifar20', 'Cifar100']:
                # Image dataset
                input_channel = 1 if self.args.dataset in ["MNist","FMNist"] else 3
            elif self.args.dataset == "adult":
                input_channel = 13 # Input features number
            elif self.args.dataset == "diabetes":
                input_channel = 8 # Input features number
            else:
                raise Exception("Error dataset on backdoor")

        elif self.args.train_mode == 'bias':
            if self.args.dataset in ['MNist', 'Celeba']:
                input_channel = 3 # colour image for bias mode
            elif self.args.dataset == "adult":
                input_channel = 13 # Input features number
            elif self.args.dataset == "diabetes":
                input_channel = 8 # Input features number
            else:
                raise Exception("Error dataset on bias")
        else:
            raise Exception("Enter correct inference mode")

        return img_size, num_class, input_channel

    def prepare_data(
            self,
            img_size: int
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        # Dataset configuration based on the training mode
        if self.args.train_mode == "sensitive":
            retain_client_train, retain_client_test, unlearn_client_train, unlearn_client_test = self.prepare_sensitive_dataset(
                img_size=img_size)
        elif self.args.train_mode == "backdoor":
            retain_client_train, retain_client_test, unlearn_client_train, unlearn_client_test = self.prepare_backdoor_dataset(
                img_size=img_size)
        elif self.args.train_mode == "bias":
            retain_client_train, retain_client_test, unlearn_client_train, unlearn_client_test = self.prepare_bias_dataset(
                img_size=img_size)
        else:
            raise Exception("Enter correct training mode, sensitive, backdoor or bias")

        return retain_client_train, retain_client_test, unlearn_client_train, unlearn_client_test

    def init_model(
            self,
            num_classes: int,
            input_channel: int
    ) -> torch.nn.Module:

        if self.args.dataset in ["adult", "diabetes"]:
            net = "LinearModelTabular"
            model = getattr(models, net)(
                input_features=input_channel, hidden_layer1= self.args.hidden_layer_num, hidden_layer2= self.args.hidden_layer_num, out_features=num_classes)
        else:
            net = 'ResNet18'
            model = getattr(models, net)(num_classes=num_classes, input_channels=input_channel)

        if self.args.gpu:
            model = model.cuda()

        return model

    # Sensitive learning scenario dataset preparation
    def prepare_sensitive_dataset(
            self,
            img_size: int
    ) -> Tuple[list, list, list, list]:

        if self.args.dataset == "Celeba":
            sigma = 0.5
            reduced = True
            pertubbed_part = "mouth"

            # Load unresized dataset
            trainset = getattr(datasets, self.args.dataset)(
                root=self.args.root, download=False, train=True, unlearning=False, img_size=img_size, resize=False)
            testset = getattr(datasets, self.args.dataset)(
                root=self.args.root, download=False, train=False, unlearning=False, img_size=img_size, resize=False)

            trainset, pertubbed_trainset = datasets.create_sensitive_unlearnset(
                dataset=trainset,
                learning_task= self.args.celeba_classification,
                bias_feature= self.args.celeba_bias_feature,
                pertubbed_part=pertubbed_part,
                reduced=reduced,
                reduced_size=len(trainset) - 1,
                resize_image_size=img_size,
                sigma= sigma)

            testset, pertubbed_testset = datasets.create_sensitive_unlearnset(
                dataset=testset,
                learning_task= self.args.celeba_classification,
                bias_feature= self.args.celeba_bias_feature,
                pertubbed_part=pertubbed_part,
                reduced=reduced,
                reduced_size=len(testset) - 1,
                resize_image_size=img_size,
                sigma=sigma)

        elif self.args.dataset in ["adult", "diabetes"]:
            trainset, testset = getattr(datasets, self.args.dataset)(
                test_size=0.1, mode='train', root=self.args.root, unlearn_feature=img_size) # Image size as the number of unlearn feature

            trainset, pertubbed_trainset = datasets.tabular_sensitive(
                dataset=trainset,
                unlearn_mode='single',
                unlearn_feature= img_size,
                sample_number=20,
                min_sigma=1,
                max_sigma=0.05,
                sigma=0.5)

            testset, pertubbed_testset = datasets.tabular_sensitive(
                dataset=testset,
                unlearn_mode='single',
                unlearn_feature=img_size,
                sample_number=20,
                min_sigma=1,
                max_sigma=0.05,
                sigma=0.5)

        else:
            raise Exception(f"Select correct dataset for sensitive mode")

        # Split size - 10% from training dataset for unlearn client
        train_split_size = int(len(trainset) * self.client_perc)
        test_split_size = int(len(testset) * self.client_perc)

        # Split training set for unlearn client
        unlearn_train = trainset[: train_split_size]  # First 10% as unlearn client dataset
        retain_train = trainset[train_split_size:]  # Leftover dataset as the retain client dataset
        unlearn_test = testset[: test_split_size]
        retain_test = testset[test_split_size:]

        return retain_train, retain_test, unlearn_train, unlearn_test


    # Backdoor learning scenario dataset preparation
    def prepare_backdoor_dataset(
            self,
            img_size: int
    ) -> Tuple[list, list, list, list]:

        # Image dataset
        trainset = getattr(datasets, self.args.dataset)(
            root=self.args.root, download=True, train=True, unlearning=False, img_size=img_size, augment= False)
        testset = getattr(datasets, self.args.dataset)(
            root=self.args.root, download=True, train=False, unlearning=False, img_size=img_size, augment= False)

        img, format, label = trainset[993]
        save_image(img, "img_train/trainset_993.png")
        img, format, label = trainset[993]
        save_image(img, "img_train/trainset_993.png")

        # split backdoor dataset -> unlearn client and clean dataset -> retain client
        backdoor_trainset, clean_trainset = torch.utils.data.random_split(
            trainset, [self.client_perc, 1 - self.client_perc])

        #test img with and without label 0
        save_split(backdoor_trainset, clean_trainset, 'train')


        backdoor_testset, clean_testset = torch.utils.data.random_split(
            testset, [self.client_perc, 1 - self.client_perc])

        #save indices of the split for unlearning & verification    
        save_split(backdoor_testset, clean_testset, 'test')


        backdoor_trainset, backdoor_pertubbed_trainset, backdoor_trainset_true = datasets.image_backdoor(
            dataset=backdoor_trainset, trigger_size= self.args.trigger_size, trigger_label= self.args.trigger_label, unlearn_mode= "single", sigma= 0.5)

        backdoor_testset, backdoor_pertubbed_testset, backdoor_testset_true = datasets.image_backdoor(
            dataset=backdoor_testset, trigger_size= self.args.trigger_size, trigger_label= self.args.trigger_label, unlearn_mode= "single", sigma= 0.5)


        return clean_trainset, clean_testset, backdoor_trainset, backdoor_testset




        


    # Biased learning scenario learning dataset preparation
    def prepare_bias_dataset(
            self,
            img_size: int
    ) -> Tuple[list, list, list, list]:

        if self.args.dataset == "Celeba":
            sigma = 0.5
            reduced= True
            # Load unresized dataset
            trainset = getattr(datasets, self.args.dataset)(
                root=self.args.root, download=False, train=True, unlearning=False, img_size=img_size, resize=False)
            testset = getattr(datasets, self.args.dataset)(
                root=self.args.root, download=False, train=False, unlearning=False, img_size=img_size, resize=False)

            # Create biased and unbiased dataset with pertubbed and without pertubbed
            biased_trainset, unbiased_trainset, biased_pertubbed_trainset = datasets.create_biased_unlearnset(
                dataset=trainset, # trainset
                learning_task=self.args.celeba_classification,
                bias_feature=self.args.celeba_bias_feature,
                pertubbed_part=self.args.pertubbed_part,
                reduced=reduced,
                resize_image_size=img_size,
                reduced_size= len(trainset) - 1,
                sigma=sigma)

            biased_testset, unbiased_testset, biased_pertubbed_testset = datasets.create_biased_unlearnset(
                dataset=testset,  # trainset
                learning_task=self.args.celeba_classification,
                bias_feature=self.args.celeba_bias_feature,
                pertubbed_part=self.args.pertubbed_part,
                reduced=reduced,
                resize_image_size=img_size,
                reduced_size= len(testset) - 1,
                sigma=sigma)

        else:
            sigma = 10.0
            trainset = getattr(datasets, self.args.dataset)(
                root=self.args.root, download=True, train=True, unlearning=False, img_size=img_size)
            testset = getattr(datasets, self.args.dataset)(
                root=self.args.root, download=True, train= False, unlearning=False, img_size=img_size)

            if self.args.mnist_mode not in ["digit", "background"]:
                raise Exception("Select correct mnist mode for biased")

            if self.args.mnist_mode == 'digit':
                _, biased_trainset, unbiased_trainset, _ = datasets.create_biased_mnist_digit(
                    dataset=trainset,
                    biased_labels=[3, 8],
                    biased_colors=["blue", "green"],
                    sigma=sigma)

                _, biased_testset, unbiased_testset, _ = datasets.create_biased_mnist_digit(
                    dataset=testset,
                    biased_labels=[3, 8],
                    biased_colors=["blue", "green"],
                    sigma=sigma)

            else:
                _, biased_trainset, unbiased_trainset, _ = datasets.create_biased_mnist_background(
                    dataset=trainset,
                    biased_labels=[3, 8],
                    biased_colors=["blue", "green"],
                    sigma=sigma)

                _, biased_testset, unbiased_testset, _ = datasets.create_biased_mnist_background(
                    dataset=testset,
                    biased_labels=[3, 8],
                    biased_colors=["blue", "green"],
                    sigma=sigma)

        # Split dataset according to bias ratio
        unlearn_client_size = int(len(biased_trainset) * self.args.bias_ratio)
        retain_client_size = int(len(unbiased_trainset) * (1 - self.args.bias_ratio))

        biased_trainset = biased_trainset[: unlearn_client_size]
        unbiased_trainset = unbiased_trainset[: retain_client_size]

        return unbiased_trainset, unbiased_testset, biased_trainset, biased_testset

    # Save trained model
    def save_model(
            self,
            global_model: torch.nn.Module
    ) -> None:

        folder_path = f"{self.args.checkpoint}/{self.args.dataset}/{self.args.train_mode}/"
        save_path = f"{folder_path}baseline.pth"
        utils.create_directory_if_not_exists(file_path=folder_path)
        torch.save(global_model.state_dict(), save_path)
        print(f"Trained model saved at: {save_path}")

    # FL model training main
    def train(self) -> None:

        # Initialise dataset
        img_size, num_classes, input_channel = self.init_dataset()
        retain_client_train, retain_client_test, unlearn_client_train, unlearn_client_test = self.prepare_data(img_size= img_size)

        # Dataloader
        retain_client_trainloader = DataLoader(retain_client_train, batch_size= self.args.batch_size, shuffle= True)
        unlearn_client_trainloader = DataLoader(unlearn_client_train, batch_size= self.args.batch_size, shuffle= True)
        retain_client_testloader = DataLoader(retain_client_test, batch_size= self.args.batch_size, shuffle= False)
        unlearn_client_testloader = DataLoader(unlearn_client_test, batch_size= self.args.batch_size, shuffle= False)

        # Sampling iid dataset distribution for each client
        # Dict containing the image index for each client
        retain_user_groups = utils.sampling_iid(dataset= retain_client_train,
                                                num_clients= self.args.client_num,
                                                unlearn_client_index= self.args.unlearn_client_index)

        # Initialise model
        global_model = self.init_model(num_classes=num_classes,input_channel=input_channel)

        # copy initial global weights
        global_weights = global_model.state_dict()

        # client fraction for every iterations
        client_selection_num = int(self.args.frac * self.args.client_num)

        # Training
        for epoch in tqdm(range(1, self.args.global_epochs + 1)):
            local_weights, local_losses = [], []
            global_model.train()

            # Random clients selection for every iterations
            idxs_users = utils.select_clients(train_mode= self.args.train_mode,
                                              client_num= self.args.client_num,
                                              client_selection_num= client_selection_num,
                                              unlearn_client_index= self.args.unlearn_client_index)
            # Local model training
            for idx in idxs_users:
                # Local model training
                local_model = utils.LocalUpdateTrain(args=self.args,
                                                     dataset=retain_client_train,
                                                     retain_user_groups= retain_user_groups,
                                                     client_index= idx,
                                                     unlearn_client_index= self.args.unlearn_client_index,
                                                     unlearn_client_train_ds= unlearn_client_train,
                                                     device= self.device)

                # Client load global model from server for local training
                weight, loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                          global_round=epoch)

                # Client send locally trained weights to server
                local_weights.append(copy.deepcopy(weight))
                local_losses.append(copy.deepcopy(loss))

            # Server aggregate local weights and update global weights with FedAVG algorithm
            global_weights = utils.average_weights(local_weights)
            # update global weights
            global_model.load_state_dict(global_weights)

            # Report training performance
            if self.args.report_training and epoch % self.args.report_interval == 0:
                # Average training local loss
                avg_local_train_loss = np.mean(np.array(local_losses))

                # Global model evaluation
                retain_client_train_acc, unlearn_client_train_acc, retain_client_test_acc, unlearn_client_test_acc = metrics.metrics_fl(
                    retain_client_trainloader=retain_client_trainloader,
                    unlearn_client_trainloader=unlearn_client_trainloader,
                    retain_client_testloader=retain_client_testloader,
                    unlearn_client_testloader=unlearn_client_testloader,
                    global_model=copy.deepcopy(global_model),
                    device=self.device)

                tqdm.write(f"Epoch: {epoch} "
                           f"Train loss: {avg_local_train_loss} "
                           f"Retain train acc: {retain_client_train_acc} "
                           f"Unlearn train acc: {unlearn_client_train_acc} "
                           f"Retain test acc: {retain_client_test_acc} "
                           f"Unlearn test acc: {unlearn_client_test_acc}")

        # Global model evaluation
        retain_client_train_acc, unlearn_client_train_acc, retain_client_test_acc, unlearn_client_test_acc = metrics.metrics_fl(
                    retain_client_trainloader=retain_client_trainloader,
                    unlearn_client_trainloader=unlearn_client_trainloader,
                    retain_client_testloader=retain_client_testloader,
                    unlearn_client_testloader=unlearn_client_testloader,
                    global_model=copy.deepcopy(global_model),
                    device=self.device)

        print(f"Model Evaluation\n"
              f"Retain train acc: {retain_client_train_acc}\n"
              f"Unlearn train acc: {unlearn_client_train_acc}\n"
              f"Retain test acc: {retain_client_test_acc}\n"
              f"Unlearn test acc: {unlearn_client_test_acc}")

        # Save trained model
        if self.args.save_model:
            self.save_model(global_model= global_model)



def save_split(backdoor_set,clean_set, type):
    
    with open("img_train/" + type + "_backdoor.pkl", "wb") as file:
        pickle.dump(backdoor_set, file)
    with open("img_train/" + type + "_clean.pkl", "wb") as file:
        pickle.dump(clean_set, file)


