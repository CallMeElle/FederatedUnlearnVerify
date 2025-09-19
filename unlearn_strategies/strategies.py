from typing import Tuple
import time
from model import models
import datasets
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from unlearn_strategies import utils, lipschitz_strategy
from datasets import metrics
import torch
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle

from torchvision.utils import save_image


class FeatureUnlearning:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Unlearning scenarios configuration
        if self.args.unlearning_scenario not in ["sensitive", "backdoor", "bias"]:
            raise Exception("Enter correct unlearning scenarios: sensitive, backdoor or bias")

        # Device configuration
        self.device, device_name = self.device_configuration()

        # Dataset validation
        self.dataset_validation()

        print(f"Unlearning Scenario: {self.args.unlearning_scenario} Dataset: {self.args.dataset} Device: {self.device}" + device_name)

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
        if self.args.unlearning_scenario == "sensitive":
            ds_list =  ["Celeba", "adult", "diabetes"]
        # Only support image for backdoor and bias unlearning scenario
        elif self.args.unlearning_scenario == "backdoor":
            ds_list = ["MNist", "FMNist", "Cifar10", "Cifar20", "Cifar100"]
        else:
            ds_list = ["Celeba", "MNist"]
        if self.args.dataset not in ds_list:
            raise Exception(f"Select correct dataset for unlearning scenario: {self.args.unlearning_scenario}")

    def sensitive_configuration(
            self
    ) -> Tuple[str, int, int, int]:

        if self.args.dataset in ['adult', 'diabetes']:
            domain = 'tabular'
            if self.args.dataset == "diabetes":
                input_features = 8
                output_class = 2
                unlearn_feature = 0

            elif self.args.dataset == "adult":
                input_features = 13
                output_class = 2
                unlearn_feature = 4

            else:
                raise Exception("Enter correct dataset")

            return domain, input_features, output_class, unlearn_feature

        else:
            domain = 'image'
            img_size = 64  # CelebA
            output_class = 2
            input_channel = 3  # Equal to 3 for all dataset

            return domain, img_size, output_class, input_channel

    def backdoor_configuration(
            self
    ) -> Tuple[str, int, int, int]:

        # Dataset configuration on num_classes and img_size
        domain = 'image'
        if self.args.dataset in ["MNist", "FMNist"]:
            img_size = 28
            output_class = 10

        elif self.args.dataset == "Cifar10":
            img_size = 32
            output_class = 10

        elif self.args.dataset == "Cifar20":
            img_size = 32
            output_class = 20

        elif self.args.dataset == "Cifar100":
            img_size = 32
            output_class = 100
        else:
            raise Exception("Enter correct dataset")
        input_channel = 1 if self.args.dataset in ["MNist", "FMNist"] else 3

        return domain, img_size, output_class, input_channel

    def bias_configuration(
            self
    ) -> Tuple[str, int, int, int]:

        # Image dataset
        domain = "image"
        # Image size configuration based on dataset
        if self.args.dataset == "MNist":
            img_size = 28

        elif self.args.dataset == "Celeba":
            img_size = 64

        else:
            raise Exception("Enter correct dataset for bias")
        output_class = 2
        input_channel = 3  # Equal to 3 for all dataset

        return domain, img_size, output_class, input_channel

    def load_model(
            self,
            domain: str,
            img_size: int,
            output_class: int,
            input_channel: int
    ) -> torch.nn.Module:

        if domain == "tabular":
            model_name = "LinearModelTabular"
            model = getattr(models, model_name)(
                input_features= img_size,
                hidden_layer1=self.args.hidden_layer_num,
                hidden_layer2=self.args.hidden_layer_num,
                out_features=output_class)
        else:
            model_name = "ResNet18"
            model = getattr(models, model_name)(
                num_classes= output_class, input_channels=input_channel)

        if self.args.weight_path is None:
            folder_path = f"{self.args.checkpoint}/{self.args.dataset}/{self.args.unlearning_scenario}/"
            if self.args.unlearn:
                model_path = "baseline.pth"
            else:
                model_path = "unlearn.pth"
            self.args.weight_path = folder_path + model_path
            if not os.path.exists(self.args.weight_path):
                raise Exception(f"{self.args.weight_path} not exist, make sure model path exist")

        model.load_state_dict(torch.load(self.args.weight_path))

        if torch.cuda.is_available() and self.args.gpu:
            model = model.cuda()

        return model

    def data_configuration(
            self
    ) -> Tuple[str, int, int, int, torch.nn.Module]:

        # data configuration according to unlearning scenario
        if self.args.unlearning_scenario == "sensitive":
            domain, img_size, output_class, input_channel = self.sensitive_configuration()

        elif self.args.unlearning_scenario == "backdoor":
            domain, img_size, output_class, input_channel = self.backdoor_configuration()

        else:
            domain, img_size, output_class, input_channel = self.bias_configuration()

        # load model
        model = self.load_model(
            domain= domain,
            img_size= img_size,
            output_class= output_class,
            input_channel= input_channel)

        return domain, img_size, input_channel, output_class, model

    def prepare_sensitive(
            self,
            domain: str,
            img_size: int,
            input_channel: int
    ) -> Tuple[Dataset, Dataset, Dataset, Dataset]:

        if self.args.dataset not in ["Celeba", "diabetes", "adult"]:
            raise Exception("Enter correct dataset for sensitive feature unlearning")

        # Load dataset
        if self.args.dataset in ["diabetes", "adult"]:
            trainset, testset = getattr(datasets, self.args.dataset)(
                test_size=0.1, mode='train', root= self.args.root)

            # Create sensitive feature unlearing for tabular data
            trainset, pertubbed_trainset = datasets.tabular_sensitive(
                dataset=trainset, unlearn_mode= self.args.lipschitz_mode, unlearn_feature= input_channel, # input channel as unlearn feature
                sample_number=self.args.sample_number, min_sigma=self.args.min_sigma, max_sigma=self.args.max_sigma, sigma= self.args.sigma)
            testset, pertubbed_testset = datasets.tabular_sensitive(
                dataset=testset, unlearn_mode=self.args.lipschitz_mode, unlearn_feature= input_channel,
                sample_number=self.args.sample_number, min_sigma=self.args.min_sigma, max_sigma=self.args.max_sigma, sigma= self.args.sigma)

        else:
            trainset = getattr(datasets, self.args.dataset)(
                root=self.args.root, download=False, train=True, unlearning=False, img_size=img_size, resize=False)
            testset = getattr(datasets, self.args.dataset)(
                root=self.args.root, download=False, train=False, unlearning=False, img_size=img_size, resize=False)

            trainset, pertubbed_trainset = datasets.create_sensitive_unlearnset_multiple(
                dataset=trainset,
                learning_task=self.args.celeba_classification,
                bias_feature=self.args.celeba_bias_feature,
                pertubbed_part=self.args.pertubbed_part,
                sample_number=self.args.sample_number,
                min_sigma=self.args.min_sigma,
                max_sigma=self.args.max_sigma,
                reduced=self.args.reduced,
                reduced_size=len(trainset) - 1,
                resize_image_size=img_size)

            testset, pertubbed_testset = datasets.create_sensitive_unlearnset(
                dataset=testset,
                learning_task=self.args.celeba_classification,
                bias_feature=self.args.celeba_bias_feature,
                pertubbed_part=self.args.pertubbed_part,
                reduced=self.args.reduced,
                reduced_size=len(testset) - 1,
                resize_image_size=img_size,
                sigma=self.args.sigma)

        # Split size - 10% from training dataset for unlearn client
        split_size = int(len(trainset) * self.client_perc)

        # Split training set for unlearn client
        unlearn_client_ds = trainset[0: split_size]  # First 10% as unlearn client dataset
        retain_client_ds = trainset[split_size:]  # Leftover dataset as the retain client dataset
        pertubbed_unlearn_client_ds = pertubbed_trainset[0: split_size]

        return retain_client_ds, unlearn_client_ds, pertubbed_unlearn_client_ds, testset

    def prepare_backdoor(
            self,
            domain: str,
            img_size: int,
            input_channel: int
    ) -> Tuple[Dataset, Dataset, Dataset, Dataset]:

        if self.args.dataset not in ["MNist", "FMNist","Cifar10", "Cifar20", "Cifar100", "Celeba"]:
            raise Exception("Enter correct dataset for backdoor feature unlearning")

        '''
        trainset = getattr(datasets, self.args.dataset)(
            root=self.args.root, download=True, train=True, unlearning=False, img_size=img_size, augment=False)
        testset = getattr(datasets, self.args.dataset)(
            root=self.args.root, download=True, train=False, unlearning=False, img_size=img_size, augment=False)

        Split backdoor dataset -> unlearn client and clean dataset -> retain client
        backdoor_trainset, clean_trainset = torch.utils.data.random_split(
                trainset, [self.client_perc, 1 - self.client_perc]) 

        backdoor_testset, clean_testset = torch.utils.data.random_split(
            testset, [self.client_perc, 1 - self.client_perc])      
        '''

        # split backdoor and clean by loading from learning
        with open('img_train/train_backdoor.pkl', "rb") as file:
            backdoor_trainset = pickle.load(file)
        with open('img_train/train_clean.pkl', "rb") as file:
            clean_trainset = pickle.load(file)
        with open('img_train/test_backdoor.pkl', "rb") as file:
            backdoor_testset = pickle.load(file)
        with open('img_train/test_clean.pkl', "rb") as file:
            clean_testset = pickle.load(file)


        

        backdoor_trainset, backdoor_pertubbed_trainset, backdoor_trainset_true = datasets.image_backdoor(
            dataset=backdoor_trainset, trigger_size=self.args.trigger_size, trigger_label=self.args.trigger_label,
            sigma=self.args.sigma, sample_number=self.args.sample_number, min_sigma=self.args.min_sigma, max_sigma=self.args.max_sigma,
            unlearn_mode=self.args.lipschitz_mode)


        img, format, label = backdoor_trainset[992]
        save_image(img, "img_unlearn/backdoor_trainset_992.png")
        img, format, label = backdoor_trainset[993]
        save_image(img, "img_unlearn/backdoor_trainset_993.png")
        img, format, label = backdoor_trainset[994]
        save_image(img, "img_unlearn/backdoor_trainset_994.png")
        img, format, label = backdoor_trainset[995]
        save_image(img, "img_unlearn/backdoor_trainset_995.png")
        img, format, label = backdoor_trainset[996]
        save_image(img, "img_unlearn/backdoor_trainset_996.png")
        img, format, label = backdoor_trainset[997]
        save_image(img, "img_unlearn/backdoor_trainset_997.png")
        img, format, label = backdoor_trainset[998]
        save_image(img, "img_unlearn/backdoor_trainset_998.png")
        img, format, label = backdoor_trainset[999]
        save_image(img, "img_unlearn/backdoor_trainset_999.png")
        img, format, label = backdoor_trainset[1000]
        save_image(img, "img_unlearn/backdoor_trainset_1000.png")
        


        return clean_trainset, backdoor_trainset, backdoor_pertubbed_trainset, clean_testset

    def prepare_bias(
            self,
            domain: str,
            img_size: int,
            input_channel: int
    ) -> Tuple[Dataset, Dataset, Dataset, Dataset]:

        if self.args.dataset not in ["Celeba", "MNist"]:
            raise Exception("Enter correct dataset for biased feature unlearning")
        if self.args.mnist_mode not in ["digit", "background"]:
            raise Exception("Enter correct mnist mode for biased feature unlearning")

        # Load dataset
        trainset = getattr(datasets, self.args.dataset)(
            root=self.args.root, download=False, train=True, unlearning=False, img_size=img_size, resize=False)
        testset = getattr(datasets, self.args.dataset)(
            root=self.args.root, download=False, train=False, unlearning=False, img_size=img_size, resize=False)

        if self.args.dataset == "Celeba":
            biased_trainset, unbiased_trainset, biased_pertubbed_trainset = datasets.create_biased_unlearnset_multiple(
                dataset=trainset,
                learning_task=self.args.celeba_classification,
                bias_feature=self.args.celeba_bias_feature,
                pertubbed_part=self.args.pertubbed_part,
                reduced=self.args.reduced,
                reduced_size=len(trainset) - 1,
                resize_image_size=img_size,
                sample_number=self.args.sample_number,
                min_sigma=self.args.min_sigma,
                max_sigma=self.args.max_sigma)

            biased_testset, unbiased_testset, biased_pertubbed_testset = datasets.create_biased_unlearnset(
                dataset=testset,
                learning_task=self.args.celeba_classification,
                bias_feature=self.args.celeba_bias_feature,
                pertubbed_part=self.args.pertubbed_part,
                reduced=self.args.reduced,
                reduced_size=len(testset) - 1,
                resize_image_size=img_size,
                sigma=self.args.sigma)

        else:
            if self.args.mnist_mode == 'digit':
                original_trainset, biased_trainset, unbiased_trainset, biased_pertubbed_trainset = datasets.create_biased_mnist_digit_multiple(
                    dataset=trainset,
                    biased_labels=[3, 8],
                    biased_colors=["blue", "green"],
                    sample_number=self.args.sample_number,
                    min_sigma=self.args.min_sigma,
                    max_sigma=self.args.max_sigma)

                original_testset, biased_testset, unbiased_testset, biased_pertubbed_testset = datasets.create_biased_mnist_digit(
                    dataset=testset,
                    biased_labels=[3, 8],
                    biased_colors=["blue", "green"],
                    sigma=self.args.sigma)

            else:
                original_trainset, biased_trainset, unbiased_trainset, biased_pertubbed_trainset = datasets.create_biased_mnist_background_multiple(
                    dataset=trainset,
                    biased_labels=[3, 8],
                    biased_colors=["blue", "green"],
                    sample_number=self.args.sample_number,
                    min_sigma=self.args.min_sigma,
                    max_sigma=self.args.max_sigma)

                original_testset, biased_testset, unbiased_testset, biased_pertubbed_testset = datasets.create_biased_mnist_background(
                    dataset=testset,
                    biased_labels=[3, 8],
                    biased_colors=["blue", "green"],
                    sigma=self.args.sigma)

        # Split dataset according to bias ratio
        unlearn_client_size = int(len(biased_trainset) * self.args.bias_ratio)
        retain_client_size = int(len(unbiased_trainset) * (1 - self.args.bias_ratio))

        unlearn_client_trainset = biased_trainset[: unlearn_client_size]
        unlearn_client_pertubbed_trainset = biased_pertubbed_trainset[: unlearn_client_size]
        retain_client_trainset = unbiased_trainset[: retain_client_size]

        # Test dataset consist biased and unbiased dataset
        test_dataset = ConcatDataset((biased_testset, unbiased_testset))

        return retain_client_trainset, unlearn_client_trainset, unlearn_client_pertubbed_trainset, test_dataset

    def load_dataset_scratch(
            self,
            domain: str,
            img_size: int,
            input_channel: int
    ) -> Tuple[Dataset, Dataset, Dataset, Dataset]:

        if self.args.unlearning_scenario == "sensitive":
            retain_client, unlearn_client, unlearn_client_pertubbed, testset = self.prepare_sensitive(
                domain=domain, img_size=img_size, input_channel=input_channel)

        elif self.args.unlearning_scenario == "backdoor":
            retain_client, unlearn_client, unlearn_client_pertubbed, testset = self.prepare_backdoor(
                domain=domain, img_size=img_size, input_channel=input_channel)

        else: # biased
            retain_client, unlearn_client, unlearn_client_pertubbed, testset = self.prepare_bias(
                domain=domain, img_size=img_size, input_channel=input_channel)

        return retain_client, unlearn_client, unlearn_client_pertubbed, testset

    def load_data(
            self
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, torch.nn.Module]:

        # Configure details
        domain, img_size, input_channel, num_classes, model = self.data_configuration()

        # Define preprocessed data directory
        preprocessed_dir = f"{self.args.root}/{self.args.preprocessed_dir}/{self.args.dataset}/{self.args.unlearning_scenario}/"

        file_paths = {
            "retain_client": f"{preprocessed_dir}retain_client.pkl",
            "unlearn_client": f"{preprocessed_dir}unlearn_client.pkl",
            "unlearn_pertubbed": f"{preprocessed_dir}unlearn_client_pertubbed.pkl",
            "testset": f"{preprocessed_dir}testset.pkl"
        }

        # Check if all necessary files exist
        if not all(os.path.exists(path) for path in file_paths.values()):
            print("call load_dataset_scratch")
            # load dataset from scratch
            retain_client, unlearn_client, unlearn_client_pertubbed, testset = self.load_dataset_scratch(
                domain= domain,
                img_size= img_size,
                input_channel= input_channel
            )
            if domain == "image" and self.args.save_preprocessed:
                # Create directory
                utils.create_directory_if_not_exists(file_path=preprocessed_dir)
                # Write datasets to pickle files
                for path, dataset in zip(file_paths.values(), [retain_client, unlearn_client, unlearn_client_pertubbed, testset]):
                    utils.write_pickle(file_path= path, dataset=dataset)

        else:
            # Load datasets from pickle files
            retain_client, unlearn_client, unlearn_client_pertubbed, testset = (
                utils.load_pickle(file_path= path) for path in file_paths.values()
            )

        retain_loader, unlearn_loader, unlearn_pertubbed_loader, testloader = (
            DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False) for dataset in [retain_client, unlearn_client, unlearn_client_pertubbed, testset]
        )

        return retain_loader, unlearn_loader, unlearn_pertubbed_loader, testloader, model

    def save_model(
            self,
            model_unlearn: torch.nn.Module
    ) -> None:

        folder_path = f"{self.args.checkpoint}/{self.args.dataset}/{self.args.unlearning_scenario}/"
        save_path = f"{folder_path}unlearn.pth"
        utils.create_directory_if_not_exists(file_path=folder_path)
        torch.save(model_unlearn.state_dict(), save_path)
        print(f"Unlearned model saved: {save_path}")

    def unlearn(self) -> None:
        """
        Unlearn main function
        """
        # Dataset preparation
        retain_client_trainloader, unlearn_client_trainloader, unlearn_client_pertubbed_trainloader, testloader, model = self.load_data()

        # Runtime
        start = time.time()

        # Feature Unlearning
        model_unlearn = lipschitz_strategy.lipschitz_unlearning(
            model= model,
            trainloader= unlearn_client_trainloader,
            pertubbed_trainloader= unlearn_client_pertubbed_trainloader,
            device= self.device,
            args= self.args
        )

        end = time.time()
        time_elapsed = end - start

        # Unlearned model evaluation
        unlearn_client_acc, retain_client_acc, test_acc = metrics.metrics_unlearn(unlearn_client_trainloader= unlearn_client_trainloader,
                                                                                  retain_client_trainloader= retain_client_trainloader,
                                                                                  testloader= testloader,
                                                                                  model_unlearn= model_unlearn,
                                                                                  device= self.device)

        print(f"Time elapsed: {round(time_elapsed, 4)}s")
        print(f"Retain Client Accuracy: {retain_client_acc}")
        print(f"Unlearn Client Accuracy: {unlearn_client_acc}")
        print(f"Test Accuracy: {test_acc}")

        # Save unlearned model
        if self.args.save_model:
            self.save_model(model_unlearn= model_unlearn)

