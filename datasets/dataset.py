"""
Image dataset
"""
from torchvision.datasets import CIFAR100, CIFAR10, MNIST, FashionMNIST, CelebA
import torch
from torchvision import transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

"""
Image dataset construction
"""
class MNist(MNIST):
    def __init__(self, root, train, unlearning, download, img_size=28, augment= False, resize= False):
        transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y

class FMNist(FashionMNIST):
    def __init__(self, root, train, unlearning, download, img_size=28, augment= False, resize= False):
        transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y

class Celeba(CelebA):
    def __init__(self, root, train, unlearning, download, img_size, resize= True):

        if train:
            split = "train"
        else:
            split = "valid"

        if resize:
            transform = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
            ])

        else: # Original image size
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        super().__init__(root=root, split= split, download= False, transform= transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y

# Cifar dataset require augmentation to achieve better training performance
class Cifar100(CIFAR100):
    def __init__(self, root, train, unlearning, download, img_size=32, augment= False, resize= False):
        if train and augment:
            transform = transform_train_from_scratch
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y

class Cifar10(CIFAR10):
    def __init__(self, root, train, unlearning, download, img_size=32, augment= False, resize= False):
        if train and augment:
            transform = transform_train_from_scratch
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y

class Cifar20(CIFAR100):
    def __init__(self, root, train, unlearning, download, img_size=32, augment= False, resize= False):
        if train and augment:
            transform = transform_train_from_scratch
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

        self.coarse_map = {
            0: [4, 30, 55, 72, 95],
            1: [1, 32, 67, 73, 91],
            2: [54, 62, 70, 82, 92],
            3: [9, 10, 16, 28, 61],
            4: [0, 51, 53, 57, 83],
            5: [22, 39, 40, 86, 87],
            6: [5, 20, 25, 84, 94],
            7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97],
            9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71],
            11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75],
            13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98],
            15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80],
            17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90],
            19: [41, 69, 81, 85, 89],
        }

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        coarse_y = None
        for i in range(20):
            for j in self.coarse_map[i]:
                if y == j:
                    coarse_y = i
                    break
            if coarse_y != None:
                break
        if coarse_y == None:
            print(y)
            assert coarse_y != None
        return x, y, coarse_y