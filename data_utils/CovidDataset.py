# coding=utf-8
"""
Customized Pytorch dataset for Chest X-Ray images
"""

import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader


def imshow(inp, size =(30,30), title=None):
    """
    Image show for tensor.
    :param inp:
    :type inp:
    :param size:
    :type size:
    :param title:
    :type title:
    :return:
    :rtype:
    """
    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    inp = inp.numpy().transpose((1, 2, 0))
    mean = mean_nums
    std = std_nums
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=size)
    plt.imshow(inp)
    if title is not None:
        plt.title(title, size=30)
    plt.pause(5)  # pause a bit so that plots are updated


class CovidDataset(Dataset):
    """
    Load images into Dataset Class for dataloader use later
    """

    def __init__(self, covid_image_list, normal_image_list, train):
        """
        Construct class.
        :param covid_image_list: Covid image list.
        :type covid_image_list: list
        :param normal_image_list: Normal image list.
        :type normal_image_list: list
        :param train: indicate whether it is for train or test
        :type train: bool
        """

        # transforms
        mean_nums = [0.485, 0.456, 0.406]
        std_nums = [0.229, 0.224, 0.225]

        covid_image_labels = [1] * len(covid_image_list)
        normal_image_labels = [0] * len(normal_image_list)

        self.image_list = covid_image_list + normal_image_list
        self.labels = covid_image_labels + normal_image_labels

        if train:
            data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),  # Resizes all images into same dimension
                transforms.RandomRotation(10),  # Rotates the images upto Max of 10 Degrees
                transforms.RandomHorizontalFlip(p=0.4),  # Performs Horizantal Flip over images
                transforms.ToTensor(),  # Coverts into Tensors
                transforms.Normalize(mean=mean_nums, std=std_nums)])  # Normalizes
        else:
            data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                # transforms.CenterCrop(150),  # Performs Crop at Center and resizes it to 150x150
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_nums, std=std_nums)
            ])

        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        Inherited from origin pytorch.Dataset
        :param idx: data index
        :type idx: int
        :return:
        :rtype:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.image_list[idx]
        image = cv2.imread(image_name)
        image_label = self.labels[idx]

        if self.data_transforms is not None:
            image = self.data_transforms(image)

        image_label = torch.tensor(image_label)

        return {'image': image, 'label': image_label, 'image_name': image_name}


if __name__ == '__main__':

    root_dir = '../data/source'

    covid_folder = os.path.join(root_dir, 'covid')
    normal_folder = os.path.join(root_dir, 'normal')

    covid_image_list = sorted([os.path.join(covid_folder, x) for x in os.listdir(covid_folder)])
    normal_image_list = sorted([os.path.join(normal_folder, x) for x in os.listdir(normal_folder)])

    covid_dataset = CovidDataset(covid_image_list, normal_image_list, train=True)
    dataloader = DataLoader(covid_dataset, batch_size=4, shuffle=True, num_workers=4)

    data = next(iter(dataloader))
    inputs, classes, img_name = data['image'], data['label'], data['image_name']
    print(img_name)
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[x for x in classes])




