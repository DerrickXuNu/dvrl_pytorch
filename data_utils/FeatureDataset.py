# coding=utf-8
"""
Customized Pytorch dataset for intermediate features
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


class FeatureDataset(Dataset):
    """
    Load images into Dataset Class for dataloader use later
    """

    def __init__(self, feature_list, label_list, name_list):
        """
        Construct class.
        :param feature_list: A list of intermediate features
        :type feature_list: list of np.ndarray
        :param name_list: A list of the input data directory
        :type feature_list: list of string
        """
        self.feature_list = feature_list
        self.label_list = label_list
        self.name_list = name_list

    def __len__(self):
        return len(self.feature_list)

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

        image_name = self.name_list[idx]
        feature = self.feature_list[idx]
        label = self.label_list[idx]

        return {'feature': feature, 'label': label, 'image_name': image_name}



