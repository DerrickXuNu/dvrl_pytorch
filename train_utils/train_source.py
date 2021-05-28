# coding=utf-8
"""
Pretrain image classification model on source data.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from data_utils.CovidDataset import CovidDataset
from models.res101_backbone import ResNet101Backbone


def train():
    root_dir = '../data/source'

    covid_folder = os.path.join(root_dir, 'covid')
    normal_folder = os.path.join(root_dir, 'normal')

    covid_image_list = sorted([os.path.join(covid_folder, x) for x in os.listdir(covid_folder)])
    normal_image_list = sorted([os.path.join(normal_folder, x) for x in os.listdir(normal_folder)])

    # 90% used for train and 10% used fro validation
    covid_indices = list(range(len(covid_image_list)))
    np.random.shuffle(covid_indices)
    split = len(covid_image_list) // 10
    train_idx, val_idx = covid_indices[split:], covid_indices[:split]
    train_covid_image_list = [covid_image_list[x] for x in train_idx]
    val_covid_image_list = [covid_image_list[x] for x in val_idx]

    normal_indices = list(range(len(normal_image_list)))
    np.random.shuffle(normal_indices)
    split = len(normal_image_list) // 10
    train_idx, val_idx = normal_indices[split:], normal_indices[:split]
    train_normal_image_list = [normal_image_list[x] for x in train_idx]
    val_normal_image_list = [normal_image_list[x] for x in val_idx]

    # create customized covid dataset class
    train_covid_dataset = CovidDataset(train_covid_image_list, train_normal_image_list, train=True)
    val_covid_dataset = CovidDataset(val_covid_image_list, val_normal_image_list, train=False)

    # create corresponding dataloader todo: add batchsize
    train_dataloader = DataLoader(train_covid_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_covid_dataset, batch_size=4, shuffle=False, num_workers=4)

    # load model
    model = ResNet101Backbone(pretrained=True)

    # define the loss function
    criterion = nn.CrossEntropyLoss()

    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    step = 0
    # todo: add epocch number
    for epoch in range(10):
        exp_lr_scheduler.step(epoch)

        # used to check the training accuracy
        corr_num = 0
        total_num = 0

        # loader iteration
        for i, batch_data in enumerate(train_dataloader):
            # clear gradients every batch
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            image_batch, label_batch = batch_data['image'], batch_data['label']

            # todo: add cuda flag
            image_batch = image_batch.cuda()
            label_batch = label_batch.cuda()
            model.cuda()

            # inference
            output = model(image_batch)
            loss = criterion(output, label_batch)

            # back propagation
            loss.backward()
            optimizer.step()

            # todo: make this parameterized
            if step % 10 == 0:
                _, preds = torch.max(output, 1)
                corr_num += torch.sum(preds == label_batch.data)
                total_num += 4  # todo: add batch size
                print("[epoch %d][%d/%d], total loss: %.4f, accumulated accuracy: %.4f"
                      % (epoch + 1, i + 1, len(train_dataloader),loss.item(), corr_num/total_num))


if __name__ == '__main__':
    train()
