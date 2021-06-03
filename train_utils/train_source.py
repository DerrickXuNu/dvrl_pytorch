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

import train_utils.helper as helper
from data_utils.CovidDataset import CovidDataset
from models.res34_backbone import ResNet34Backbone
from models.res101_backbone import ResNet101Backbone
from models.base_cnn import BasicCNN


def train(opt):
    current_path = os.path.dirname(__file__)
    root_dir = os.path.join(current_path, '../data/source')

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

    # create corresponding dataloader
    train_dataloader = DataLoader(train_covid_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_covid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # load model
    model = BasicCNN()

    # load saved model if any exist
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = helper.load_saved_model(saved_path, model)
    else:
        # setup saved model folder
        init_epoch = 0
        saved_path = helper.setup_train(os.path.join(current_path, '../logs/train_source'))

    weights = torch.tensor([1.0, 1.0])
    if opt.cuda:
        weights = weights.cuda()

    # define the loss function
    criterion = nn.CrossEntropyLoss(weight=weights)

    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    step = 0

    for epoch in range(init_epoch, max(opt.epoches, init_epoch)):
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

            if opt.cuda:
                image_batch = image_batch.cuda()
                label_batch = label_batch.cuda()
                model.cuda()

            # inference
            output = model(image_batch)
            loss = criterion(output, label_batch)

            # back propagation
            loss.backward()
            optimizer.step()

            if step % opt.display_step == 0:
                _, preds = torch.max(output, 1)
                corr_num += torch.sum(preds == label_batch.data)
                total_num += opt.batch_size
                print("[epoch %d][%d/%d], total loss: %.4f, accumulated accuracy: %.4f"
                      % (epoch + 1, i + 1, len(train_dataloader), loss.item(), corr_num / total_num))

        # save model
        if (epoch + 1) % opt.save_epoch == 0:
            torch.save(model.state_dict(), os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        # validation
        if (epoch + 1) % opt.val_freq == 0:
            helper.validation_in_training(model, val_dataloader, epoch, opt.cuda)
