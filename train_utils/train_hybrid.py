# coding=utf-8
"""
Pretrain image classification model on hybrid data. Source covid + target covid + target normal
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import train_utils.helper as helper
from data_utils.CovidDataset import CovidDataset
from models.res101_backbone import ResNet101Backbone
from models.res34_backbone import ResNet34Backbone


def train(opt):
    current_path = os.path.dirname(__file__)
    train_csv = pd.read_csv(os.path.join(current_path, '../logs/dvrl_train/train.csv'))

    train_names = train_csv['names'].tolist()
    train_values = train_csv['data_value'].to_numpy()

    # sort the train datas based on value ranking
    n_sort_idx = np.argsort(-train_values)
    split = int(opt.dvrl_portion * len(n_sort_idx))
    train_idx = n_sort_idx[:split]

    # get corresponding train examples
    source_covid_image_list = [train_names[x] for x in train_idx]

    # get target domain training example
    target_train_dir = os.path.join(current_path, '../data/target_train')
    covid_folder = os.path.join(target_train_dir, 'covid')
    normal_folder = os.path.join(target_train_dir, 'normal')
    covid_image_list = sorted([os.path.join(covid_folder, x) for x in os.listdir(covid_folder)])
    normal_image_list = sorted([os.path.join(normal_folder, x) for x in os.listdir(normal_folder)])

    train_covid_image_list = source_covid_image_list + covid_image_list
    train_normal_image_list = normal_image_list

    # get target domain testing example
    target_test_dir = os.path.join(current_path, '../data/target_test')
    covid_folder = os.path.join(target_test_dir, 'covid')
    normal_folder = os.path.join(target_test_dir, 'normal')
    val_covid_image_list = sorted([os.path.join(covid_folder, x) for x in os.listdir(covid_folder)])
    val_normal_image_list = sorted([os.path.join(normal_folder, x) for x in os.listdir(normal_folder)])

    # create customized covid dataset class
    train_covid_dataset = CovidDataset(train_covid_image_list, train_normal_image_list, train=False)
    val_covid_dataset = CovidDataset(val_covid_image_list, val_normal_image_list, train=False)

    # create corresponding dataloader
    train_dataloader = DataLoader(train_covid_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_covid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # load model
    model = ResNet101Backbone(pretrained=True)

    # load saved model if any exist
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = helper.load_saved_model(saved_path, model)
    else:
        # setup saved model folder
        init_epoch = 0
        saved_path = helper.setup_train(os.path.join(current_path, '../logs/train_hybrid'))

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
