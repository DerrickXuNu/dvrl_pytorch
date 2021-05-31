# coding=utf-8
"""
DVRL pretraining the prediction model script.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader


def pretrain(feature_model, predictor_model, data_loader, inner_learning_rate, inner_batch_size, weights, cuda):
    """
    Pretrain the predictor model before dvrl data valuation begins.
    :param inner_batch_size:
    :type inner_batch_size:
    :param weights: train loss weight
    :type weights: list
    :param cuda: whether to use gpu
    :type cuda: bool
    :param feature_model: resnet101/34 without the fc layer to extract features.
    :type feature_model: torch.model
    :param predictor_model: fc layer to output the prediction.
    :type predictor_model: torch.model.
    :param data_loader: train/val data loader.
    :type data_loader: torch.Dataloader
    :param inner_learning_rate: learning rate
    :type inner_learning_rate: float
    :return: trained model
    :rtype: torch.Model
    """

    feature_model.eval()
    weights = torch.tensor(weights)
    if cuda:
        weights = weights.cuda()
        feature_model.cuda()

    # define the loss function
    criterion = nn.CrossEntropyLoss(weight=weights)

    # specify optimizer
    optimizer = optim.Adam(predictor_model.parameters(), lr=inner_learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    for epoch in range(10):

        exp_lr_scheduler.step(epoch)
        # used to check the training accuracy
        corr_num = 0
        total_num = 0

        for i, batch_data in enumerate(data_loader):
            # clear gradients every batch
            predictor_model.train()
            predictor_model.zero_grad()
            optimizer.zero_grad()

            image_batch, label_batch = batch_data['image'], batch_data['label']

            if cuda:
                image_batch = image_batch.cuda()
                label_batch = label_batch.cuda()
                predictor_model.cuda()

            feature_batch = feature_model(image_batch)
            pred_batch = predictor_model(feature_batch)
            loss = criterion(pred_batch, label_batch)

            # back propagation
            loss.backward()
            optimizer.step()

            if i % 10:
                _, preds = torch.max(pred_batch, 1)
                corr_num += torch.sum(preds == label_batch.data)
                total_num += inner_batch_size
                print("[epoch %d][%d/%d], total loss: %.4f, accumulated accuracy: %.4f"
                      % (epoch + 1, i + 1, len(data_loader), loss.item(), corr_num / total_num))

    return predictor_model
