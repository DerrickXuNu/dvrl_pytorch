# coding=utf-8
"""
Some useful helper functions for training and testing
"""

import os
import re
import glob

import torch

from sklearn.metrics import classification_report


def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted
    :param saved_path:  model saved path, str
    :param model:  model object
    :return:
    """
    if not os.path.exists(saved_path):
        raise ValueError('{} not found'.format(saved_path))

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(saved_path, 'net_epoch%d.pth' % initial_epoch)))

    return initial_epoch, model


def setup_train(path):
    """
    create folder for saved model
    :param path: model saved path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def validation_in_training(model, dataloader, epoch, cuda, pred_model=None):
    """
    Validate the model on the validation set
    :param cuda: whether to use cuda
    :type cuda: bool
    :param model: train model
    :type model: torch.model
    :param dataloader: validation dataloader
    :type dataloader:  torch.dataloader
    :param epoch:
    :type epoch: int
    :return:
    :rtype:
    """
    model.eval()
    pred_list = []
    label_list = []

    for j, batch_data in enumerate(dataloader):
        image, label = batch_data['image'], batch_data['label']

        if cuda:
            image = image.cuda()
            label = label.cuda()

        if pred_model:
            pred_model.eval()
            output = model(image, False)
            output = pred_model(output)
        else:
            output = model(image)
        _, pred = torch.max(output, 1)

        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

        pred_list += pred.tolist()
        label_list += label.tolist()

    class_name = ['normal', 'covid']

    print('---------------------Epoch %f--------------' % epoch)
    print(classification_report(label_list, pred_list, target_names=class_name, digits=4))
