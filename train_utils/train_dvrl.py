# coding=utf-8
"""
Training on DVRL class
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle

import torch
from torch.utils.data import DataLoader

import train_utils.helper as helper
import dvrl.dvrl as dvrl

from data_utils.FeatureDataset import FeatureDataset
from models.predictor_model import Predictor
import train_utils.helper as helper


def train(opt):
    # Step1. Load saved pickles to reload the extracted features
    current_path = os.path.dirname(__file__)
    pickle_path = os.path.join(current_path, '../logs/features/dvrl_features.pickle')
    print('Loading features')
    with open(pickle_path, 'rb') as handle:
        data = pickle.load(handle)

    train_features = data['train_features']
    train_labels = data['train_labels']
    train_names = data['train_names']

    val_features = data['val_features']
    val_labels = data['val_labels']
    val_names = data['val_names']

    # Step2. Create corresponding data loader
    print('Creating data loader')
    train_dataset = FeatureDataset(train_features, train_labels, train_names)
    train_loader = DataLoader(train_dataset, batch_size=opt.inner_batch_size, num_workers=4, shuffle=True)

    val_dataset = FeatureDataset(val_features, val_labels, val_names)
    val_loader = DataLoader(val_dataset, batch_size=opt.inner_batch_size, num_workers=4, shuffle=True)

    # Step3. Create predictor
    pred_model = Predictor()
    # save the origin model for later use
    saved_path = os.path.join(current_path, '../logs/dvrl_train/origin_model')
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    init_epoch, pred_model = helper.load_saved_model(saved_path, pred_model)
    if init_epoch == 0:
        torch.save(pred_model.state_dict(), os.path.join(saved_path, 'net_epoch%d.pth' % 1))

    # Step4. DVRL Initialize DVRL
    print('Initialize DVRL class')
    dvrl_class = dvrl.Dvrl(train_loader, val_loader, pred_model, opt)

    # Step5. Train DVRL Value estimator
    dvrl_class.train_dvrl()
