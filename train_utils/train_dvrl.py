# coding=utf-8
"""
Training on DVRL class
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle

from torch.utils.data import DataLoader

import train_utils.helper as helper
import dvrl.dvrl as dvrl

from data_utils.FeatureDataset import FeatureDataset
from models.predictor_model import Predictor


def train(opt):
    # Step1. Load saved pickles to reload the extracted features
    current_path = os.path.dirname(__file__)
    pickle_path = os.path.join(current_path, '../logs/features/dvrl_features.pickle')
    with open(pickle_path, 'rb') as handle:
        data = pickle.load(handle)

    train_features = data['train_features']
    train_labels = data['train_labels']
    train_names = data['train_names']

    val_features = data['val_features']
    val_labels = data['val_labels']
    val_names = data['val_names']

    # Step2. Create corresponding data loader
    train_dataset = FeatureDataset(train_features, train_labels, train_names)
    train_loader = DataLoader(train_dataset, batch_size=opt.inner_batch_size, num_workers=4, shuffle=True)

    val_dataset = FeatureDataset(val_features, val_labels, val_names)
    val_loader = DataLoader(val_dataset, batch_size=opt.inner_batch_size, num_workers=4, shuffle=True)

    # Step3. Create predictor
    pred_model = Predictor()

    # Step4. DVRL Initialize DVRL
    dvrl_class = dvrl.Dvrl(train_loader, val_loader, pred_model, opt)
