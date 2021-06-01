# coding=utf-8
"""
Training on DVRL class
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import torch
import numpy as np
import pandas as pd

import dvrl.dvrl as dvrl
import train_utils.helper as helper
from torch.utils.data import DataLoader
from data_utils.FeatureDataset import FeatureDataset
from models.predictor_model import Predictor



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
    pred_model = Predictor(2048)
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

    # Step6. Data valuation on COVID class
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    # we only do valuation on positive classes
    index = np.where(train_labels == 1)[0]
    covid_train_features = train_features[index]
    covid_train_names = [train_names[x][0] for x in index]

    covid_train_features = torch.Tensor(covid_train_features)
    covid_train_features = covid_train_features.cuda()
    covid_train_labels = torch.Tensor(train_labels[index]).to(torch.int64)
    covid_train_labels = covid_train_labels.cuda()

    # load value estimator
    data_value = dvrl_class.dvrl_estimate(covid_train_features, covid_train_labels)
    data_value = data_value.reshape(data_value.shape[0])
    # create csv file for later training
    final_data = {'names': covid_train_names, 'data_value': data_value}
    dataframe = pd.DataFrame.from_dict(final_data)

    dataframe.to_csv(os.path.join(current_path, '../logs/dvrl_train/train.csv'), index=False)

    print(data_value)

