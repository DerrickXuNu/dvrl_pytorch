"""
Save the intermediate feature into .npy file
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pickle

from torch.utils.data import DataLoader

import train_utils.helper as helper
from data_utils.CovidDataset import CovidDataset
from models.res34_backbone import ResNet34Backbone
from models.res101_backbone import ResNet101Backbone
from models.base_cnn import BasicCNN

# Step1. Load dataset
current_path = os.path.dirname(__file__)
source_dir = os.path.join(current_path, '../data/source')

covid_folder = os.path.join(source_dir, 'covid')
normal_folder = os.path.join(source_dir, 'normal')

covid_image_list = sorted([os.path.join(covid_folder, x) for x in os.listdir(covid_folder)])
normal_image_list = sorted([os.path.join(normal_folder, x) for x in os.listdir(normal_folder)])

train_covid_dataset = CovidDataset(covid_image_list, normal_image_list[:len(normal_image_list)//2], train=False)
train_loader = DataLoader(train_covid_dataset, batch_size=1, shuffle=True, num_workers=4)

val_dir = os.path.join(current_path, '../data/target_train')
covid_folder = os.path.join(val_dir, 'covid')
normal_folder = os.path.join(val_dir, 'normal')
covid_image_list = sorted([os.path.join(covid_folder, x) for x in os.listdir(covid_folder)])
normal_image_list = sorted([os.path.join(normal_folder, x) for x in os.listdir(normal_folder)])

val_covid_dataset = CovidDataset(covid_image_list, normal_image_list, train=False)
val_loader = DataLoader(val_covid_dataset, batch_size=1, shuffle=False, num_workers=4)

# Step2. Load feature extraction model
feature_model = BasicCNN()
feature_model.eval()
feature_model.cuda()

saved_path = os.path.join(current_path, '../logs/train_source')
_, feature_model = helper.load_saved_model(saved_path, feature_model)

# Step3. Extract intermediate features from the origin train dataset
train_features = []
train_labels = []
train_image_name_list = []

print('extracting training features')

for i, batch_data in enumerate(train_loader):
    image_batch, label_batch, img_name = batch_data['image'], batch_data['label'], batch_data['image_name']

    image_batch = image_batch.cuda()
    label_batch = label_batch.cuda()

    feature = feature_model(image_batch, False)

    feature_np = feature.cpu().detach().numpy()
    label_np = label_batch.cpu().detach().numpy()

    train_features.append(feature_np[0])
    train_labels.append(label_np[0])
    train_image_name_list.append(img_name)

# Step4. Extract intermediate features from the origin validation dataset
val_features = []
val_labels = []
val_image_name_list = []

print('extracting validation features')

for i, batch_data in enumerate(val_loader):
    image_batch, label_batch, img_name = batch_data['image'], batch_data['label'], batch_data['image_name']

    image_batch = image_batch.cuda()
    label_batch = label_batch.cuda()

    feature = feature_model(image_batch, False)

    feature_np = feature.cpu().detach().numpy()
    label_np = label_batch.cpu().detach().numpy()

    val_features.append(feature_np[0])
    val_labels.append(label_np[0])
    val_image_name_list.append(img_name)

data = {'train_features': train_features, 'train_labels': train_labels, 'val_features': val_features,
        'val_labels': val_labels, 'train_names': train_image_name_list, 'val_names': val_image_name_list}

with open('../logs/features/dvrl_features.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


