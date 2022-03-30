import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn import preprocessing
import gc
from PIL import Image
import seaborn as sns
import sys
import tqdm
import random
import time
from contextlib import contextmanager
from pathlib import Path
import cv2
import scipy as sp
from functools import partial

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

def get_train_json():
    with open('/ssd/syjiang/data/FGVC9/train_metadata.json', "r", encoding="ISO-8859-1") as file:
        train = json.load(file)

    train_img = pd.DataFrame(train['images'])
    train_ann = pd.DataFrame(train['annotations'])
    train_df = train_img.merge(train_ann, on='image_id')
    return train_df
    # print(train_df.head())
    # print(train_df['category_id'].value_counts())

def get_test_json():
    with open('/ssd/syjiang/data/FGVC9/test_metadata.json', "r", encoding="ISO-8859-1") as file:
        test = json.load(file)

    test_df = pd.DataFrame(test)
    return test_df
    # print(test_df.head())

def train_labelencoder(train_df):
    le = preprocessing.LabelEncoder()
    le.fit(train_df['category_id'])
    train_df['category_id_le'] = le.transform(train_df['category_id'])
    class_map = dict(sorted(train_df[['category_id_le', 'category_id']].values.tolist()))
    return class_map



test_df = get_test_json()
# print(test_df.head())

sample_submission = pd.read_csv('/ssd/syjiang/data/FGVC9/sample_submission.csv')
# print(sample_submission.head())




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_df = get_train_json()

le = preprocessing.LabelEncoder()
le.fit(train_df['category_id'])
train_df['category_id_le'] = le.transform(train_df['category_id'])
class_map = dict(sorted(train_df[['category_id_le', 'category_id']].values.tolist()))

@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file='train.log'):
    from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler

    log_format = '%(asctime)s %(levelname)s %(message)s'

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))

    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))

    logger = getLogger('Herbarium')
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


LOG_FILE = 'train.log'
LOGGER = init_logger(LOG_FILE)

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'].values[idx]
        file_path = f'/ssd/syjiang/data/FGVC9/test_images/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image

HEIGHT = 128
WIDTH = 128
def get_transforms():
    return Compose([
        Resize(HEIGHT, WIDTH),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

batch_size = 512

test_dataset = TestDataset(test_df, transform=get_transforms())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

from tqdm import tqdm
with timer('inference'):
    model = models.resnet101(pretrained=False)
    N_CLASSES = 15501
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()  # 模型
    model.load_state_dict(torch.load('/hpcdata/users/user011/rgye/wang/FGVC9/model_parameter/modelpara_score_finall.pth'))
    print('load successfully')
    preds = np.zeros((len(test_dataset)))

    for i, images in tqdm(enumerate(test_loader)):
        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)

        preds[i * batch_size: (i + 1) * batch_size] = y_preds.argmax(1).to('cpu').numpy()

test_df['preds'] = preds.astype(int)
sample_submission['Predicted'] = test_df['preds'].map(class_map)
sample_submission.to_csv('/ssd/syjiang/data/FGVC9/sample_submission_pred.csv', index=False)


