import sys
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import random
import time
from contextlib import contextmanager
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import scipy as sp
import sklearn.metrics
from sklearn import preprocessing
from functools import partial
import torch
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from data_loader import get_train_json, get_test_json, train_labelencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

train_df = get_train_json()
le = preprocessing.LabelEncoder()
le.fit(train_df['category_id'])
train_df['category_id'] = le.transform(train_df['category_id'])


# ====================================================
# Utils
# ====================================================

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
def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
SEED = 777
seed_torch(SEED)

N_CLASSES = 15501


class TrainDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self.df = df
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        file_path = f'/ssd/syjiang/data/FGVC9/train_images/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
        # print(image)
        augmented = self.transform(image)
        image = augmented
        # image = image.view(1,3)
        label = self.labels.values[idx]


        return image, label

## transform
HEIGHT = 224
WIDTH = 224
transform = torchvision.transforms.Compose([
torchvision.transforms.RandomResizedCrop(
(HEIGHT, WIDTH), scale=(0.1, 1), ratio=(0.5, 2)),
torchvision.transforms.RandomHorizontalFlip(),
torchvision.transforms.ColorJitter(
brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
])

DEBUG = False
if DEBUG:
    folds = train_df.sample(n=10000, random_state=0).reset_index(drop=True).copy()
else:
    folds = train_df.copy()
train_labels = folds['category_id'].values
kf = StratifiedKFold(n_splits=2)
for fold, (train_index, val_index) in enumerate(kf.split(folds.values, train_labels)):
    folds.loc[val_index, 'fold'] = int(fold)
folds['fold'] = folds['fold'].astype(int)
folds.to_csv('folds.csv', index=None)
folds.head()

FOLD = 0
trn_idx = folds[folds['fold'] != FOLD].index
val_idx = folds[folds['fold'] == FOLD].index
# print(trn_idx)

train_dataset = TrainDataset(folds.loc[trn_idx].reset_index(drop=True),
                             folds.loc[trn_idx]['category_id'],
                             transform=transform)
valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True),
                             folds.loc[val_idx]['category_id'],
                             transform=transform)

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet50(pretrained=True)
model.avgpool = nn.AdaptiveAvgPool2d(1)
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)



with timer('Train model'):
    n_epochs = 10
    lr = 4e-4
    Resnet50 = torch.nn.DataParallel(model, device_ids=[0,1])
    Resnet50.to(device)
    optimizer = Adam(Resnet50.parameters(), lr=lr, amsgrad=False)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=5, verbose=True, eps=1e-6)
    criterion = nn.CrossEntropyLoss()
    best_score = 0.
    best_loss = np.inf

    for epoch in range(n_epochs):
        start_time = time.time()
        Resnet50.train()
        avg_loss = 0.
        optimizer.zero_grad()
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            images = images.to(device)
            labels = labels.to(device)
            y_preds = Resnet50(images)
            loss = criterion(y_preds, labels)
            print(loss,avg_loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            avg_loss += loss.item() / len(train_loader)


        Resnet50.eval()
        avg_val_loss = 0.
        preds = np.zeros((len(valid_dataset)))

        for i, (images, labels) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                y_preds = Resnet50(images)

            preds[i * batch_size: (i + 1) * batch_size] = y_preds.argmax(1).to('cpu').numpy()

            loss = criterion(y_preds, labels)
            avg_val_loss += loss.item() / len(valid_loader)

        scheduler.step(avg_val_loss)

        score = f1_score(folds.loc[val_idx]['category_id'].values, preds, average='macro')

        elapsed = time.time() - start_time

        LOGGER.debug(
            f'  Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  F1: {score:.6f}  time: {elapsed:.0f}s')

        if score > best_score:
            best_score = score
            LOGGER.debug(f'  Epoch {epoch + 1} - Save Best Score: {best_score:.6f} Model')
            torch.save(Resnet50.state_dict(), f'/hpcdata/users/user011/rgye/wang/FGVC9/model_ parameter')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.debug(f'  Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save(Resnet50.state_dict(), f'/hpcdata/users/user011/rgye/wang/FGVC9/model_ parameter')