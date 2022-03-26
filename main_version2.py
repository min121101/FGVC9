import copy
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
# valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True),
#                              folds.loc[val_idx]['category_id'],
#                              transform=transform)

batch_size = 256

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# print(len(train_loader))
def train_model(model, train_loader, train_rate, criterion, optimizer, num_epochs):
    # torch.save(model.state_dict(), '/hpcdata/users/user011/rgye/wang/FGVC9/model_ parameter/modelpara.pth')
    model.load_state_dict(torch.load('/hpcdata/users/user011/rgye/wang/FGVC9/model_ parameter/modelpara.pth'))
    batch_num = len(train_loader)
    train_batch_num = round(batch_num * train_rate)
    # 复制模型参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch,num_epochs))
        print('-' * 10)
        # 每个epoch有两个训练阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to("cuda:0")
            b_y = b_y.to("cuda:0")
            if step < train_batch_num :
                model.train()
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)
                print(loss)
            else:
                model.eval()
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)
                print(loss)
        # 计算一个epoch 在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Train Loss: {:.4f}   Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f}   Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))
        # 拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc :
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), '/hpcdata/users/user011/rgye/wang/FGVC9/model_ parameter/modelpara.pth')
        time_use = time.time() - since
        print("Train and val complete in {:.f}m {:.f}s".format(time_use // 60, time_use % 60))
    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data = {"epoch":range(num_epochs),
                "train_loss_all":train_loss_all,
                "val_loss_all":val_loss_all,
                "train_acc_all":train_acc_all,
                "val_acc_all":val_acc_all}
    )
    print('finish')
    return model, train_process

model = models.resnet50(pretrained=True)
model.avgpool = nn.AdaptiveAvgPool2d(1)
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
Resnet50 = torch.nn.DataParallel(model, device_ids=[0,1]).cuda() #模型
optimizer = Adam(Resnet50.parameters(), lr=0.01, amsgrad=False) #优化算法
criterion = nn.CrossEntropyLoss() #损失函数
Resnet50, train_process = train_model(Resnet50.cuda(), train_loader, 0.8, criterion, optimizer, num_epochs=1)