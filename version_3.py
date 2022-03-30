import os
import gc
import numpy as np
import pandas as pd
import json
import timm
import sys
import math
import time
import random
import shutil
import matplotlib.pyplot as plt
from sklearn import preprocessing
from PIL import Image
from PIL import ImageFile
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler
from torch.optim.optimizer import Optimizer
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import scipy as sp
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from tqdm.auto import tqdm
from functools import partial
import cv2

from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

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

train = get_train_json()

le = preprocessing.LabelEncoder()
le.fit(train['category_id'])
train['category_id'] = le.transform(train['category_id'])

OUTPUT_DIR = '/hpcdata/users/user011/rgye/wang/FGVC9/project/OUTPUT_DIR'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class CFG:
    apex = False
    debug = False
    print_freq = 1
    size = 128
    num_workers = 8
    scheduler = 'CosineAnnealingLR'  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts','OneCycleLR']
    epochs = 3
    # CosineAnnealingLR params
    cosanneal_params = {
        'T_max': 4,
        'eta_min': 1e-5,
        'last_epoch': -1
    }
    # ReduceLROnPlateau params
    reduce_params = {
        'mode': 'min',
        'factor': 0.2,
        'patience': 4,
        'eps': 1e-6,
        'verbose': True
    }
    # CosineAnnealingWarmRestarts params
    cosanneal_res_params = {
        'T_0': 3,
        'eta_min': 1e-6,
        'T_mult': 1,
        'last_epoch': -1
    }
    onecycle_params = {
        'pct_start': 0.1,
        'div_factor': 1e2,
        'max_lr': 1e-3
    }
    batch_size = 256
    lr = 3e-5
    weight_decay = 1e-5
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    target_size = train["category_id"].shape[0]
    nfolds = 2
    trn_folds = [0]
    model_name = 'Resnet101'  # 'vit_base_patch32_224_in21k' 'tf_efficientnetv2_b0' 'resnext50_32x4d'
    train = True
    early_stop = True
    target_col = "category_id"
    fc_dim = 512
    early_stopping_steps = 5
    grad_cam = False
    seed = 42


if CFG.debug:
    CFG.epochs = 1
    train = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)


ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_score(y_true, y_pred):
    score = f1_score(y_true, y_pred, average="macro")
    return score


def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)

skf = StratifiedKFold(n_splits=CFG.nfolds, shuffle=True, random_state=CFG.seed)
for fold, (trn_idx, vld_idx) in enumerate(skf.split(train, train[CFG.target_col])):
    train.loc[vld_idx, "folds"] = int(fold)
train["folds"] = train["folds"].astype(int)

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['file_name'].values
        self.labels = df[CFG.target_col].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        try:
            file_path = f'/ssd/syjiang/data/FGVC9/train_images/{file_name}'
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image = Image.open(file_path)
            image = image.convert("RGB")
            image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.labels[idx]).float()
        return image, label

def get_transforms(*, data):
    if data == 'train':
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                A.Flip(p=0.05),

                A.Cutout(p=0.05),
                A.HorizontalFlip(p=0.05),
                A.VerticalFlip(p=0.05),
                A.Rotate(limit=180, p=0.05),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.05
                ),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.4,
                    val_shift_limit=0.2, p=0.05
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1),
                    contrast_limit=(-0.1, 0.1), p=0.05
                ),
                ToTensorV2(p=1.0),
            ]
        )

    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

train_dataset = TrainDataset(train, transform=get_transforms(data='train'))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    if CFG.apex:
        scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device).float()
        labels = labels.to(device).long()
        batch_size = labels.size(0)
        if CFG.apex:
            with autocast():
                y_preds = model(images)
                loss = criterion(y_preds, labels)
        else:
            y_preds = model(images)
            loss = criterion(y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if CFG.apex:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            if CFG.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f} '
                  'LR: {lr:.6f}  '
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        # wandb.log({f"[fold{fold}] loss": losses.val,
        #            f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device).float()
        labels = labels.to(device).long()
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        preds.append(y_preds.argmax(1).to('cpu').numpy())
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step + 1) / len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    trn_idx = folds[folds['folds'] != fold].index
    val_idx = folds[folds['folds'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds["category_id"].values

    train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    def get_scheduler(optimizer):
        if CFG.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, **CFG.reduce_params)
        elif CFG.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, **CFG.cosanneal_params)
        elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, **CFG.reduce_params)
        return scheduler

    model = models.resnet101(pretrained=False)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    N_CLASSES = 15501
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()  # 模型
    model.load_state_dict(torch.load('/hpcdata/users/user011/rgye/wang/FGVC9/model_parameter/modelpara_score_finall.pth'))

    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = get_scheduler(optimizer)

    criterion = nn.CrossEntropyLoss()
    best_score = 0.
    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}')

        if score >= best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save(model.state_dict(), '/hpcdata/users/user011/rgye/wang/FGVC9/model_parameter/modelpara_score_best.pth')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save(model.state_dict(), '/hpcdata/users/user011/rgye/wang/FGVC9/model_parameter/modelpara_loss_best.pth')

    # valid_folds["preds_score"] = torch.load(OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_score.pth',
    #                                         map_location=torch.device('cpu'))['preds_score']
    # valid_folds["preds_loss"] = torch.load(OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_loss.pth',
    #                                        map_location=torch.device('cpu'))['preds_loss']
    #
    # return valid_folds

def main():

    def get_result(result_df):
        preds_score = result_df['preds_score'].values
        preds_loss = result_df['preds_loss'].values
        labels = result_df["category_id"].values
        score = get_score(labels, preds_score)
        score_loss = get_score(labels, preds_loss)
        LOGGER.info(f'Score with best score weights: {score:<.4f}')
        LOGGER.info(f'Score with best loss weights: {score_loss:<.4f}')

    if CFG.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(CFG.nfolds):
            if fold in CFG.trn_folds:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
    print('train_successfully')


if __name__ == "__main__":
    main()