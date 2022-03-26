import os
import numpy as np
import pandas as pd
import json
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import preprocessing

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

# for i in range(5):
#     image = cv.imread('/ssd/syjiang/data/FGVC9/train_images/{a}'.format(a = train_df['file_name'][i]))
#     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     target = train_df['category_id'][i]
#     plt.imshow(image)
#     plt.title(f"target: {target}")
#     plt.show()

train_df = get_train_json()
train_labelencoder(train_df)
