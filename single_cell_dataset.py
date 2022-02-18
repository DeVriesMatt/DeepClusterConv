import torch
from torch.utils.data import Dataset
from skimage import io
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os


def pad_img(img, new_size):
    new_z, new_y, new_x = new_size, new_size, new_size
    z = img.shape[0]
    y = img.shape[1]
    x = img.shape[2]
    delta_z = new_z - z
    delta_y = new_y - y
    delta_x = new_x - x

    if delta_z % 2 == 1:
        z_padding = (delta_z // 2, delta_z // 2 + 1)
    else:
        z_padding = (delta_z // 2, delta_z // 2)

    if delta_y % 2 == 1:
        y_padding = (delta_y // 2, delta_y // 2 + 1)
    else:
        y_padding = (delta_y // 2, delta_y // 2)

    if delta_x % 2 == 1:
        x_padding = (delta_x // 2, delta_x // 2 + 1)
    else:
        x_padding = (delta_x // 2, delta_x // 2)

    padded_data = np.pad(img, (z_padding, y_padding, x_padding), 'constant')
    return padded_data


class SingleCellDataset(Dataset):
    def __init__(self, annotations_file,
                 img_dir,
                 img_size=128,
                 label_col='Treatment',
                 transform=None,
                 target_transform=None):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform

        self.new_df = self.annot_df[(self.annot_df.xDim <= self.img_size) &
                                    (self.annot_df.yDim <= self.img_size) &
                                    (self.annot_df.zDim <= self.img_size)].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df['label_col_enc'] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        img_path = os.path.join(self.img_dir, self.new_df.loc[idx, 'serialNumber'])
        image = io.imread(img_path + '.tif').astype(np.float16)
        image = pad_img(image, self.img_size)

        # return encoded label as tensor
        label = self.new_df.loc[idx, 'label_col_enc']
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        if self.transform:
            image = self.transform(image)
            image = image.unsqueeze(0)

        serial_number = self.new_df.loc[idx, 'serialNumber']

        return image, label, feats


class SingleCellDatasetAll(Dataset):
    def __init__(self, annotations_file,
                 img_dir,
                 img_size=128,
                 label_col='Treatment',
                 transform=None,
                 target_transform=None,
                 cell_component='cell'):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component

        self.new_df = self.annot_df[(self.annot_df.xDim <= self.img_size) &
                                    (self.annot_df.yDim <= self.img_size) &
                                    (self.annot_df.zDim <= self.img_size)].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df['label_col_enc'] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, 'Treatment']
        plate_num = 'Plate' + str(self.new_df.loc[idx, 'PlateNumber'])
        if self.cell_component == 'cell':
            component_path = 'stacked'
        else:
            component_path = 'stacked_nucleus'

        img_path = os.path.join(self.img_dir,
                                plate_num,
                                component_path,
                                'Cells',
                                self.new_df.loc[idx, 'serialNumber'])

        image = io.imread(img_path + '.tif').astype(np.float16)
        image = pad_img(image, self.img_size)

        # return encoded label as tensor
        label = self.new_df.loc[idx, 'label_col_enc']
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        if self.transform:
            image = self.transform(image)
            image = image.unsqueeze(0)

        serial_number = self.new_df.loc[idx, 'serialNumber']

        return image, label, feats
