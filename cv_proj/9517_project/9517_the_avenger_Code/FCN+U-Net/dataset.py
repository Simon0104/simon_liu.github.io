import os
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from get_mask import get_mask_with_id
import h5py


IMAGE_NET_MEAN = [0., 0., 0.]
IMAGE_NET_STD = [1., 1., 1.]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

def find_id_by_filename(csv_file, file_name):
    df = pd.read_csv(csv_file)

    matching_row = df[df['file_name'] == file_name]

    if not matching_row.empty:
        return matching_row['id'].values[0]
    else:
        return None

class T_Dataset(Dataset):
    def __init__(self, path_to_csv, images_path,if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path =  images_path

        if if_train:
            self.transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            normalize])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            normalize])

        self.transform_mask = transforms.Compose([
            transforms.Resize((480, 480))
            ]
        )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]

        image_id = row['file_name']

        image_path = os.path.join(self.images_path, f'{image_id}')

        #mask_id = find_id_by_filename('filename-id.csv',image_id )
        #mask = get_mask_with_id(mask_id)

        x = self.transform(default_loader(image_path))

        with h5py.File(os.path.join(self.images_path, "annotations", '%s.h5'%(image_id.replace("/", "_")[:-4]) ), 'r') as hf:
            mask = hf['arr'][:]

        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask, dim=0)
        mask = self.transform_mask(mask)
        mask = torch.squeeze(mask)

        return x, mask
