import glob
import os
import json
import platform
import random

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
from typing import Union, List, Tuple

from utils import instantiate_from_config, to_2tuple
from datasets.util import nearest_multiple

IMG_FORMATS = ['png', 'jpg']
STR_FORMATS = ['txt', 'csv']


class AnimeLevelDataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 size: int | List[int] | Tuple[int] = 224,
                 scale: List[float] | Tuple[float] = (0.08, 1.0),
                 ratio: List[float] | Tuple[float] = (1.0, 1.0),
                 level: int = 0,
                 ):
        super().__init__()

        self.to_tensor = transforms.ToTensor()

        self.size = list(to_2tuple(size))
        self.scale = list(to_2tuple(scale))
        self.ratio = list(to_2tuple(ratio))

        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')

        self.img_names = glob.glob(f'{root}/*/images/*.*')
        self.level = level

        self.color_jitter = transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5, hue=0.5)
        self.invert = transforms.RandomInvert(p=1.0)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.sep = '\\' if platform.system() == 'Windows' else '/'

    def __getitem__(self, index):
        img_name = self.img_names[index]
        path, name = img_name.split(f'{self.sep}images{self.sep}', 1)

        img = cv2.imread(f'{img_name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        edge = cv2.imread(f'{path}/edges_{self.level}/{name}', cv2.IMREAD_GRAYSCALE)

        img = self.to_tensor(img)
        edge = self.to_tensor(edge)

        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)

        img = tf.resized_crop(img, i, j, h, w, size=self.size, antialias=True)
        edge = tf.resized_crop(edge, i, j, h, w, size=self.size, antialias=True)

        if random.random() < 0.5:
            if random.random() < 0.5:
                img = self.color_jitter(img)
            else:
                img = self.invert(img)

        if random.random() < 0.5:
            img = self.horizontal_flip(img)
            edge = self.horizontal_flip(edge)

        return img, edge

    def __len__(self):
        return len(self.img_names)
