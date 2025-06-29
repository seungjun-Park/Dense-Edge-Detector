import glob
import os
import json
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


class BIPEDDataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 size: int | List[int] | Tuple[int] = 224,
                 scale: List[float] | Tuple[float] = (0.08, 1.0),
                 ratio: List[float] | Tuple[float] = (1.0, 1.0),
                 color_space: str = 'rgb',
                 ):
        super().__init__()
        color_space = color_space.lower()
        if color_space == 'rgb':
            self.color_space = cv2.COLOR_BGR2RGB
        elif color_space == 'rgba':
            self.color_space = cv2.COLOR_BGR2RGBA
        elif color_space == 'gray':
            self.color_space = cv2.COLOR_BGR2GRAY
        elif color_space == 'xyz':
            self.color_space = cv2.COLOR_BGR2XYZ
        elif color_space == 'ycrcb':
            self.color_space = cv2.COLOR_BGR2YCrCb
        elif color_space == 'hsv':
            self.color_space = cv2.COLOR_BGR2HSV
        elif color_space == 'lab':
            self.color_space = cv2.COLOR_BGR2LAB
        elif color_space == 'luv':
            self.color_space = cv2.COLOR_BGR2LUV
        elif color_space == 'hls':
            self.color_space = cv2.COLOR_BGR2HLS
        elif color_space == 'yuv':
            self.color_space = cv2.COLOR_BGR2YUV

        self.to_tensor = transforms.ToTensor()

        self.size = list(to_2tuple(size))
        self.scale = list(to_2tuple(scale))
        self.ratio = list(to_2tuple(ratio))

        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'valid')

        self.edge_names = glob.glob(f'{root}/*/edges/*.*')
        self.img_names = glob.glob(f'{root}/*/images/*.*')

        self.color_jitter = transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5, hue=0.5)
        self.invert = transforms.RandomInvert(p=1.0)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

        assert len(self.edge_names) == len(self.img_names)

    def __getitem__(self, index):
        edge_name = self.edge_names[index]
        img_name = self.img_names[index]

        img = cv2.imread(f'{img_name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, self.color_space)
        edge = cv2.imread(f'{edge_name}', cv2.IMREAD_GRAYSCALE)

        img = self.to_tensor(img)
        edge = 1. - self.to_tensor(edge)

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
