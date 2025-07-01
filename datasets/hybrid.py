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


class HybridDataset(Dataset):
    def __init__(self,
                 anime_root,
                 biped_root,
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
            anime_root = os.path.join(anime_root, 'train')
            biped_root = os.path.join(biped_root, 'train')
        else:
            anime_root = os.path.join(anime_root, 'val')
            biped_root = os.path.join(biped_root, 'valid')


        self.img_names = [*glob.glob(f'{anime_root}/*/images/*.*'), *glob.glob(f'{biped_root}/*/images/*.*')]
        self.edge_names = [*glob.glob(f'{anime_root}/*/edges/*.*'), *glob.glob(f'{biped_root}/*/edges/*.*')]

        # self.color_jitter = transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5, hue=0.5)
        # self.invert = transforms.RandomInvert(p=1.0)
        # self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

        self.anime_len = len(self.img_names[0]) - 1

        assert len(self.img_names) == len(self.edge_names)

    def get_img_edge_granularity(self, index: int):
        edge_name = self.edge_names[index]
        img_name = self.img_names[index]

        img = cv2.imread(f'{img_name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, self.color_space)

        img = self.to_tensor(img)

        if random.random() < 0.025:
            c, h, w = img.shape
            edge = torch.full([1, h, w], 1.0)

            granularity = torch.tensor([0.0])
        else:
            edge = cv2.imread(f'{edge_name}', cv2.IMREAD_GRAYSCALE)

            granularity_path = img_name.rsplit('/images', 1)[0]
            granularity = torch.tensor(np.load(f'{granularity_path}/granularity.npy'))

            edge = self.to_tensor(edge)

        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)

        img = tf.resized_crop(img, i, j, h, w, size=self.size, antialias=True)
        edge = tf.resized_crop(edge, i, j, h, w, size=self.size, antialias=True)

        return img, edge, granularity

    def __getitem__(self, index):
        return self.get_img_edge_granularity(index)

    def __len__(self):
        return len(self.img_names)
