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
                 anime_root: str,
                 biped_root: str,
                 train: bool = True,
                 size: int | List[int] | Tuple[int] = 224,
                 scale: List[float] | Tuple[float] = (0.08, 1.0),
                 ratio: List[float] | Tuple[float] = (1.0, 1.0),
                 ):
        super().__init__()

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
        self.granularity = [*glob.glob(f'{anime_root}/*/granularity/*.*'), *glob.glob(f'{biped_root}/*/granularity/*.*')]
        self.color_jitter = transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5, hue=0.5)
        self.invert = transforms.RandomInvert(p=1.0)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

        assert len(self.img_names) == len(self.edge_names)

    def get_img_edge_granularity(self, index: int):
        edge_name = self.edge_names[index]
        img_name = self.img_names[index]
        granularity_name = self.granularity[index]

        img = cv2.imread(f'{img_name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edge = cv2.imread(f'{edge_name}', cv2.IMREAD_GRAYSCALE)

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

        granularity = torch.from_numpy(np.load(granularity_name))

        return img, edge, granularity

    def __getitem__(self, index):
        return self.get_img_edge_granularity(index)

    def __len__(self):
        return len(self.img_names)
