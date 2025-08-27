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


class EdgeGranularityDataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 size: int | List[int] | Tuple[int] = 224,
                 scale: List[float] | Tuple[float] = (0.08, 1.0),
                 ratio: List[float] | Tuple[float] = (1.0, 1.0),
                 ):
        super().__init__()

        self.to_tensor = transforms.ToTensor()

        self.size = list(to_2tuple(size))
        self.scale = list(to_2tuple(scale))
        self.ratio = list(to_2tuple(ratio))

        self.root = root

        if train:
            with open("train_pair_datasets.json", "r", encoding="utf-8") as f:
                self.pairs = json.load(f)
        else:
            with open("val_pair_datasets.json", "r", encoding="utf-8") as f:
                self.pairs = json.load(f)

        self.color_jitter = transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5, hue=0.5)
        self.invert = transforms.RandomInvert(p=1.0)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

    def get_img_edge_pair(self, img_name: str, edge_name: str) -> List[torch.Tensor]:
        img = cv2.imread(f'{self.root}/{img_name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edge = cv2.imread(f'{self.root}/{edge_name}', cv2.IMREAD_GRAYSCALE)

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

    def __getitem__(self, index):
        pair = self.pairs[index]
        img0_name = pair[0][0]
        edge0_name = img0_name.replace('images', 'edges')
        img1_name = pair[0][1]
        edge1_name = img1_name.replace('images', 'edges')
        label = torch.tensor([float(pair[1])])

        img0, edge0 = self.get_img_edge_pair(img0_name, edge0_name)
        img1, edge1 = self.get_img_edge_pair(img1_name, edge1_name)

        return (img0, edge0), (img1, edge1), label

    def __len__(self):
        return len(self.pairs)