import os
import argparse
import glob
import math
import os.path
import platform

from PIL import Image

import numpy as np
import torch.nn.functional as F

import cv2
import torch.cuda

import torchvision
import tqdm
from omegaconf import OmegaConf
import torch
from pytorch_lightning.trainer import Trainer
import time

from torchvision.transforms import InterpolationMode

from utils import instantiate_from_config
from models.unet import UNet



def test():
    versions = ['vgg']
    sep = '\\' if platform.system() == 'Windows' else '/'
    data_path = f'/local_datasets/anime/*/*/images'
    # data_path = 'D:/datasets/BIPED/val/images'
    # data_path = '../BSDS500/images/test'
    # data_path = 'D:/datasets/div2k/test/x2'
    file_names = glob.glob(f'{data_path}/*.*')
    gs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(len(file_names))
    for v in versions:
        model = UNet.load_from_checkpoint(f'./checkpoints/unet/{v}/vanilla/best.ckpt', strict=False).eval().cuda()
        model.loss_fn = None
        with torch.inference_mode():
            for name in tqdm.tqdm(file_names):
                img = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torchvision.transforms.ToTensor()(img).cuda()
                img = torchvision.transforms.Resize(520, interpolation=InterpolationMode.BICUBIC, antialias=True, max_size=1024)(img)
                c, h, w = img.shape
                img = img.unsqueeze(0)
                if w % 8 != 0:
                    w = int(round(w / 8) * 8)
                if h % 8 != 0:
                    h = int(round(h / 8) * 8)

                img = F.interpolate(img, [h, w], mode='bicubic', antialias=True)

                # edges = model(img, torch.tensor([0.0]).cuda())
                # edges = edges.float().detach().cpu()[0]
                # edges = torchvision.transforms.ToPILImage()(edges)
                n = name.rsplit(f'{sep}', 4)
                f = n[-1]
                n = "/".join([n[1], n[2]])
                for i, g in enumerate(gs):
                    edge = model(img, torch.tensor([g]).cuda())
                    edge = edge.float().detach().cpu()[0]
                    edge = torchvision.transforms.ToPILImage()(edge)

                    os.makedirs(f'./edge_test/{v}/{n}/{i}', exist_ok=True)
                    edge.save(f'./edge_test/{v}/{n}/{i}/{f}', 'png')

if __name__ == '__main__':
    test()