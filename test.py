import os
import argparse
import glob
import math
import os.path
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
from models.model import Model
from models.unet import UNet



def test():
    model = UNet.load_from_checkpoint('./checkpoints/unet/hybrid/best.ckpt', strict=False).eval().cuda()

    data_path = 'D:/datasets/anime/train/yae_miko/images'
    # data_path = '../BSDS500/images/test'
    file_names = glob.glob(f'{data_path}/*.*')
    gs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with torch.inference_mode():
        for name in tqdm.tqdm(file_names):
            img = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.ToTensor()(img).cuda()
            img = torchvision.transforms.Resize(840, interpolation=InterpolationMode.BICUBIC, antialias=True, max_size=1600)(img)
            c, h, w = img.shape
            img = img.unsqueeze(0)
            if w % 8 != 0:
                w = int(round(w / 8) * 8)
            if h % 8 != 0:
                h = int(round(h / 8) * 8)

            img = F.interpolate(img, [h, w], mode='bicubic', antialias=True)

            for i, granularity in enumerate(gs):
                edge = model(img, torch.tensor([[granularity]]).cuda())
                edge = edge.float().detach().cpu()
                if len(img.shape) == 4:
                    edge = edge[0]
                edge = torchvision.transforms.ToPILImage()(edge)
                p1, p2 = name.rsplit('images', 1)
                p2, _ = p2.rsplit('.', 1)
                if not os.path.isdir(f'{p1}/edges_test'):
                    os.mkdir(f'{p1}/edges_test')
                edge.save(f'{p1}/edges_test/{p2}_{i}.png', 'png')

if __name__ == '__main__':
    test()