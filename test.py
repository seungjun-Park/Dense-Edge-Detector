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

from utils import instantiate_from_config
from models.model import Model
from models.unet import UNet



def test():
    model = UNet.load_from_checkpoint('./checkpoints/unet/vanilla/last.ckpt').eval().cuda()

    # data_path = 'D:/datasets/edge_detection/yae_miko_genshin/test/images'
    data_path = '/local_datasets/yae_miko_genshin/test/images'
    file_names = glob.glob(f'{data_path}/*.*')

    with torch.inference_mode():
        for name in tqdm.tqdm(file_names):
            img = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.transforms.ToTensor()(img).cuda()
            img = img.unsqueeze(0)
            b, c, h, w = img.shape
            if w % 8 != 0:
                w = int(round(w / 8) * 8)
            if h % 8 != 0:
                h = int(round(h / 8) * 8)

            img = F.interpolate(img, [h, w], mode='bicubic', antialias=True)

            img = model(img)
            img = img.float().detach().cpu()
            if len(img.shape) == 4:
                img = img[0]
            img = torchvision.transforms.ToPILImage()(img)
            p1, p2 = name.rsplit('images', 1)
            if not os.path.isdir(f'{p1}/edges'):
                os.mkdir(f'{p1}/edges')
            img.save(f'{p1}/edges/{p2}.png', 'png')

if __name__ == '__main__':
    test()