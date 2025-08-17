import os
import argparse
import glob
import math
import os.path

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
from models.model import Model
from models.granularity_prediction import GranularityPredictor



def test():
    model = GranularityPredictor.load_from_checkpoint('./checkpoints/granularity_prediction/hybrid/best.ckpt', strict=False).eval().cuda()

    data_path = 'D:/datasets/anime/train/yae_miko'
    # data_path = 'D:/datasets/BIPED/*/*'
    file_names = zip(glob.glob(f'{data_path}/images/*.*'), glob.glob(f'{data_path}/edges/*.*'), glob.glob(f'{data_path}/granularity/*.*'))
    file_names = zip(['D:/datasets/anime/train/surtr/images/20.png'], ['D:/datasets/anime/train/surtr/edges/20.png'])
    with torch.inference_mode():
        for names in file_names:
            #granularity = torch.from_numpy(np.load(names[2])).cuda()
            img = cv2.imread(f'{names[0]}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.ToTensor()(img).cuda()

            edge = cv2.imread(f'{names[1]}', cv2.IMREAD_GRAYSCALE)
            edge = torchvision.transforms.ToTensor()(edge).cuda()

            img = torchvision.transforms.Resize(800, interpolation=InterpolationMode.BICUBIC, antialias=True, max_size=1200)(img)
            edge = torchvision.transforms.Resize(800, interpolation=InterpolationMode.BICUBIC, antialias=True, max_size=1200)(edge)
            c, h, w = img.shape
            img = img.unsqueeze(0)
            edge = edge.unsqueeze(0)
            if w % 8 != 0:
                w = int(round(w / 8) * 8)
            if h % 8 != 0:
                h = int(round(h / 8) * 8)

            img = F.interpolate(img, [h, w], mode='bicubic', antialias=True)
            edge = F.interpolate(edge, [h, w], mode='bicubic', antialias=True)
            # edge = torch.ones(1, 1, *img.shape[2:]).cuda()

            g, d_align, d_raw, d_shift = model(img, edge, True)

            print(f'g: {g}, d_align: {d_align}, d_raw: {d_raw}, d_shift: {d_shift}')

            f_imgs, f_edges, f_bars = model.get_features(img, edge)
            for i in range(5):
                for n in ['f_imgs', 'f_edges', 'f_bars']:
                    os.makedirs(f'features/{i}/{n}', exist_ok=True)
                for j in range(f_imgs[i][0].shape[0]):
                    pil = torchvision.transforms.ToPILImage()

                    f_img = pil(F.sigmoid(f_imgs[i][0][j]))
                    f_edge = pil(F.sigmoid(f_edges[i][0][j]))
                    f_bar = pil(F.sigmoid(f_bars[i][0][j]))

                    f_img.save(f'features/{i}/f_imgs/{j}.png', 'png')
                    f_edge.save(f'features/{i}/f_edges/{j}.png', 'png')
                    f_bar.save(f'features/{i}/f_bars/{j}.png', 'png')


if __name__ == '__main__':
    test()