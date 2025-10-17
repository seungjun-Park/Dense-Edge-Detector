import json
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
from kornia.models.detection import results_from_detections
from omegaconf import OmegaConf
import torch
from pytorch_lightning.trainer import Trainer
import time

from torchvision.transforms import InterpolationMode

from utils import instantiate_from_config
from models.model import Model
from models.discriminator import Discriminator
from models.lpieps import LPIEPS
from models.unet import UNet
from taming.modules.losses import LPIPS


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help='path to base configs. Loaded from left-to-right. '
             'Parameters can be oeverwritten or added with command-line options of the form "--key value".',
        default=list(),
    )

    return parser


def test():
    model = LPIEPS.load_from_checkpoint('./checkpoints/lpieps/vgg/best.ckpt', strict=False).eval().cuda()
    model.loss = None
    # print(model.lins[0].model[1].weight)
    # for lin in model.lins:
    #     print(torch.where(lin.model[1].weight < 0.))
    data_path = 'D:/datasets/anime/train/*'
    # data_path = 'D:/datasets/BIPED/*'
    file_names = zip(glob.glob(f'{data_path}/images/*.*'), glob.glob(f'{data_path}/edges_0/*.*'), glob.glob(f'{data_path}/edges_1/*.*'), glob.glob(f'{data_path}/edges_2/*.*'))
    avg_scores = []
    with torch.inference_mode():
        for names in file_names:
            img = cv2.imread(f'{names[0]}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.ToTensor()(img).cuda()

            img = torchvision.transforms.Resize(800, interpolation=InterpolationMode.BICUBIC, antialias=True,
                                                max_size=1200)(img)
            c, h, w = img.shape
            img = img.unsqueeze(0)
            if w % 8 != 0:
                w = int(round(w / 8) * 8)
            if h % 8 != 0:
                h = int(round(h / 8) * 8)
            img = F.interpolate(img, [h, w], mode='bicubic', antialias=True)

            scores = []
            for i in range(1, 4):
                edge = cv2.imread(f'{names[i]}', cv2.IMREAD_GRAYSCALE)
                edge = torchvision.transforms.ToTensor()(edge).cuda()

                edge = torchvision.transforms.Resize(720, interpolation=InterpolationMode.BICUBIC, antialias=True,
                                                     max_size=800)(edge)
                edge = edge.unsqueeze(0)
                edge = F.interpolate(edge, [h, w], mode='bicubic', antialias=True)
                # edge = torch.ones(1, 1, *img.shape[2:]).cuda()

                score = model(img, edge).reshape(1).cpu()[0]
                scores.append(score)
            avg_scores.append(torch.tensor(scores))
            print(f'{names[0]}: {scores}')

            # pil = torchvision.transforms.ToPILImage()

    avg_scores = torch.stack(avg_scores, dim=1)
    scores_1, scores_2, scores_3 = torch.chunk(avg_scores, 3, dim=0)
    scores_1 = scores_1.mean()
    scores_2 = scores_2.mean()
    scores_3 = scores_3.mean()

    print(scores_1, scores_2, scores_3)

def visualization():
    model = LPIEPS.load_from_checkpoint('./checkpoints/lpieps/convnext/best.ckpt', strict=False).eval().cuda()
    model.loss = None
    data_path = 'D:/datasets/anime/*/*'
    # data_path = 'D:/datasets/BIPED/*'
    file_names = zip(glob.glob(f'{data_path}/images/*.*'), glob.glob(f'{data_path}/edges_0/*.*'),
                     glob.glob(f'{data_path}/edges_1/*.*'), glob.glob(f'{data_path}/edges_2/*.*'))

    with torch.inference_mode():
        pil = torchvision.transforms.ToPILImage()
        for names in file_names:
            img = cv2.imread(f'{names[0]}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.ToTensor()(img).cuda()

            img = torchvision.transforms.Resize(800, interpolation=InterpolationMode.BICUBIC, antialias=True,
                                                max_size=1200)(img)
            c, h, w = img.shape
            img = img.unsqueeze(0)
            if w % 8 != 0:
                w = int(round(w / 8) * 8)
            if h % 8 != 0:
                h = int(round(h / 8) * 8)
            img = F.interpolate(img, [h, w], mode='bicubic', antialias=True)

            scores = []
            for i in range(1, 4):
                edge = cv2.imread(f'{names[i]}', cv2.IMREAD_GRAYSCALE)
                edge = torchvision.transforms.ToTensor()(edge).cuda()

                edge = torchvision.transforms.Resize(800, interpolation=InterpolationMode.BICUBIC, antialias=True,
                                                     max_size=1200)(edge)
                edge = edge.unsqueeze(0)
                edge = F.interpolate(edge, [h, w], mode='bicubic', antialias=True)
                # edge = torch.ones(1, 1, *img.shape[2:]).cuda()

                feats_imgs, moderators, feats_edges = model._get_features(img, edge)

                for level, (feat_imgs, moderator, feat_edges) in enumerate(zip(feats_imgs, moderators, feats_edges)):
                    os.makedirs(f'./feats/imgs/{level}', exist_ok=True)
                    os.makedirs(f'./feats/moderators/{level}', exist_ok=True)
                    os.makedirs(f'./feats/edges/{level}', exist_ok=True)
                    feat_imgs = F.sigmoid(feat_imgs.squeeze(0)).cpu()
                    moderator = F.sigmoid(moderator.squeeze(0)).cpu()
                    feat_edges = F.sigmoid(feat_edges.squeeze(0)).cpu()
                    for idx, (im, m, e) in enumerate(zip(feat_imgs, moderator, feat_edges)):
                        pil(im).save(f'./feats/imgs/{level}/{idx}.png')
                        pil(m).save(f'./feats/moderators/{level}/{idx}.png')
                        pil(e).save(f'./feats/edges/{level}/{idx}.png')

            print(f'{names[0]}: {scores}')


def save_score():
    model1 = LPIEPS.load_from_checkpoint('./checkpoints/lpieps/convnext/best.ckpt', strict=False).eval().cuda()
    model2 = LPIEPS.load_from_checkpoint('./checkpoints/lpieps/vgg/best.ckpt', strict=False).eval().cuda()
    data_path = 'D:/datasets/anime/*/*'
    # data_path = 'D:/datasets/BIPED/*'
    data_path = glob.glob(data_path)
    with torch.inference_mode():
        for path in data_path:
            file_names = zip(glob.glob(f'{path}/images/*.*'), glob.glob(f'{path}/edges_0/*.*'),
                             glob.glob(f'{path}/edges_1/*.*'), glob.glob(f'{path}/edges_2/*.*'))
            for i, names in enumerate(file_names):
                img = cv2.imread(f'{names[0]}', cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torchvision.transforms.ToTensor()(img).cuda()

                img = torchvision.transforms.Resize(800, interpolation=InterpolationMode.BICUBIC, antialias=True,
                                                    max_size=1200)(img)
                c, h, w = img.shape
                img = img.unsqueeze(0)
                if w % 8 != 0:
                    w = int(round(w / 8) * 8)
                if h % 8 != 0:
                    h = int(round(h / 8) * 8)
                img = F.interpolate(img, [h, w], mode='bicubic', antialias=True)

                edge = cv2.imread(f'{names[2]}', cv2.IMREAD_GRAYSCALE)
                edge = torchvision.transforms.ToTensor()(edge).cuda()

                edge = torchvision.transforms.Resize(800, interpolation=InterpolationMode.BICUBIC, antialias=True,
                                                     max_size=1200)(edge)
                edge = edge.unsqueeze(0)
                edge = F.interpolate(edge, [h, w], mode='bicubic', antialias=True)

                score1 = model1(img, edge).reshape(1).cpu()[0].numpy()
                score2 = model2(img, edge).reshape(1).cpu()[0].numpy()
                score = (score1 + score2) / 2

                os.makedirs(f'{path}/lpieps', exist_ok=True)
                np.save(f'{path}/lpieps/{i}', score)

                print(f'{names[2]}: {score}')

if __name__ == '__main__':
    # visualization()
    test()
    # save_score()