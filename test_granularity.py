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
from models.unet import UNet


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
    # model = GranularityPredictor.load_from_checkpoint('./checkpoints/granularity_prediction/hybrid/best.ckpt', strict=False).eval().cuda()
    # model = UNet.load_from_checkpoint('./checkpoints/unet/vanilla/best.ckpt', strict=False).eval().cuda()
    model = Discriminator.load_from_checkpoint('./checkpoints/granularity/discriminator/best.ckpt').eval().cuda()
    data_path = 'D:/datasets/anime/*/*'
    data_path = 'D:/datasets/BIPED/*/v3'
    file_names = zip(glob.glob(f'{data_path}/images/*.*'), glob.glob(f'{data_path}/edges/*.*'), glob.glob(f'{data_path}/granularity/*.*'))
    # file_names = zip(['D:/datasets/anime/train/surtr/images/20.png'], ['D:/datasets/anime/train/surtr/edges/20.png'])
    with torch.inference_mode():
        for names in file_names:
            granularity = torch.from_numpy(np.load(names[2])).cuda()
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

            score = model(img, edge).squeeze(0).cpu().numpy()
            # np.save(f'{names[2]}', score)
            # score_l = np.load(f'{names[2]}')

            print(f'{names[0]}: {granularity}, {score}')

            pil = torchvision.transforms.ToPILImage()

            # f_imgs, f_edges, f_bars = model.get_features(img, edge)
            # for i in range(5):
            #     for n in ['f_imgs', 'f_edges', 'f_bars']:
            #         os.makedirs(f'features/{i}/{n}', exist_ok=True)
            #     for j in range(f_imgs[i][0].shape[0]):
            #
            #         f_img = pil(F.sigmoid(f_imgs[i][0][j]))
            #         f_edge = pil(F.sigmoid(f_edges[i][0][j]))
            #         f_bar = pil(F.sigmoid(f_bars[i][0][j]))
            #
            #         f_img.save(f'features/{i}/f_imgs/{j}.png', 'png')
            #         f_edge.save(f'features/{i}/f_edges/{j}.png', 'png')
            #         f_bar.save(f'features/{i}/f_bars/{j}.png', 'png')

            # feats = model.get_features(img)
            # for i, feat in enumerate(feats):
            #     os.makedirs(f'features_e/{i}/', exist_ok=True)
            #     for j, f in enumerate(feat[0]):
            #         f = pil(F.sigmoid(f))
            #         f.save(f'features_e/{i}/{j}.png', 'png')


def validate():
    parsers = get_parser()

    opt, unknown = parsers.parse_known_args()

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    datamodule = instantiate_from_config(config.data)
    datamodule.prepare_data()
    datamodule.setup()

    model = Discriminator.load_from_checkpoint('./checkpoints/granularity/discriminator/best.ckpt').eval().cuda()

    results = []

    threshold = 0.1

    for pair0, pair1, margin in tqdm.tqdm(datamodule.val_dataloader()):
        with torch.inference_mode():
            score0 = model(pair0[0].cuda(), pair0[1].cuda()).cpu()
            score1 = model(pair1[0].cuda(), pair1[1].cuda()).cpu()

        result = ((score0 - score1) * margin.sign() >= (margin.abs() - threshold)).long()
        results.append(result)

    results = torch.cat(results, dim=0)
    l = results.shape[0]
    s = results.sum()
    print(s / l)


if __name__ == '__main__':
    # validate()
    test()