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
from models import DefaultModel
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

    parser.add_argument(
        '--epoch',
        nargs='?',
        type=int,
        default=100,
    )

    return parser


def main():
    parsers = get_parser()

    opt, unknown = parsers.parse_known_args()

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    trainer_configs = config.trainer
    ckpt_path = None
    if 'ckpt_path' in trainer_configs.keys():
        ckpt_path = trainer_configs.pop('ckpt_path')

    logger = instantiate_from_config(config.logger)

    callbacks = [instantiate_from_config(config.checkpoints[cfg]) for cfg in config.checkpoints]

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        detect_anomaly=False,
        **trainer_configs
    )

    # datamodule
    datamodule = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though

    with trainer.init_module():
        model: DefaultModel = instantiate_from_config(config.module)

    # model = torch.compile(model)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    # trainer.test(model=model)

if __name__ == '__main__':
    main()
