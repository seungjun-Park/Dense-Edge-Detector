import os
import argparse
import glob
import math
import os.path

import cv2
import torch.cuda

import torchvision
import tqdm
from omegaconf import OmegaConf
import torch
from pytorch_lightning.trainer import Trainer

from utils import instantiate_from_config
from models.model import Model


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

    # datamodule
    datamodule = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though

    model: Model = instantiate_from_config(config.module)

    logger = instantiate_from_config(config.logger)

    callbacks = [instantiate_from_config(config.checkpoints[cfg]) for cfg in config.checkpoints]

    trainer_configs = config.trainer
    ckpt_path = None
    if 'ckpt_path' in trainer_configs.keys():
        ckpt_path = trainer_configs.pop('ckpt_path')

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        detect_anomaly=False,
        **trainer_configs
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    # trainer.test(model=model)


def test():
    parsers = get_parser()

    opt, unknown = parsers.parse_known_args()

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # datamodule
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    device = torch.device('cuda')
    model = instantiate_from_config(config.module).eval().to(device)

    # data_path = './datasets/arknights_v2/train/surtr/images'
    data_path = '/local_datasets/wakamo/val/images'
    file_names = glob.glob(f'{data_path}/*.*')
    with torch.no_grad():
        for name in tqdm.tqdm(file_names):
            img = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.transforms.ToTensor()(img).to(device)
            img = torchvision.transforms.transforms.Resize(768)(img)
            c, h, w = img.shape
            if w % 8 != 0:
                w = math.ceil(w / 8) * 8
            if h % 8 != 0:
                h = math.ceil(h / 8) * 8
            img = torchvision.transforms.transforms.Resize([h, w])(img)
            img = img.unsqueeze(0)
            img = model(img)
            img = img.detach().cpu()
            if len(img.shape) == 4:
                img = img[0]
            img = torchvision.transforms.ToPILImage()(img)
            p1, p2 = name.rsplit('images', 1)
            if not os.path.isdir(f'{p1}/edges_v2'):
                os.mkdir(f'{p1}/edges_v2')
            img.save(f'{p1}/edges_v2/{p2}.png', 'png')
            # p1, p2 = name.rsplit('imgs', 1)
            # img.save(f'{p1}/edge_maps/{p2}', 'png')


if __name__ == '__main__':
    main()
    # test()
