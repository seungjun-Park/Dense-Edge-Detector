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
import time

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
        model: Model = instantiate_from_config(config.module)

    model = torch.compile(model)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    # trainer.test(model=model)


def test():
    parsers = get_parser()

    opt, unknown = parsers.parse_known_args()

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    device = torch.device('cuda')

    model: Model = instantiate_from_config(config.module).to(device).eval()
    state_dict = torch.load('./checkpoints/unet/convnext/hd/best.ckpt', map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    data_path = 'D:/datasets/edge_detection/yae_miko_genshin/test/images'
    # data_path = '/local_datasets/wakamo/val/images'
    file_names = glob.glob(f'{data_path}/*.*')

    mean_time = []
    mean_h = []
    mean_w = []

    with torch.no_grad():
        for name in tqdm.tqdm(file_names):
            img = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.transforms.ToTensor()(img).to(device)
            # img = torchvision.transforms.transforms.Resize(720)(img)
            c, h, w = img.shape
            if w % 40 != 0:
                w = math.ceil(w / 40) * 40
                # mean_w.append(w)
            if h % 40 != 0:
                h = math.ceil(h / 40) * 40
                # mean_h.append(h)
            img = torchvision.transforms.transforms.Resize([h, w])(img)
            img = img.unsqueeze(0)
            with torch.autocast(dtype=torch.float, device_type='cuda'):
                #start = time.time()
                img = model(img)
                #end = time.time()
                #mean_time.append(end - start)
            img = img.float().detach().cpu()
            if len(img.shape) == 4:
                img = img[0]
            img = torchvision.transforms.ToPILImage()(img)
            p1, p2 = name.rsplit('images', 1)
            if not os.path.isdir(f'{p1}/edges'):
                os.mkdir(f'{p1}/edges')
            img.save(f'{p1}/edges/{p2}.png', 'png')
            # p1, p2 = name.rsplit('imgs', 1)
            # img.save(f'{p1}/edge_maps/{p2}', 'png')

    # print(f'avg time: {sum(mean_time) / len(mean_time)}')
    # print(f'avg h: {sum(mean_h) / len(mean_h)}')
    # print(f'avg w: {sum(mean_w) / len(mean_w)}')

if __name__ == '__main__':
    main()
    # test()
