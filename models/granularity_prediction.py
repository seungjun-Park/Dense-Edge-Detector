import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple, Any, Optional

from models.model import Model
from torchvision import models
from modules.norm.layer_norm import LayerNorm



class GranularityPredictor(Model):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        model = models.vgg16(pretrained=True)

        # 加载features部分
        self.features = model.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False

        self.embed = self.features.pop(0)
        self.fuse_embed = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )

        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )

        self.save_hyperparameters(ignore='loss_config')

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        imgs_features = self.embed(imgs)
        edges_features = self.embed(edges)
        fused_features = self.fuse_embed(torch.cat([imgs_features, edges_features], dim=1))
        logits = self.features(fused_features)
        logits = self.avgpool(logits)
        logits = torch.flatten(logits, 1)
        logits = self.classifier(logits)
        return F.sigmoid(logits)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        images, edges, granularity = batch
        edges = edges.repeat(1, 3, 1, 1)
        outputs = self(images, edges)

        loss, loss_log = self.loss(outputs, granularity, split='train' if self.training else 'valid')

        self.log_dict(loss_log, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Any:
        params = list(self.fuse_embed.parameters()) + list(self.classifier.parameters())
        opt_net = torch.optim.AdamW(params,
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    betas=(0.5, 0.9),
                                    )

        opts = [opt_net]

        return opts



if __name__ == '__main__':
    import glob
    import tqdm
    import cv2
    import torchvision
    from torchvision.transforms import InterpolationMode

    model = GranularityPredictor.load_from_checkpoint('../checkpoints/granularity_prediction/hybrid/last.ckpt').eval().cuda()

    data_path = 'D:/datasets/anime/train/amiya/images'
    edge_path = 'D:/datasets/anime/train/amiya/edges'

    # data_path = '../BSDS500/images/test'
    # data_path = '/local_datasets/yae_miko_genshin/test/images'
    file_names = glob.glob(f'{data_path}/*.*')
    edge_names = glob.glob(f'{edge_path}/*.*')

    features = [model.features[0: 4], model.features[4: 9], model.features[9: 16], model.features[16: 23], model.features[23: 30]]

    with torch.inference_mode():
        for name, edge_name in (zip(file_names, edge_names)):
            path, file = name.rsplit('images', 1)
            file = file.rsplit('.')[0]

            img = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.ToTensor()(img).cuda()
            img = torchvision.transforms.Resize(840, interpolation=InterpolationMode.BICUBIC, antialias=True,
                                                max_size=1600)(img)
            c, h, w = img.shape
            img = img.unsqueeze(0)
            if w % 32 != 0:
                w = int(round(w / 32) * 32)
            if h % 32 != 0:
                h = int(round(h / 32) * 32)

            img = F.interpolate(img, [h, w], mode='bicubic', antialias=True)

            edge = cv2.imread(f'{edge_name}', cv2.IMREAD_GRAYSCALE)
            edge = torchvision.transforms.ToTensor()(edge).cuda()
            edge = torchvision.transforms.Resize(840, interpolation=InterpolationMode.BICUBIC, antialias=True,
                              max_size=1600)(edge)

            c, h, w = edge.shape
            edge = edge.unsqueeze(0)
            if w % 32 != 0:
                w = int(round(w / 32) * 32)
            if h % 32 != 0:
                h = int(round(h / 32) * 32)

            edge = F.interpolate(edge, [h, w], mode='bicubic', antialias=True)
            edge = torch.full([1, 1, h, w], 1.).cuda()

            feats = torch.cat([img, edge], dim=1)
            to_pil = torchvision.transforms.ToPILImage()
            for i, feature in enumerate(features):
                os.makedirs(f'{path}/features/{file}/no_feat{i}', exist_ok=True)
                feats = feature(feats)
                for j, feat in enumerate(feats[0]):
                    f = to_pil(F.sigmoid(feat).cpu())
                    f.save(f'{path}/features/{file}/no_feat{i}/{j}.png','png')

            granularity = model(torch.cat([img, edge], dim=1))

            print(f'file name: {name}, granularity: {granularity}')