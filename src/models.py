import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

import timm

from . import utils

class G2Net2Model(nn.Module):
    def __init__(self, pretrained_model, CONFIG):
        super(G2Net2Model, self).__init__()

        self.CONFIG = CONFIG

        model = timm.create_model(pretrained_model, pretrained=True, in_chans=2, num_classes=1, drop_rate=CONFIG.dropout)
        # print(model)
        # clsf = model.default_cfg['classifier']
        # n_features = model._modules[clsf].in_features
        # model._modules[clsf] = nn.Identity()

        # self.dropout = nn.Dropout(CONFIG.dropout)

        # self.fc1 = nn.Linear(n_features, n_features)
        # self.act = nn.LeakyReLU()
        # self.fc = nn.Linear(n_features, 1)
        # torch.nn.init.normal_(self.fc.weight)
        self.model = model

    def forward(
        self, x, targets=None, criterion=None
    ):

        with autocast():
            # x, target_a, target_b, lam = utils.mixup(x, targets, alpha=0.1)

            x = self.model(x)
            # x = self.act(self.fc1(x))
            # x = self.fc(self.dropout(x))

            if targets is not None:
                loss = criterion(x.squeeze(1), targets)
                # loss = criterion(x.squeeze(1), target_a) * lam + (1 - lam) * criterion(targets, target_b)

                return x, loss

            return x