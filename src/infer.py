import numpy as np

import pandas as pd

import torch

from torch.utils.data import DataLoader
import albumentations as A

from . import engine
from . import datasets
from . import models
from . import utils

import importlib
import argparse
import sys
from copy import copy

sys.path.append("configs")

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)

CONFIG = copy(importlib.import_module(parser_args.config).cfg)


def infer_fold(sample_subs, fold):
    print(f"infer fold {fold}")

    valid_transform = A.Compose(
        [   
            # A.ChannelShuffle(p=0.5),
            # A.CoarseDropout(max_holes=4, max_height=8, max_width=8, p=0.25),
            # A.CoarseDropout(max_holes=2, max_height=256, max_width=16, min_height=256, p=0.25),
            # A.CoarseDropout(max_holes=2, max_height=16, max_width=360, min_width=360, p=0.25),
            # A.OneOf(
            #     [
            #         # A.CoarseDropout(max_holes=2, max_height=32, max_width=32, p=0.25),
            #         A.CoarseDropout(max_holes=1, max_height=256, max_width=32, min_height=256, p=1),
            #         A.CoarseDropout(max_holes=1, max_height=32, max_width=360, min_width=360, p=1)
            #     ],
            #     p=0.5
            # ),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.Resize(224, 224, p=1),
            # A.GaussNoise(p=0.1),
            # A.SmallestMaxSize(max_size=160),
            # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
            # A.RandomCrop(height=256, width=256),
            # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            # A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.5, 0.5), std=(0.1, 0.1)),
            # ToTensorV2(),
        ]
    )

    # ====================================================
    # loader
    # ====================================================

    print("test size", len(sample_subs))

    test_dataset = datasets.G2Net2Dataset_Test(
        sample_subs, CONFIG, valid_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG.valid_batch_size,
        shuffle=False,
        num_workers=CONFIG.n_workers,
        pin_memory=True,
        drop_last=False
    )

    model = models.G2Net2Model(CONFIG.pretrained_model, CONFIG)
    state = torch.load(f"/mnt/datastore/g2net2/{CONFIG.save_directory}/{CONFIG.model_name}_{fold}.pth", map_location="cpu")
    model.load_state_dict(state)
    model.to("cuda")

    preds = engine.infer_fn(test_loader, model)

    return preds


if __name__ == "__main__":
    sample_sub = pd.read_csv(f"{CONFIG.data_loc}/sample_submission.csv")
    
    print("infer g2net2")

    final_predictions = []

    for i, fold in enumerate(CONFIG.folds):
        print(CONFIG)
        fold_preds = infer_fold(sample_sub, fold)

        final_predictions.append(fold_preds)

    final_predictions = np.mean(final_predictions, axis=0)

    sample_sub['target'] = final_predictions
    sample_sub.to_csv("outputs/submission.csv", index=False)
