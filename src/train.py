import ast
import numpy as np
import gc

import pandas as pd
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import transformers

from . import engine
from . import datasets
from . import models
from . import utils

import importlib
import argparse
import sys
from copy import copy

from sklearn.metrics import roc_auc_score

import optuna

sys.path.append("configs")

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)

CONFIG = copy(importlib.import_module(parser_args.config).cfg)


def train_loop(folds, fold, save_dir=None):
    utils.set_seeds(CONFIG.seed)

    print(f"training fold {fold}")

    writer = SummaryWriter()

    # todo move to config
    train_transform = A.Compose(
        [   
            # A.ChannelShuffle(p=0.5),
            # A.CoarseDropout(max_holes=2, max_height=8, max_width=8, p=0.25),
            # A.CoarseDropout(max_holes=2, max_height=256, max_width=16, min_height=256, p=0.25),
            # A.CoarseDropout(max_holes=2, max_height=16, max_width=360, min_width=360, p=0.25),
            A.OneOf(
                [
                    # A.CoarseDropout(max_holes=2, max_height=32, max_width=32, p=0.25),
                    A.CoarseDropout(max_holes=1, max_height=256, max_width=16, min_height=256, p=1),
                    A.CoarseDropout(max_holes=1, max_height=16, max_width=360, min_width=360, p=1)
                ],
                p=CONFIG.p1
            ),
            A.HorizontalFlip(p=CONFIG.p2),
            A.VerticalFlip(p=CONFIG.p2),
            # A.Resize(360, 360, p=1),
            # A.GaussNoise(var_limit p=CONFIG.p4),
            # A.SmallestMaxSize(max_size=160),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=CONFIG.p3),
            # A.RandomCrop(height=256, width=256),
            # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            # A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
            # ToTensorV2(),
        ]
    )

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
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
            # ToTensorV2(),
        ]
    )

    # ====================================================
    # loader
    # ====================================================

    train_folds = folds[folds["kfold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["kfold"] == fold].reset_index(drop=True)

    print("train, valid", len(train_folds), len(valid_folds))

    train_dataset = datasets.G2Net2Dataset(
        train_folds, CONFIG, train_transform
    )

    valid_dataset = datasets.G2Net2Dataset(
        valid_folds, CONFIG, valid_transform
    )




    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.n_workers,
        pin_memory=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG.valid_batch_size,
        shuffle=False,
        num_workers=CONFIG.n_workers,
        pin_memory=True,
        drop_last=False
    )


    model = models.G2Net2Model(CONFIG.pretrained_model, CONFIG)
    model.to("cuda")

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": CONFIG.weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


    if CONFIG.optim == "Adam":
        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=CONFIG.lr,
            weight_decay=CONFIG.weight_decay,
        )
    else:
        optimizer = torch.optim.RMSprop(
            optimizer_parameters,
            lr=CONFIG.lr,
            weight_decay=CONFIG.weight_decay
        )



    if CONFIG.scheduler == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=CONFIG.warmup,
            num_training_steps=int(
                len(train_loader) * (CONFIG.epochs) / CONFIG.n_accumulate
            ),
        )
    elif CONFIG.scheduler == "ptorch":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=CONFIG.lr, total_steps=len(train_loader) * CONFIG.epochs, pct_start=CONFIG.ONE_CYCLE_PCT_START)
    else:
        scheduler = None

    criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler()

    best_score = 0.0


    for epoch in range(CONFIG.epochs):

        ave_train_loss = engine.train_fn(
            epoch,
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            CONFIG,
            scaler
        )

        writer.add_scalar("train/loss", ave_train_loss, epoch)

        valid_loss, predictions, = engine.valid_fn(
            valid_loader, model, criterion, CONFIG
        )

        try:
            valid_score = roc_auc_score(valid_folds.target, predictions)
        except:
            print(f"pruning trial - nans")
            raise optuna.TrialPruned()


        # if valid_score < 0.6:
        #     print(f"pruning trail {valid_score}")
        #     raise optuna.TrialPruned()

        print(f"Valid Score: {valid_score}")
        writer.add_scalar(
                "valid/loss",
                valid_loss,
                len(train_loader) * (epoch + 1),
            )
        writer.add_scalar(
                "valid/score",
                valid_score,
                len(train_loader) * (epoch + 1),
            )


        score = valid_score

        print(f"Score: {score}")
        writer.add_scalar(
                "combined/score",
                score,
                len(train_loader) * (epoch + 1),
            )
            
        if score > best_score:
            print(
                f"results for epoch {epoch + 1}: new best score {best_score} ====> {score}"
            )
            best_score = score

            if save_dir:
                torch.save(
                    model.state_dict(), f"{save_dir}/{CONFIG.model_name}_{fold}.pth"
                )


        else:
            print(
                f"results for epoch {epoch + 1}: score {score}, best {best_score}"
            )

    # if save_dir:
    #     torch.save(
    #         model.state_dict(), f"{save_dir}/{CONFIG.model_name}_{fold}.pth"
    #     )

    # if best_score < 0.7:
    #     print("pruning trail")
    #     raise optuna.TrialPruned()

    del model
    gc.collect()
    return best_score


def objective(trial):
    # CONFIG.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    CONFIG.lr = trial.suggest_float("lr", 5e-5, 1e-3, step=1e-5)
    CONFIG.dropout = trial.suggest_float("dropout", 0, 0.5, step=0.05)
    CONFIG.max_grad = trial.suggest_int("max_grad", 1, 1000)
    CONFIG.warmup = trial.suggest_int("warmup", 0, 150, step=25)

    CONFIG.p1 = trial.suggest_float("p1", 0, 0.5, step=0.1)
    CONFIG.p2 = trial.suggest_float("p2", 0, 0.5, step=0.1)
    CONFIG.p3 = trial.suggest_float("p3", 0, 0.5, step=0.1)

    CONFIG.optim = trial.suggest_categorical("optim", ["Adam", "RMSprop"])


    CONFIG.gaussian = trial.suggest_float("gaussian", 0, 10, step=0.1)

    train = pd.read_csv(f"{CONFIG.data_loc}/train_folds.csv")
    train = train[train['target'] >= 0]
    # train = train.sample(10000)

    CV = []

    if CONFIG.save_directory:
        dt = CONFIG.save_directory
        save_dir = CONFIG.save_root + f"/{dt}"

        Path(save_dir).mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None  # ie don't save

    for i, fold in enumerate(CONFIG.folds):
        print(CONFIG)
        score = train_loop(train, fold, save_dir)

        CV.append(score)

    return np.mean(CV)


if __name__ == "__main__":
    train = pd.read_csv(f"{CONFIG.data_loc}/train_folds.csv")
    train = train[train['target'] >= 0]

    
    print("note removed -1 target!!")

    pruned = pd.read_csv("pruned.csv")
    pruned = pruned[pruned['preds'] > 0.5]

    print(len(pruned), len(train))

    # train = train[train['id'].isin(pruned.id.values)]

    print(len(train))

    print("training g2net2")

    utils.set_seeds(CONFIG.seed)

    print(train.head(3))


    if CONFIG.tune_training:
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=25)

        print("best trail")
        print(study.best_value)
        print(study.best_params)
        print(study.trials_dataframe().sort_values('value'))
    
    else:
        CV = []

        if CONFIG.save_directory:
            dt = CONFIG.save_directory
            save_dir = CONFIG.save_root + f"/{dt}"

            Path(save_dir).mkdir(parents=True, exist_ok=True)
        else:
            save_dir = None  # ie don't save

        for i, fold in enumerate(CONFIG.folds):
            print(CONFIG)
            score = train_loop(train, fold, save_dir)

            CV.append(score)

        print(f"final CVs {CV}")
        print(f"mean CV {np.mean(CV)}, std {np.std(CV)}")
