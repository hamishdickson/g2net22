import importlib
import argparse
import sys
from copy import copy
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import optuna

from sklearn.model_selection import StratifiedKFold


import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

import transformers
from torch.utils.data import DataLoader

from . import datasets
from . import models
from . import engine
from . import utils

from sklearn.metrics import roc_auc_score, log_loss

sys.path.append("configs")

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)

CONFIG = copy(importlib.import_module(parser_args.config).cfg)

def train_discriminator(df_real, df_fake, eval_mode="all"):
    df_real = df_real[df_real['target'] >= 0]
    df_real['dtarget'] = 1
    df_fake['dtarget'] = 0

    scores = []


    df = pd.concat([df_real, df_fake])
    df = df.reset_index(drop=True)

    print(len(df), len(df_real), len(df_fake))

    df_out = pd.DataFrame()

    for seed in [42]: #, 123, 666]:
        utils.set_seeds(seed)

        df['kfold'] = -1

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for fold, (_, val_) in enumerate(cv.split(df, df.dtarget.values)):
            df.loc[val_,"kfold"] = fold

        for fold in range(5):
            df_train = df[df["kfold"] != fold]
            df_valid = df[df["kfold"] == fold]

            # df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42)

            if eval_mode == "noise":
                df_valid = df_valid[((df_valid['target'] == 0))]
            elif eval_mode == "cw":
                df_valid = df_valid[((df_valid['target'] == 1))]

            df_train = df_train.reset_index(drop=True)
            df_valid = df_valid.reset_index(drop=True)

            # if eval_mode == "noise":
            #     df_valid = df_valid[(df_valid['dtarget'] == 1) | (( df_valid['dtarget'] == 0) & (df_valid['target'] == 0))]
            # elif eval_mode == "cw":
            #     df_valid = df_valid[(df_valid['dtarget'] == 1) | (( df_valid['dtarget'] == 0) & (df_valid['target'] == 1))]

            train_dataset = datasets.DiscriminatorDataset(
                df_train, CONFIG
            )

            valid_dataset = datasets.DiscriminatorDataset(
                df_valid, CONFIG
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


            optimizer = torch.optim.AdamW(
                optimizer_parameters,
                lr=CONFIG.lr,
                weight_decay=CONFIG.weight_decay,
            )


            if CONFIG.scheduler == "cosine":
                scheduler = transformers.get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=CONFIG.warmup,
                    num_training_steps=int(
                        len(train_loader) * (CONFIG.epochs) / CONFIG.n_accumulate
                    ),
                )
            else:
                scheduler = None

            criterion = nn.BCEWithLogitsLoss()

            scaler = GradScaler()

            best_score = 100000.0

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

                valid_loss, predictions, = engine.valid_fn(
                    valid_loader, model, criterion, CONFIG
                )

                # try:
                #     score = roc_auc_score(df_valid.dtarget, predictions)
                # except:
                #     print("pruning, there be nans")
                #     raise optuna.TrialPruned()

                print(f"Valid loss: {valid_loss}")

                if valid_loss < best_score:
                    print(
                        f"results for epoch {epoch + 1}: new best loss {best_score} ====> {valid_loss}"
                    )
                    best_score = valid_loss

                    df_valid['dpred'] = predictions



                else:
                    print(
                        f"results for epoch {epoch + 1}: loss {valid_loss}, best {best_score}"
                    )



            scores.append(best_score)

            
            df_out = pd.concat([df_out, df_valid])


    df_out.to_csv(f"{CONFIG.save_root}/discrim_preds.csv", index=False)

    del model
    gc.collect()
    return np.mean(scores)


if __name__ == "__main__":
    print("training discriminator")

    print(CONFIG)

    # df_real = pd.read_csv(f"{CONFIG.data_loc}/sample_submission.csv")
    df_real = pd.read_csv(f"{CONFIG.data_loc}/train_labels.csv")
    # print(df_real)

    np_fake = np.load(f"{CONFIG.save_root}/TARGETS.npy")
    df_fake = pd.DataFrame(np_fake, columns=['target'])
    df_fake['id'] = df_fake.index

    print(len(df_fake), len(df_real))


    # df_fake = df_fake.sample(len(df_real))
    # print(df_fake)

    print(train_discriminator(df_real, df_fake))