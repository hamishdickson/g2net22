
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

import numpy as np
import importlib
import argparse
import sys
from copy import copy

import glob

sys.path.append("configs")

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)

CONFIG = copy(importlib.import_module(parser_args.config).cfg)


if __name__ == "__main__":
    print("Creating folds")
    print(CONFIG)

    df_training = pd.read_csv(f"{CONFIG.data_loc}/train_labels.csv")
    df_training = df_training[df_training['target'] >= 0] # there are 3 with -1 as target

    # realistic noise only
    realistic_noise_ids = [x.split("/")[6] for x in glob.glob('/mnt/datastore/g2net2/realistic_noise/images/*/*.png')]
    df_realistic_noise = pd.DataFrame(realistic_noise_ids, columns=['id'])
    df_realistic_noise['target'] = 0
    df_realistic_noise['image_loc'] = "realistic_noise"

    # sim noise only
    fake_noise_np = np.load(f"{CONFIG.save_root}/fake_noise/TARGETS.npy")
    df_fake_noise = pd.DataFrame(fake_noise_np, columns=['target'])
    df_fake_noise['id'] = df_fake_noise.index
    df_fake_noise['image_loc'] = "fake_noise"

    df_fake_noise = df_fake_noise.sample(10000)

    # pure signal only
    pure_signal_np = np.load(f"{CONFIG.save_root}/pure_signal/TARGETS.npy")
    df_pure_signal = pd.DataFrame(pure_signal_np, columns=['target'])
    df_pure_signal['id'] = df_pure_signal.index


    ##############


    # split pure signal
    df_pure_signal = df_pure_signal.sample(frac=1, random_state=42).reset_index(drop=True)
    kfld = KFold(n_splits=CONFIG.n_folds)
    df_pure_signal["kfold"] = -1
    for f, (t_, v_) in enumerate(kfld.split(df_pure_signal)):
        df_pure_signal.loc[v_, "kfold"] = int(f)

    # split fake noise
    df_fake_noise = pd.concat([df_fake_noise, df_realistic_noise])

    df_fake_noise = df_fake_noise.sample(frac=1, random_state=42).reset_index(drop=True)
    kfld = KFold(n_splits=CONFIG.n_folds)
    df_fake_noise["kfold"] = -1
    for f, (t_, v_) in enumerate(kfld.split(df_fake_noise)):
        df_fake_noise.loc[v_, "kfold"] = int(f)


    sim_dataset = []
    for i in range(1):
        for _, row in df_fake_noise.iterrows():
            make_real = np.random.choice([True, False])
            kfold = row.kfold
            idx = row.id
            image_loc = row.image_loc
            if make_real:
                target = 1
                signal_idx = np.random.choice(df_pure_signal[df_pure_signal['kfold'] == kfold].id.values)
            else:
                target = 0
                signal_idx = -1

            sim_dataset.append((idx, target, signal_idx, image_loc, kfold))


    df_sim = pd.DataFrame(sim_dataset, columns=['id', 'target', 'sim_id', 'image_loc', 'kfold'])
    


    df_training['sim_id'] = -1
    df_training['image_loc'] = "train"
    df_training = df_training.sample(frac=1, random_state=42).reset_index(drop=True)
    cv = StratifiedKFold(n_splits=CONFIG.n_folds)
    df_training["kfold"] = -1

    for f, (t_, v_) in enumerate(cv.split(df_training, df_training.target)):
        df_training.loc[v_, "kfold"] = int(f)


    df = pd.concat([df_training, df_sim])
    df = df.sample(frac=1).reset_index(drop=True)

    print(df.head())
    print(len(df))
    df.to_csv(f"{CONFIG.data_loc}/train_folds.csv", index=False)
