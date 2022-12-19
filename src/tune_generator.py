import pandas as pd
import optuna
import numpy as np

from . import generate_data
from . import discriminator

import importlib
import argparse
import sys
from copy import copy


import logging
logging.getLogger('pyfstat').setLevel(logging.ERROR)


sys.path.append("configs")

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)

CONFIG = copy(importlib.import_module(parser_args.config).cfg)


def create_dataset_and_discriminate():
    
    # df_test = pd.read_csv(f"{CONFIG.data_loc}sample_submission.csv")
    df_test = pd.read_csv(f"{CONFIG.data_loc}/sample_submission.csv")
    # df_test = df_test.sample(CONFIG.NUM_SIGNALS)

    # CONFIG.NUM_SIGNALS = len(df_test)
    print(CONFIG)
    generate_data.generate_data()
 
    targets = np.load(f'{CONFIG.save_root}/TARGETS.npy')
    df_fake = pd.DataFrame(targets, columns=['target'])
    df_fake['id'] = df_fake.index

    score = discriminator.train_discriminator(df_test, df_fake)

    return score

def objective(trial):
    # CONFIG.band = trial.suggest_float("band", 0.1, 0.4)
    # CONFIG.SFTWindowBeta = trial.suggest_float("SFTWindowBeta", 0.00001, 0.1, log=True)
    # 185 412
    CONFIG.F0_lower = trial.suggest_int("F0_lower", 50, 400)
    CONFIG.F0_upper = trial.suggest_int("F0_upper", 51, 500)

    CONFIG.band = trial.suggest_float("band", 0.05, 1.0, step=0.05)

    CONFIG.h0_mean = trial.suggest_float("h0_mean", 0.01, 0.5, step=0.01)
    CONFIG.h0_std = trial.suggest_float("h0_std", 0.01, 0.5, step=0.01)
    CONFIG.h0_low = trial.suggest_float("h0_low", 0.001, 0.2, step=0.001)
    CONFIG.h0_high = trial.suggest_float("h0_high", 0.01, 0.5, step=0.01)

    CONFIG.F1_mean = trial.suggest_int("F1_mean", -12, -6)
    CONFIG.F1_std = trial.suggest_float("F1_std", 0.1, 10, step=0.1)
    CONFIG.F1_low = trial.suggest_float("F1_low", -1e-6, -1e-13, step=1e-13)
    CONFIG.F1_high = trial.suggest_float("F1_high", -1e-9, 1, step=1e-9)

    return create_dataset_and_discriminate()


if __name__ == "__main__":
    print("tune generator")
    

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25)

    print("best trail")
    print(study.best_value)
    print(study.best_params)
    print(study.trials_dataframe().sort_values('value'))
