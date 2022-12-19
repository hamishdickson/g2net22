import pandas as pd
import cv2
import h5py
import numpy as np

from tqdm import tqdm

from joblib import Parallel, delayed

import importlib
import argparse
import sys
from copy import copy


sys.path.append("configs")

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)

CONFIG = copy(importlib.import_module(parser_args.config).cfg)


def run_all(df, mode):
    Parallel(n_jobs=32, backend="multiprocessing")(
        delayed(convert_and_save)(idx, mode) for idx in df.id.values
    )


def convert_and_save(idx, mode):
        filename = f"{CONFIG.data_loc}{mode}/{idx}.hdf5"


        with h5py.File(filename, 'r') as file:
            SFT_H = np.array(file[idx]['H1']['SFTs'], dtype=np.complex128)
            SFT_L = np.array(file[idx]['L1']['SFTs'], dtype=np.complex128)

        # Normalize Signal
        signal_norm = {
            'H1': SFT_H.real ** 2 + SFT_H.imag ** 2,
            'L1': SFT_L.real ** 2 + SFT_L.imag ** 2,
        }
    
        # For each detector, process sample and make predictions
        for d_idx, (detector, signal) in enumerate(signal_norm.items()):
            # Save as PNG
            img_ = signal[:, :4096]
            img_ = img_.reshape(CONFIG.TARGET_HEIGHT, CONFIG.TARGET_WIDTH_COMPRESSED, CONFIG.TARGET_WIDTH_COMPRESS_RATIO).mean(axis=2)
            img_ = (img_ - img_.min())
            img_ = img_ * (255 / img_.max())
            img_ = img_.astype(np.uint8)

            # raise Exception(img.shape)
            cv2.imwrite(f'{CONFIG.save_root}/preprocessed_{mode}/{idx}_{detector}.png', img_, [cv2.IMWRITE_PNG_COMPRESSION, 1])


if __name__ == "__main__":
    print("preprocessing train and test files")

    print("starting train")

    train = pd.read_csv(f"{CONFIG.data_loc}/train_labels.csv")
    train = train[train['target'] >= 0]
    run_all(train, "train")


    print("starting test")
    test = pd.read_csv(f"{CONFIG.data_loc}/sample_submission.csv")
    run_all(test, "test")