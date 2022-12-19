import numpy as np
import pandas as pd
import pyfstat
from pyfstat.utils import get_sft_as_arrays
import cv2
import copy
import multiprocessing

from joblib import Parallel, delayed

import os
import glob
from scipy import stats
from tqdm import tqdm
import h5py

import shutil

from . import discriminator

from pathlib import Path

import logging
logging.getLogger('pyfstat').setLevel(logging.ERROR)

import importlib
import argparse
import sys
from copy import copy

import random
import math

from tqdm import tqdm

sys.path.append("configs")

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)

CONFIG = copy(importlib.import_module(parser_args.config).cfg)




TRAIN_DIR = f'{CONFIG.data_loc}'

train_labels = pd.read_csv(f"{TRAIN_DIR}/train_labels.csv")
N_SAMPLES = len(train_labels)

time_delta_recordings = []

for row_idx, row in tqdm(train_labels.iterrows(), total=N_SAMPLES):
    train_id = row['id']
    c = 0
    with h5py.File(f'{TRAIN_DIR}/train/{train_id}.hdf5', 'r') as file:
        timestamps_h = np.array(file[train_id]['H1']['timestamps_GPS'])
        for td in (timestamps_h[1:] - timestamps_h[:-1]):
            if td // 1800 > 0:
                time_delta_recordings.append(td // 1800)
            
        timestamps_l = np.array(file[train_id]['L1']['timestamps_GPS'])
        for td in (timestamps_l[1:] - timestamps_l[:-1]):
            if td // 1800 > 0:
                time_delta_recordings.append(td // 1800)
            
                
time_delta_recordings = np.array(time_delta_recordings, dtype=np.int32)

def get_timestamp_idxs():
    while True:
        return np.cumsum(np.random.choice(time_delta_recordings, CONFIG.TARGET_WIDTH))

# https://pyfstat.readthedocs.io/en/latest/pyfstat.html

def get_signal_config():
    t_start = 1238166018 #+ np.random.randint(0, 200000)
    totaldiff = 200000
    tdiff = int(random.random() * totaldiff) - (totaldiff / 2)
    t_start -= tdiff

    # These parameters describe background noise and data format
    writer_kwargs = {
            'sqrtSX': 1e-23, # Single-sided Amplitude Spectral Density of the noise
            'Tsft': 1800, # Fourier transform time duration
            "SFTWindowType": "tukey",  # Window function to compute short Fourier transforms
            "SFTWindowBeta": 0.01,  # Parameter associated to the window function
            'timestamps': {
                'H1': t_start + 1800 * get_timestamp_idxs(),
                'L1': t_start + 1800 * get_timestamp_idxs(),
            }
       }

    # This class allows us to sample signal parameters from a specific population.
    # Implicitly, sky positions are drawn uniformly across the celestial sphere.
    # PyFstat also implements a convenient set of priors to sample a population
    # of isotropically oriented neutron stars.
    signal_params = {
        'F0': np.random.uniform(50, 500),
        'Band': 0.3, # Frequency band-width around F0 [Hz]
    }
    
    return writer_kwargs, signal_params

def read_data(file):
    file = Path(file)
    with h5py.File(file, "r") as f:
        filename = file.stem
        f = f[filename]
        h1 = f["H1"]
        l1 = f["L1"]
        freq_hz = list(f["frequency_Hz"])

        h1_stft = h1["SFTs"][()]
        h1_timestamp = h1["timestamps_GPS"][()]
        # H2 data
        l1_stft = l1["SFTs"][()]
        l1_timestamp = l1["timestamps_GPS"][()]

        return [h1_stft, h1_timestamp],            [l1_stft, l1_timestamp], np.array(freq_hz)




def generate_sim_noise(ind):
    writer_kwargs, signal_params = get_signal_config()

    writer_kwargs['outdir'] = f'{CONFIG.save_root}/tmp_{ind}'
    writer_kwargs['label'] = 'Signal'

    writer = pyfstat.Writer(**writer_kwargs, **signal_params)

    writer.make_data()
    return writer.sftfilepath




def generate_noise(ind):
    np.random.seed(ind)

    failed_on_real = False
    for i in range(10):
        try:
            target = 0
            snr = 0
            if True:
                writer_path = generate_sim_noise(ind)
                frequency, timestamps, amplitudes = pyfstat.utils.get_sft_as_arrays(
                    writer_path
                )

            
                # Cast to complex 128
                for detector, signal in amplitudes.items():
                    amplitudes[detector] = signal.astype(np.complex128)

                # Normalize Signal
                signal_norm = {
                    'H1': amplitudes['H1'].real ** 2 + amplitudes['H1'].imag ** 2,
                    'L1': amplitudes['L1'].real ** 2 + amplitudes['L1'].imag ** 2,
                }

                for detector, signal in amplitudes.items():
                    amplitudes[detector] = signal.reshape(-1, CONFIG.TARGET_WIDTH_COMPRESSED, CONFIG.TARGET_WIDTH_COMPRESS_RATIO).mean(axis=2)


                for d_idx, (detector, signal) in enumerate(signal_norm.items()):
                    # Save as PNG
                    offset_y = np.random.randint(0, signal.shape[0] - CONFIG.TARGET_HEIGHT)
                    patch_uint8 = signal[offset_y:offset_y + CONFIG.TARGET_HEIGHT, :4096]
                    patch_uint8 = patch_uint8[:, :4096].reshape(
                            CONFIG.TARGET_HEIGHT_COMPRESSED, CONFIG.TARGET_HEIGHT_COMPRESS_RATIO, CONFIG.TARGET_WIDTH_COMPRESSED, CONFIG.TARGET_WIDTH_COMPRESS_RATIO
                        ).mean(axis=(1,3))
                    patch_uint8 = (patch_uint8 - patch_uint8.min())
                    patch_uint8 = patch_uint8 * (255 / patch_uint8.max())
                    patch_uint8 = patch_uint8.astype(np.uint8)
                    cv2.imwrite(f'{CONFIG.save_root}/fake_noise/{ind}_{detector}.png', patch_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 1])


            if (ind % 100) == 0:
                print(f'***** succesfuly created ind {ind}, shape: {amplitudes["H1"].shape}, snr: {int(snr)}')
                sys.stdout.flush()

            # Remove Temp Files
            for fp in glob.glob(f'{CONFIG.save_root}/tmp_{ind}/*'):
                os.remove(fp)

            shutil.rmtree(f'{CONFIG.save_root}/tmp_{ind}', ignore_errors=True)

            return ind, target, snr
        except Exception as e:
                print(f'ind {ind} retrying | {e}')
                sys.stdout.flush()




def get_signal_config_cw_pure():
    t_start = 1238166018 #+ np.random.randint(0, 200000)
    totaldiff = 200000
    tdiff = int(random.random() * totaldiff) - (totaldiff / 2)
    t_start -= tdiff

    # These parameters describe background noise and data format
    writer_kwargs = {
            'sqrtSX': 1e-23, # Single-sided Amplitude Spectral Density of the noise
            'Tsft': 1800, # Fourier transform time duration
            "SFTWindowType": "tukey",  # Window function to compute short Fourier transforms
            "SFTWindowBeta": 0.01,  # Parameter associated to the window function
            "detectors": "H1,L1",
            'timestamps': {
                'H1': t_start + 1800 * get_timestamp_idxs(),
                'L1': t_start + 1800 * get_timestamp_idxs(),
            },
            # 'tstart': t_start,
            # 'duration': (1248512757 - t_start) // 1800 * 1800
       }

    # "1 - 2 orders"
    h0 = 1e-21 # np.clip(1e-23 / (10 ** np.random.normal(1.5, 0.5)), 1e-25, 2e-23)

    signal_params = {
        # polarization angle
        'psi': np.random.uniform(-math.pi / 2, math.pi / 2),
        # phase
        'phi': np.random.uniform(0, math.pi * 2),
        # Cosine of the angle between the source and us. Range: [-1, 1]
        'cosi': np.random.uniform(-1, 1),
        # Central frequency of the band to be generated [Hz]
        'F0': np.random.uniform(50, 500), #np.random.uniform(CONFIG.F0_lower, CONFIG.F0_upper),
        'F1': -10**stats.uniform(-12, 4).rvs(),
        'F2': 0.0,
        'Band': 0.3, # Frequency band-width around F0 [Hz]
        'Alpha': np.random.uniform(0, math.pi * 2), # Right ascension of the source's position on the sky
        'Delta': np.random.uniform(-math.pi / 2, math.pi / 2), # Declination of the source's position on the sky,
        'tp': t_start + 86400 * random.randint(0, 30), # signal offset
        'h0': h0, #* np.clip(np.random.normal(CONFIG.h0_mean, CONFIG.h0_std), CONFIG.h0_low, CONFIG.h0_high), #np.clip(np.random.normal(0.075, 0.15), 0.03, 0.5),
        # 'asini': random.randint(50, 500), # amplitude of signal
        # 'period': random.randint(90, 730) * 86400,
    }


    return writer_kwargs, signal_params


def generate_pure(ind):
    np.random.seed(ind)

    for i in range(10):
        try:


            if True: # no idea why
                writer_kwargs, signal_params = get_signal_config_cw_pure()
                writer_kwargs['outdir'] = f'{CONFIG.save_root}/tmp_{ind}'
                writer_kwargs['label'] = 'Signal'

                writer = pyfstat.BinaryModulatedWriter(**writer_kwargs, **signal_params)
                writer.make_data()
                frequency, timestamps, amplitudes = pyfstat.utils.get_sft_as_arrays(
                    writer.sftfilepath
                )


                # Cast to complex 128
                for detector, signal in amplitudes.items():
                    amplitudes[detector] = signal.astype(np.complex128)


                # Normalize Signal
                signal_norm = {
                    'H1': amplitudes['H1'].real ** 2 + amplitudes['H1'].imag ** 2,
                    'L1': amplitudes['L1'].real ** 2 + amplitudes['L1'].imag ** 2,
                }


                for detector, signal in amplitudes.items():
                    amplitudes[detector] = signal.reshape(-1, CONFIG.TARGET_WIDTH_COMPRESSED, CONFIG.TARGET_WIDTH_COMPRESS_RATIO).mean(axis=2)


                for d_idx, (detector, signal) in enumerate(signal_norm.items()):
                    # Save as PNG
                    offset_y = np.random.randint(0, signal.shape[0] - CONFIG.TARGET_HEIGHT)
                    patch_uint8 = signal[offset_y:offset_y + CONFIG.TARGET_HEIGHT, :4096]
                    patch_uint8 = patch_uint8[:, :4096].reshape(
                            CONFIG.TARGET_HEIGHT_COMPRESSED, CONFIG.TARGET_HEIGHT_COMPRESS_RATIO, CONFIG.TARGET_WIDTH_COMPRESSED, CONFIG.TARGET_WIDTH_COMPRESS_RATIO
                        ).mean(axis=(1,3))
                    
                    patch_uint8 = (patch_uint8 - patch_uint8.min())
                    patch_uint8 = patch_uint8 * (255 / patch_uint8.max())
                    patch_uint8 = patch_uint8.astype(np.uint8)

                    cv2.imwrite(f'{CONFIG.save_root}/pure_signal/{ind}_{detector}.png', patch_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 1])

   

            target = 1
            snr = 0


            if (ind % 100) == 0:
                print(f'***** succesfuly created ind {ind}, shape: {amplitudes["H1"].shape}, snr: {int(snr)}')
                sys.stdout.flush()

            # Remove Temp Files
            for fp in glob.glob(f'{CONFIG.save_root}/tmp_{ind}/*'):
                os.remove(fp)

            shutil.rmtree(f'{CONFIG.save_root}/tmp_{ind}', ignore_errors=True)

            return ind, target, snr
        except Exception as e:
            # if IS_INTERACTIVE:
                print(f'cw ind {ind} retrying | {e}')
                sys.stdout.flush()



def generate_data():
    # print(get_real_noise())

    NUMBER_OF_PROCESSES = 24  #multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=NUMBER_OF_PROCESSES)
    try:
        map_jobs = [(i,) for i in range(0, CONFIG.NUM_SIGNALS)]
        result = pool.starmap(generate_pure, map_jobs)
        # result = pool.starmap(generate_noise, map_jobs)
    finally:
        pool.close()
        pool.join()


    result_sorted = sorted(result, key=lambda t: t[0])
    _, TARGETS, SNRS = np.array(result_sorted).T
    TARGETS = TARGETS.astype(np.int8)
    SNRS = SNRS.astype(np.float32)

    N_SAMPLES_CREATED = len(glob.glob(f'{CONFIG.save_root}/pure_signal/*.png'))
    print(f'N_SAMPLES_CREATED: {N_SAMPLES_CREATED}')

    np.save(f'{CONFIG.save_root}/pure_signal/TARGETS.npy', TARGETS)
    np.save(f'{CONFIG.save_root}/pure_signal/SNRS.npy', SNRS)


    pool = multiprocessing.Pool(processes=NUMBER_OF_PROCESSES)
    try:
        map_jobs = [(i,) for i in range(0, CONFIG.NUM_SIGNALS)]
        result = pool.starmap(generate_noise, map_jobs)
        # result = pool.starmap(generate_noise, map_jobs)
    finally:
        pool.close()
        pool.join()


    result_sorted = sorted(result, key=lambda t: t[0])
    _, TARGETS, SNRS = np.array(result_sorted).T
    TARGETS = TARGETS.astype(np.int8)
    SNRS = SNRS.astype(np.float32)

    N_SAMPLES_CREATED = len(glob.glob(f'{CONFIG.save_root}/fake_noise/*.png'))
    print(f'N_SAMPLES_CREATED: {N_SAMPLES_CREATED}')

    np.save(f'{CONFIG.save_root}/fake_noise/TARGETS.npy', TARGETS)
    np.save(f'{CONFIG.save_root}/fake_noise/SNRS.npy', SNRS)


if __name__ == "__main__":
    print("generating pure signal only")
    print(CONFIG)

    generate_data()
