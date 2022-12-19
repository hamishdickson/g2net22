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
        'F0': np.clip(np.random.normal(100, 25), 50, 350),
        'Band': 0.36, # Frequency band-width around F0 [Hz]
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


def get_real_noise():
    dir_paths = glob.glob(f"{CONFIG.save_root}/real/npy/*/", recursive = True)
    dir_path = np.random.choice(dir_paths)

    idx = dir_path.split("/")[6]

    (h1_sfts, h1_ts), (l1_sfts, l1_ts), freq = read_data(f'{CONFIG.data_loc}/test/{idx}.hdf5')


    return {
        "H1": np.load(f"{dir_path}/H1.npy"),
        "L1": np.load(f"{dir_path}/L1.npy"),
    }


def generate_sim_noise(ind):
    writer_kwargs, signal_params = get_signal_config()

    writer_kwargs['outdir'] = f'{CONFIG.save_root}/tmp_{ind}'
    writer_kwargs['label'] = 'Signal'

    writer = pyfstat.Writer(**writer_kwargs, **signal_params)

    writer.make_data()
    return writer.sftfilepath




def generate_noise(ind):
    np.random.seed(ind)

    gen_real = False #np.random.choice([True, False], p=[0.1, 0.9])

    failed_on_real = False
    for i in range(10):
        try:
            target = 0
            snr = 0

            if gen_real and (ind % 100 != 0): # no idea why
                amplitudes_noise = get_real_noise()

                # Normalize Signal
                signal_norm = {
                    'H1': amplitudes_noise['H1'].real ** 2 + amplitudes_noise['H1'].imag ** 2,
                    'L1': amplitudes_noise['L1'].real ** 2 + amplitudes_noise['L1'].imag ** 2,
                }

                for d_idx, (detector, signal) in enumerate(signal_norm.items()):
                    noise = amplitudes_noise[detector].real ** 2 + amplitudes_noise[detector].imag ** 2
                    noise = noise.reshape(360, 256, -1).mean(axis=2)
                        
                    noise = (noise - noise.min())
                    noise = noise * (255 / noise.max())
                    noise = noise.astype(np.uint8)


                    cv2.imwrite(f'{CONFIG.save_root}/generated/{ind}_{detector}.png', noise, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            else:
            # if True:
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
                    cv2.imwrite(f'{CONFIG.save_root}/generated/{ind}_{detector}.png', patch_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 1])


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




def get_signal_config_cw():
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

    # "1 - 2 orders"
    h0 = np.clip(1e-23 / (10 ** np.random.normal(1.5, 0.5)), 1e-25, 2e-23)
    f0 = np.clip(np.random.normal(100, 25), 50, 350) #  np.random.uniform(100, 350)

    signal_params = {
        # polarization angle
        'psi': np.random.uniform(-math.pi / 2, math.pi / 2),
        # phase
        'phi': np.random.uniform(0, math.pi * 2),
        # Cosine of the angle between the source and us. Range: [-1, 1]
        'cosi': np.random.uniform(-1, 1),
        # Central frequency of the band to be generated [Hz]
        'F0': f0, #np.random.uniform(CONFIG.F0_lower, CONFIG.F0_upper),
        'F1': np.clip(np.random.normal(-1e-15, 1e-20), -1e-13, -1e-20), #np.clip(-10**np.random.normal(CONFIG.F1_mean, CONFIG.F1_std), CONFIG.F1_low, CONFIG.F1_high),
        'F2': 0.0,
        'Band': 0.36, # Frequency band-width around F0 [Hz]
        'Alpha': np.random.uniform(0, math.pi * 2), # Right ascension of the source's position on the sky
        'Delta': np.random.uniform(-math.pi / 2, math.pi / 2), # Declination of the source's position on the sky,
        'tp': t_start + 86400 * random.randint(0, 30), # signal offset
        'h0': h0, #* np.clip(np.random.normal(CONFIG.h0_mean, CONFIG.h0_std), CONFIG.h0_low, CONFIG.h0_high), #np.clip(np.random.normal(0.075, 0.15), 0.03, 0.5),
        'asini': random.randint(50, 500), # amplitude of signal
        'period': random.randint(90, 730) * 86400,
    }


    return writer_kwargs, signal_params


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
            'timestamps': {
                'H1': t_start + 1800 * get_timestamp_idxs(),
                'L1': t_start + 1800 * get_timestamp_idxs(),
            }
       }

    # "1 - 2 orders"
    h0 = 1e-21 # np.clip(1e-23 / (10 ** np.random.normal(1.5, 0.5)), 1e-25, 2e-23)
    f0 = np.clip(np.random.normal(100, 25), 50, 350) #  np.random.uniform(100, 350)

    signal_params = {
        # polarization angle
        'psi': np.random.uniform(-math.pi / 2, math.pi / 2),
        # phase
        'phi': np.random.uniform(0, math.pi * 2),
        # Cosine of the angle between the source and us. Range: [-1, 1]
        'cosi': np.random.uniform(-1, 1),
        # Central frequency of the band to be generated [Hz]
        'F0': f0, #np.random.uniform(CONFIG.F0_lower, CONFIG.F0_upper),
        'F1': np.clip(np.random.normal(-1e-15, 1e-20), -1e-13, -1e-20), #np.clip(-10**np.random.normal(CONFIG.F1_mean, CONFIG.F1_std), CONFIG.F1_low, CONFIG.F1_high),
        'F2': 0.0,
        'Band': 0.36, # Frequency band-width around F0 [Hz]
        'Alpha': np.random.uniform(0, math.pi * 2), # Right ascension of the source's position on the sky
        'Delta': np.random.uniform(-math.pi / 2, math.pi / 2), # Declination of the source's position on the sky,
        'tp': t_start + 86400 * random.randint(0, 30), # signal offset
        'h0': h0, #* np.clip(np.random.normal(CONFIG.h0_mean, CONFIG.h0_std), CONFIG.h0_low, CONFIG.h0_high), #np.clip(np.random.normal(0.075, 0.15), 0.03, 0.5),
        'asini': random.randint(50, 500), # amplitude of signal
        'period': random.randint(90, 730) * 86400,
    }


    return writer_kwargs, signal_params


def generate_cw(ind):
    np.random.seed(ind)

    gen_real = False # np.random.choice([True, False], p=[0.1, 0.9])
    for i in range(10):
        try:


            if gen_real and (ind % 100 != 0): # no idea why
                print(f"gen real cw {ind}")
                writer_kwargs, signal_params = get_signal_config_cw_pure()
                writer_kwargs['outdir'] = f'{CONFIG.save_root}/tmp_{ind}'
                writer_kwargs['label'] = 'Signal'

                writer = pyfstat.BinaryModulatedWriter(**writer_kwargs, **signal_params)
                writer.make_data()
                frequency, timestamps, signal_amplitudes = pyfstat.utils.get_sft_as_arrays(
                    writer.sftfilepath
                )
                SIGNAL_LOW = 0.02
                SIGNAL_HIGH = 0.10
                K = np.random.uniform(SIGNAL_LOW, SIGNAL_HIGH)
                amplitudes_noise = get_real_noise()


                # Cast to complex 128
                for detector, signal in signal_amplitudes.items():
                    signal_amplitudes[detector] = signal.astype(np.complex128)


                # Normalize Signal
                signal_norm = {
                    'H1': signal_amplitudes['H1'].real ** 2 + signal_amplitudes['H1'].imag ** 2,
                    'L1': signal_amplitudes['L1'].real ** 2 + signal_amplitudes['L1'].imag ** 2,
                }


                for detector, signal in signal_amplitudes.items():
                    signal_amplitudes[detector] = signal.reshape(-1, CONFIG.TARGET_WIDTH_COMPRESSED, CONFIG.TARGET_WIDTH_COMPRESS_RATIO).mean(axis=2)

                # for detector, signal in amplitudes_noise.items():
                #     amplitudes_noise[detector] = signal.reshape(-1, CONFIG.TARGET_WIDTH_COMPRESSED, CONFIG.TARGET_WIDTH_COMPRESS_RATIO).mean(axis=2)


                # raise Exception(amplitudes_noise['H1'].shape, signal_amplitudes['H1'].shape)

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

                    noise = amplitudes_noise[detector].real ** 2 + amplitudes_noise[detector].imag ** 2
                    noise = noise.reshape(360, 256, -1).mean(axis=2)
                    
                    noise = (noise - noise.min())
                    noise = noise * (255 / noise.max())
                    noise = noise.astype(np.uint8)

                    patch_uint8 = K*patch_uint8 + noise


                    cv2.imwrite(f'{CONFIG.save_root}/generated/{ind}_{detector}.png', patch_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 1])

                

            else:
            # if True:
                writer_kwargs, signal_params = get_signal_config_cw()
                writer_kwargs['outdir'] = f'{CONFIG.save_root}/tmp_{ind}'
                writer_kwargs['label'] = 'Signal'

                writer = pyfstat.BinaryModulatedWriter(**writer_kwargs, **signal_params)

                writer.make_data()
                # SNR can be compute from a set of SFTs for a specific set
                # of parameters as follows:
                # snr = pyfstat.SignalToNoiseRatio.from_sfts(
                #     F0=writer.F0, sftfilepath=writer.sftfilepath
                # )
                
                # squared_snr = snr.compute_snr2(
                #     Alpha=writer.Alpha, 
                #     Delta=writer.Delta,
                #     psi=writer.psi,
                #     phi=writer.phi, 
                #     h0=writer.h0,
                #     cosi=writer.cosi
                # )
                
                # snr = np.sqrt(squared_snr)
                
                # Data can be read as a numpy array using PyFstat
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
                    cv2.imwrite(f'{CONFIG.save_root}/generated/{ind}_{detector}.png', patch_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 1])

                # print("******", amplitudes["H1"].shape)



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

def generate_random(idx):
    if np.random.choice([True, False], p=[2/3, 1/3]):
        return generate_cw(idx)
    else:
        return generate_noise(idx)


def generate_data():
    # print(get_real_noise())

    NUMBER_OF_PROCESSES = 24  #multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=NUMBER_OF_PROCESSES)
    try:
        map_jobs = [(i,) for i in range(0, CONFIG.NUM_SIGNALS)]
        result = pool.starmap(generate_random, map_jobs)
        # result = pool.starmap(generate_noise, map_jobs)
    finally:
        pool.close()
        pool.join()


    result_sorted = sorted(result, key=lambda t: t[0])
    _, TARGETS, SNRS = np.array(result_sorted).T
    TARGETS = TARGETS.astype(np.int8)
    SNRS = SNRS.astype(np.float32)

    N_SAMPLES_CREATED = len(glob.glob(f'{CONFIG.save_root}/generated/*.png'))
    print(f'N_SAMPLES_CREATED: {N_SAMPLES_CREATED}')

    np.save(f'{CONFIG.save_root}/TARGETS.npy', TARGETS)
    np.save(f'{CONFIG.save_root}/SNRS.npy', SNRS)


if __name__ == "__main__":
    print("generating new data")
    print(CONFIG)

    generate_data()

    print("training discriminator")

    print(CONFIG)

    df_real = pd.read_csv(f"{CONFIG.data_loc}/sample_submission.csv")
    # df_real = pd.read_csv(f"{CONFIG.data_loc}/train_labels.csv")
    # print(df_real)

    np_fake = np.load(f"{CONFIG.save_root}/TARGETS.npy")
    df_fake = pd.DataFrame(np_fake, columns=['target'])
    df_fake['id'] = df_fake.index

    print(len(df_fake), len(df_real))


    # df_fake = df_fake.sample(len(df_real))
    # print(df_fake)

    # print(discriminator.train_discriminator(df_real, df_fake))