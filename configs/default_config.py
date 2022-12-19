import os
from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# baseline config

cfg.seed = 42

cfg.data_loc = "/mnt/datastore/g2net2/g2net-detecting-continuous-gravitational-waves/"
cfg.save_root = "/mnt/datastore/g2net2"
cfg.model_name = "baseline"
cfg.save_directory = "hwd1"
cfg.pretrained_model = "resnet18" # "tf_efficientnet_b2_ns" # "resnet34"

cfg.generated_data_loc = "generated"

cfg.batch_size = 64
cfg.n_accumulate = 1
cfg.n_workers = 8
cfg.valid_batch_size = 64

cfg.epochs = 5

# 20k
#0.8330702124431566
#{'lr': 0.0006200000000000001, 'dropout': 0.15000000000000002, 'max_grad': 253, 'warmup': 125, 'p1': 0.1, 'p2': 0.4, 'p3': 0.0, 'gaussian': 2.0}


# 0.8508568042470692 and parameters: {'lr': 0.00046000000000000007, 'dropout': 0.35000000000000003, 'max_grad': 227, 'warmup': 150, 'p1': 0.0, 'p2': 0.30000000000000004, 'p3': 0.30000000000000004, 'optim': 'RMSprop', 'gaussian': 0.9}
cfg.n_folds = 5
cfg.folds = [i for i in range(cfg.n_folds)]
cfg.weight_decay = 0
cfg.warmup = 150
cfg.dropout = 0.35
cfg.sample = False
cfg.max_grad = 227

cfg.p1 = 0
cfg.p2 = 0.3
cfg.p3 = 0.3

cfg.gaussian = 0.9

cfg.optim = "RMSprop"
cfg.scheduler = "cosine"
cfg.ONE_CYCLE_PCT_START = 0.1

cfg.lr = 0.00046

cfg.tta_rounds = 0

cfg.tune_training = False


# generation config
cfg.NUM_SIGNALS = 50000 #600 #7976

cfg.TARGET_HEIGHT = 360
cfg.TARGET_HEIGHT_COMPRESSED = 360
cfg.TARGET_HEIGHT_COMPRESS_RATIO = cfg.TARGET_HEIGHT // cfg.TARGET_HEIGHT_COMPRESSED

cfg.TARGET_WIDTH = 4096
cfg.TARGET_WIDTH_COMPRESSED = 256
cfg.TARGET_WIDTH_COMPRESS_RATIO = cfg.TARGET_WIDTH // cfg.TARGET_WIDTH_COMPRESSED


cfg.F0_lower = 185
cfg.F0_upper = 412

cfg.h0_mean = 0.075
cfg.h0_std = 0.15
cfg.h0_low = 0.01
cfg.h0_high = 0.5

cfg.F1_mean = -9
cfg.F1_std = -10
cfg.F1_low = -1e-7
cfg.F1_high = 1

cfg.band = 0.3

basic_cfg = cfg
