# python -m src.create_folds -C default_config

# export TOKENIZERS_PARALLELISM=true

accelerate launch  train.py -C default_config

