"""Main applications of the currently implemented methods."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from src.AutoEncoder import autoencoder, training_step
from src.dataloaders.dataloaders import build_dataloader_from_disk
from src.utils import process_batch, set_config

# Try running
#               ipython -c "%run src/horoscopo.py"

# Config JSON
path_to_config_file = Path("./config.json").absolute()

kwargs: Dict = json.load(fp=path_to_config_file.open())

# Config Device
device = set_config(seed=kwargs["seed"],
                    device=kwargs["device"],
                    enforce_reproducibility=kwargs["enforce_reproducibility"])

# We load the dataset from disk
dataset = build_dataloader_from_disk(**kwargs["build_dataloader_from_disk"])

# We initialize the model
model = autoencoder(**kwargs["autoencoder"]).to(device)

for index, batch in enumerate(dataset):

    # shape : [batch_size, order, order, vocabulary_size]
    # batch = hilbert_map

    batch = batch.to(device)

    # shape : [batch_size, vocabulary_size, order, order]
    batch = process_batch(batch=batch)

    output = model(batch)

    if index == 1:
        break
