"""Main applications of the currently implemented methods."""

from typing import Any, Dict, List, Tuple, Union

from src.AutoEncoder import autoencoder, training_step
from src.dataloaders.dataloaders import build_dataloader_from_disk
from src.utils import process_batch, set_config

# Try running
#               ipython -c "%run src/horoscopo.py"

# Config JSON
kwargs: Dict = {}

kwargs = {"seed": 42, "device": None, "enforce_reproducibility": False}

kwargs["build_dataloader_from_disk"] = {
    "filename": "./data/binary/dataset_HDF5.h5",
    "batch_size": 2,
    "shuffle": True
}

kwargs["autoencoder"] = {"nc": 13964, "ndf": 256}

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

    # TODO : Currently the model expects a different input shape.
    # output = model(batch)

    if index == 1:
        break
