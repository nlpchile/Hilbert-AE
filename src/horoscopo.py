"""Main applications of the currently implemented methods."""

from typing import Any, Dict, List, Union

from src.dataloaders.dataloaders import build_dataloader_from_disk
from src.utils import process_batch

# Try running
#               ipython -c "%run src/horoscopo.py"

# Config JSON
kwargs = {
    "build_dataloader_from_disk": {
        "filename": "dataset_HDF5.h5",
        "batch_size": 2,
        "shuffle": True
    }
}

# We load the dataset from disk
dataset = build_dataloader_from_disk(**kwargs["build_dataloader_from_disk"])

for index, batch in enumerate(dataset):

    # hilbert_map
    # shape : [batch_size, order, order, vocabulary_size]

    batch = process_batch(batch=batch)

    if index == 1:
        break
