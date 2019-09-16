"""Main applications of the currently implemented methods."""

from typing import Any, Dict, List, Union

import numpy as np

from src.dataloaders.dataloaders import build_dataloader_from_disk
from src.dataloaders.datasets import LanguageModelDataset, process_file
from src.HilbertMapper import HilbertMapper

# Try running "%run src/horoscopo.py" magic in IPython

# TODO : Make an independent script to load the raw txt and save it as binary dataset.

# TODO : This is quite harcoded, we should infer the max sequence length from the dataset or truncate the sequences lenghs by this number
max_sequence_length = 64

# TODO : We should have this value precomputed.
order = int(np.ceil(np.sqrt(max_sequence_length)))

################################################################################
# Config JSONs
kwargs = {
    "language_model_dataset": {
        "filename": "./data/raw/horoscopo_raw.txt",
        "separator": " ",
        "max_number_of_examples": -1
    },
    "process_file": {
        "output_file": "dataset_HDF5.h5",
        "order": order,
        "name": "hilbert",
        "dtype": "int32"
    },
    "build_dataloader_from_disk": {
        "filename": "dataset_HDF5.h5",
        "batch_size": 2,
        "shuffle": True
    }
}
################################################################################

# We initialize our Language Model Dataset
# TODO : We could implement a "get_dataset(**kwargs)" method, in order to choose the desired dataset from config files too
language_model_dataset = LanguageModelDataset(
    **kwargs["language_model_dataset"])

# TODO : We should have this value precomputed.
vocabulary_size = len(language_model_dataset.tokens)

# We initialize our Hilbert Mapper (callable)
hilbert_mapper = HilbertMapper(order=order, number_of_channels=vocabulary_size)

info = {
    "Hilbert Curve Order": order,
    "Vocabulary Size": vocabulary_size,
    "Dataset Length": len(language_model_dataset)
}

# print(info)

# We process our dataset and export it as H5PY
process_file(dataset=language_model_dataset,
             mapper=hilbert_mapper,
             vocabulary_size=vocabulary_size,
             **kwargs["process_file"])

# We load the dataset from disk
dataset = build_dataloader_from_disk(**kwargs["build_dataloader_from_disk"])

for index, batch in enumerate(dataset):

    # shape : [batch_size, order, order, vocabulary_size]
    hilbert_mapped_sequence = batch

    print(hilbert_mapped_sequence.shape)

    if index == 1:
        break
