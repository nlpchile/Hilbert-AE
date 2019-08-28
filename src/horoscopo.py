"""Main applications of the currently implemented methods."""

from typing import Any, Dict

import numpy as np

from src.dataloaders.dataloaders import build_dataloader_from_disk
from src.dataloaders.datasets import LanguageModelDataset, process_file
from src.utils import HilbertMapper

# Try running "%run src/horoscopo.py" magic in IPython

###############################################################################
# TODO : we should aim to be able to load all configurations, or example, from a config JSON file

# Configs
kwargs: Dict[str, Any] = {}

# Language Model Dataset
filename_raw = "./data/raw/horoscopo_raw.txt"

kwargs["language_model_dataset"] = {
    "filename": filename_raw,
    "separator": " ",
    "max_number_of_examples": 100
}

# TODO : This is quite harcoded, we should infer the max sequence length from the dataset or truncate the sequences lenghs by this number
max_sequence_length = 64
order = int(np.ceil(np.sqrt(max_sequence_length)))

# DataLoader
batch_size = 2
shuffle = True

path_to_dataset_HDF5 = "dataset_HDF5.h5"

# Process file args
kwargs["process_file"] = {
    "output_file": path_to_dataset_HDF5,
    "order": order,
    "name": "hilbert",
    "dtype": "int32"
}

# Build dataloader from disk
kwargs["build_dataloader_from_disk"] = {
    "filename": path_to_dataset_HDF5,
    "batch_size": batch_size,
    "shuffle": shuffle
}
################################################################################

# We initialize our Language Model Dataset
# TODO : We could implement a "get_dataset(**kwargs)" method, in order to choose the desired dataset from config files too
language_model_dataset = LanguageModelDataset(
    **kwargs["language_model_dataset"])

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

for index, minibatch in enumerate(dataset):

    # shape : [batch_size, order, order, vocabulary_size]
    hilbert_mapped_sequence = minibatch

    print(hilbert_mapped_sequence.shape)

    if index > 1:
        break
