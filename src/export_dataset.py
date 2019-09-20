# import numpy as np

from src.dataloaders.datasets import LanguageModelDataset, process_file
from src.HilbertMapper import HilbertMapper

# Try running
#               ipython -c "%run src/export_dataset.py"
# Or run this script once
#               python3

# sequence_length = 64
# order = int(np.ceil(np.sqrt(max_sequence_length)))

# Config JSON
kwargs = {
    "language_model_dataset": {
        "filename": "./data/raw/horoscopo_raw.txt",
        "separator": " ",
        "max_number_of_examples": -1
    },
    "hilbert_mapper": {
        "order": 8,
        "number_of_channels": 13964
    },
    "process_file": {
        "vocabulary_size": 13964,
        "output_folder": "./data/binary/",
        "output_filename": "dataset_HDF5.h5",
        "order": 8,
        "name": "hilbert",
        "dtype": "int32"
    }
}

# We initialize our Language Model Dataset
# TODO : We could implement a "get_dataset(**kwargs)" method, in order to choose the desired dataset from config files too
language_model_dataset = LanguageModelDataset(
    **kwargs["language_model_dataset"])

# vocabulary_size = len(language_model_dataset.tokens)

# We initialize our Hilbert Mapper (callable)
hilbert_mapper = HilbertMapper(**kwargs["hilbert_mapper"])

print("Dataset Length : {}".format(len(language_model_dataset)))

# We process our dataset and export it as H5PY
path_to_processed_file = process_file(dataset=language_model_dataset,
                                      mapper=hilbert_mapper,
                                      **kwargs["process_file"])
