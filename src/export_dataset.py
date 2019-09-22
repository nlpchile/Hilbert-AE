# import numpy as np
import json
from pathlib import Path
from typing import Dict

from src.dataloaders.datasets import LanguageModelDataset, process_file
from src.HilbertMapper import HilbertMapper

# Try running
#               ipython -c "%run src/export_dataset.py"

# sequence_length = 64
# order = int(np.ceil(np.sqrt(max_sequence_length)))

# Config JSON
path_to_config_file = Path("./config.json").absolute()

kwargs: Dict = json.load(fp=path_to_config_file.open())

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
