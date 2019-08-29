"""This module implements the Datasets classes and methods."""

from typing import Callable, List, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataloaders.Tokens import Tokens


class LanguageModelDataset(Dataset):
    """Language Model Dataset."""

    def __init__(self,
                 filename: str,
                 separator: str,
                 max_number_of_examples: int = -1,
                 **kwargs) -> None:
        """
        Language Model Dataset.

        Args:
            filename (str) : Path to the raw text.

            separator (str) : A string identifier used to split the string
                              into its respective tokens.

            max_number_of_examples (int) : Max number of examples to load. To
                                           load them all use
                                           max_number_of_examples=-1 .

        """
        super(LanguageModelDataset, self).__init__()

        self.filename = filename
        self.separator = separator
        self.max_number_of_examples = max_number_of_examples

        # Dictionaries
        self.tokens = Tokens()

        self.lines, self.out, self.lengths = self.__load__()

        # TODO: Perhaps we could add a stride parameter and see what happens
        # with its latent space representation if we train a model to predict
        # the next character given a context, for example, when using teacher
        # forcing.

    def __load__(self) -> Tuple[List[List[str]], List[List[int]], List[int]]:
        """Return the tokenized strings, its respectives tokens and lengths."""
        lines = []
        out, lengths = [], []

        with open(self.filename, "r") as file:

            # Tokenize
            for line in file:
                line = line.replace(".", " . ").replace("\n", "")
                elements = [
                    element.strip() for element in line.split(self.separator)
                    if element.strip() != ""
                ]
                lines.append(elements)

                # Add words to the Dictionary
                for element in elements:
                    self.tokens.add_token(element)

                if len(
                        lines
                ) > self.max_number_of_examples and self.max_number_of_examples != -1:
                    break

            # After the words has been added to the dictionary
            for line in lines:
                indexes = [self.tokens.token2index[token] for token in line]
                length = len(indexes)
                lengths.append(length)
                out.append(indexes)

        return lines, out, lengths

    def __getitem__(self, index: int):
        """
        Get one item at a time.

        Args:
            index (int) : An integer containing the index of the item to
                          retrieve.

        Returns:
            (torch.Tensor) :  It returns an item as a torch.Tensor.

        """
        x = self.lines[index]

        sequence_list = []

        for token in x:

            if token not in self.tokens.token2index:
                raise ValueError("Not in vocabulary: '" + token + "'")

            # Check this eventually when using embeddings
            embedding = np.array(self.tokens.token2index[token])

            sequence_list.append(torch.from_numpy(embedding))

        x = torch.stack(sequence_list)

        return x

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.lines)


class HilbertDataset(torch.utils.data.Dataset):
    """Hilbert Dataset."""

    def __init__(self, filename: str, **kwargs) -> None:
        """
        Hilbert Dataset.

        Args:
            filename (str) : A string containing the filename to the dataset.

        """
        super(HilbertDataset, self).__init__()

        # TODO : We could rename this class and make it a more general dataloader that reads from binary files

        # we could use a list of keys
        self.key_name = "hilbert"

        self.h5pyfile = h5py.File(filename, "r")
        self.number_of_sequences = self.h5pyfile[self.key_name].shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """

        Get one item at a time.

        Args:
            index (int) : An integer containing the index of the item to
                          retrieve.

        Returns:
            (torch.Tensor) : It returns an item as a torch.Tensor.

        """
        mapped_sequence = torch.Tensor(
            self.h5pyfile[self.key_name][index, :, :, :]).type(
                dtype=torch.long)

        return mapped_sequence

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.number_of_sequences


def process_file(dataset: Dataset,
                 output_file: str,
                 order: int,
                 vocabulary_size: int,
                 mapper: Callable = None,
                 name: str = "hilbert",
                 dtype: Union[str, type] = "int32",
                 **kwargs) -> None:
    """
    Create a binary file from a given dataset.

    Args:
        dataset (Dataset): A torch Dataset object.

        output_file (str): A string with the output file name.

        name (str) : Name of the dataset.

        dtype (Union[int, type]) : Numpy dtype or string. It overrides the
                                      data dtype.

        mapper (Callable): A callable that recieves an input item from the
                           Dataset and outputs its mapped representation.

        order (int): An integer with the Hilbert curve order.

        vocabulary_size (int): Vocabulary size.

    """
    # TODO: Perhaps we could rename this method and process the input earlier, perhaps using the Callable mapper as an argument for the Dataset Class, and then infer the data shape also from the Dataset.

    # TODO: Perhaps we can get an item "shape" Tuple as input instead of an "order" and "vocabulary_size" argument.
    shape: Tuple = (order, order, vocabulary_size)

    # create output file
    f = h5py.File(output_file, "w")

    current_buffer_size = 1
    current_buffer_allocation = 0

    dataset_HDF5 = f.create_dataset(
        name=name,
        shape=(current_buffer_size, *shape),
        dtype=dtype,
        maxshape=(None, *shape),
    )

    for index in range(0, len(dataset)):

        data = dataset[index]

        if mapper is not None:
            data = mapper(data)

        if current_buffer_allocation >= current_buffer_size:
            current_buffer_size = current_buffer_size + 1
            dataset_HDF5.resize((current_buffer_size, *shape))

        dataset_HDF5[current_buffer_allocation] = data

        current_buffer_allocation += 1

    # TODO : Return absolute output_path
    return
