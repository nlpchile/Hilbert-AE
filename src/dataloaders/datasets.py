from typing import Callable, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataloaders.Tokens import Tokens
from src.utils import HilbertMapper, hilbert_curve, sequence2hilbert


class LanguageModelDataset(Dataset):

    # TODO: Perhaps we could add a stride parameter and see what happens
    # with its latent space representation if we train a model to predict
    # the next character given a context, for example, when using teacher
    # forcing.

    def __init__(
            self,
            filename: str,
            separator: str,
            max_examples: int = -1,
            #start_token: str = '<SOS>',
            #end_token: str = '<EOS>',
    ) -> None:
        super(LanguageModelDataset, self).__init__()

        self.filename = filename
        self.separator = separator
        self.max_examples = max_examples
        #self.start_token = start_token
        #self.end_token = end_token

        # Dictionaries
        self.tokens = Tokens()
        #self.tokens.add_token(self.start_token)  # start of sentence
        #self.tokens.add_token(self.end_token)  # end of sentence

        self.lines, self.out, self.lengths = self.__load__()

    def __load__(self) -> Tuple[List[List[str]], List[List[int]], List[int]]:
        '''

        It returns a list of tokenized strings, the list of its
        respectives tokens, and the lengths of each one.

        '''

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

                if len(lines) > self.max_examples and self.max_examples != -1:
                    break

            # After the words has been added to the dictionary
            for line in lines:
                idxs = [self.tokens.token2index[token] for token in line]
                length = len(idxs)
                lengths.append(length)
                out.append(idxs)

        return lines, out, lengths

    def __getitem__(self, idx: int):

        x = self.lines[idx]

        sequence_list = []

        #embedding = np.array(self.tokens.token2index[self.start_token])
        #sequence_list.append(torch.from_numpy(embedding))

        for token in x:

            if token not in self.tokens.token2index:
                raise ValueError("Not in vocabulary: '" + token + "'")

            # Check this eventually when using embeddings
            embedding = np.array(self.tokens.token2index[token])

            sequence_list.append(torch.from_numpy(embedding))

        #embedding = np.array(self.tokens.token2index[self.end_token])
        #sequence_list.append(torch.from_numpy(embedding))

        # Sentence :
        #                 "hola como estás ?"
        # Then,
        #         x = [ '<SOS>' , 'hola' , 'cómo' , 'estás' , '?' ]
        #         y = [ 'hola' , 'cómo' , 'estás' , '?' , '<EOS>' ]

        x = torch.stack(sequence_list)

        return x

    def __len__(self) -> int:

        return len(self.lines)


class HilbertDataset(torch.utils.data.Dataset):
    def __init__(self, filename: str) -> None:
        super(HilbertDataset, self).__init__()

        # TODO : We could rename it and make it a more general
        # dataloader that reads from binary files

        # we could use a list of keys
        self.key_name = "hilbert"

        self.h5pyfile = h5py.File(filename, "r")
        self.num_seq = self.h5pyfile[self.key_name].shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:

        mapped_sequence = torch.Tensor(
            self.h5pyfile[self.key_name][index, :, :, :]).type(
                dtype=torch.long)

        return mapped_sequence

    def __len__(self) -> int:
        return self.num_seq


def process_file(dataset: Dataset, mapper: Callable, output_file: str,
                 order: int, vocabulary_size: int) -> None:

    # TODO : Test process_file method

    # create output file
    f = h5py.File(output_file, "w")

    current_buffer_size = 1
    current_buffer_allocation = 0

    # mapper.shape = tuple( [dim for dim in hilbert_map.shape] + [number_of_channels])

    dataset_HDF5 = f.create_dataset(
        "hilbert", (current_buffer_size, order, order, vocabulary_size),
        maxshape=(None, order, order, vocabulary_size),
        dtype="int32")

    for idx in range(0, len(dataset)):

        encoded_sequence = dataset[idx]

        mapped_sequence = mapper(encoded_sequence)

        if current_buffer_allocation >= current_buffer_size:
            current_buffer_size = current_buffer_size + 1
            dataset_HDF5.resize(
                (current_buffer_size, order, order, vocabulary_size))

        dataset_HDF5[current_buffer_allocation] = mapped_sequence

        current_buffer_allocation += 1

    return
