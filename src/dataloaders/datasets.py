import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataloaders.Tokens import Tokens


class LanguageModelDataset(Dataset):
    def __init__(self, filename, separator, start_token, end_token):
        super(LanguageModelDataset, self).__init__()

        self.filename = filename
        self.separator = separator

        self.start_token = start_token
        self.end_token = end_token

        # Dictionary
        self.tokens = Tokens()

        self.tokens.add_token(self.start_token)  # start of sentence
        self.tokens.add_token(self.end_token)  # end of sentence

        self.lines, self.out, self.lengths = self.__load__()

    def __load__(self):

        lines = []
        out, lengths = [], []

        with open(self.filename, 'r') as file:

            # Tokenize
            for line in file:
                line = line.replace('.', ' . ').replace('\n', '')
                elements = [
                    element.strip() for element in line.split(self.separator)
                    if element.strip() != ''
                ]
                lines.append(elements)

                # Add words to the Dictionary
                for element in elements:
                    self.tokens.add_token(element)

            # After the words has been added to the dictionary
            for line in lines:
                idxs = [self.tokens.token2index[token] for token in line]
                length = len(idxs)
                lengths.append(length)
                out.append(idxs)

        return lines, out, lengths

    def __getitem__(self, idx):

        x = self.lines[idx]

        sequence_list = []
        embedding = np.array(self.tokens.token2index[self.start_token])
        sequence_list.append(torch.from_numpy(embedding))

        for token in x:

            if token not in self.tokens.token2index:
                raise ValueError("Not in vocabulary: '" + token + "'")

            embedding = np.array(
                self.tokens.token2index[token]
            )  # Check this eventually when using embeddings
            sequence_list.append(torch.from_numpy(embedding))

        embedding = np.array(self.tokens.token2index[self.end_token])
        sequence_list.append(torch.from_numpy(embedding))

        # Sentence :
        #                 "hola como estás ?"
        # Then,
        #         x = [ '<SOS>' , 'hola' , 'cómo' , 'estás' , '?' ]
        #         y = [ 'hola' , 'cómo' , 'estás' , '?' , '<EOS>' ]

        x = torch.stack(sequence_list[:-1])
        y = torch.stack(sequence_list[1:])

        return (x, y)

    def __len__(self):
        return len(self.lines)
