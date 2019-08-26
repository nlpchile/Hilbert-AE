from typing import Tuple

import torch


class PadCollate:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, batch: int):  # TODO : Add type hint
        return pad_collate(batch, **self.kwargs)


def pad_collate(batch: int, batch_first: bool = True, padding_value: int = 0
                ) -> Tuple[torch.nn.utils.rnn.pad_sequence, torch.nn.utils.rnn.
                           pad_sequence, torch.tensor]:

    batch.sort(key=lambda b: len(b[0]), reverse=True)

    X_seq, Y_seq = zip(*batch)

    X_seq = [torch.tensor(X) for X in X_seq]
    Y_seq = [torch.tensor(Y) for Y in Y_seq]

    lengths = torch.tensor([len(X) for X in X_seq])

    X = torch.nn.utils.rnn.pad_sequence(sequences=X_seq,
                                        batch_first=batch_first,
                                        padding_value=padding_value)

    Y = torch.nn.utils.rnn.pad_sequence(sequences=Y_seq,
                                        batch_first=batch_first,
                                        padding_value=padding_value)

    return (X, Y, lengths)
