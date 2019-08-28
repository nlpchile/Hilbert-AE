"""PadCollate class useful as collate_fn for torch DataLoaders."""

from typing import Tuple

import torch


class PadCollate:
    """Padding class useful as collate_fn for torch DataLoaders."""

    def __init__(self, **kwargs):
        """PadCollate."""
        self.kwargs = kwargs

    def __call__(self, batch: Tuple[int]) -> Tuple[torch.Tensor]:
        """

        Call method that applies the path_collate method.

        Args:
            batch (Tuple[int]) : As input it recieves a batch Tuple.

        Returns:
            (Tuple[torch.Tensor]) : It returns a Tuple .

        """
        return __class__.pad_collate(batch, **self.kwargs)

    @staticmethod
    def pad_collate(batch: Tuple[int],
                    batch_first: bool = True,
                    padding_value: int = 0) -> Tuple[torch.Tensor]:
        """
        Apply padding to the input batch Tuple.

        Args:
            batch (Tuple) : A batch Tuple containing (X, Y) input sequences.

            batch_first (bool) : If True, the batch dimension is first,
                                 so [B, T, *].

                                 If False, the timesteps dimension is first,
                                 so then [T, B, *].

                                Default = False.

            padding_value (int) : An int value that is considered as padding
                                  symbol.

        Returns:
            (Tuple[torch.Tensor]) : It returns a Tuple containing
                                    (X, Y, lengths) tensors.

        """
        batch.sort(key=lambda b: len(b[0]), reverse=True)

        X_seq, Y_seq = zip(*batch)

        X_seq = [torch.tensor(X) for X in X_seq]
        Y_seq = [torch.tensor(Y) for Y in Y_seq]

        # torch.tensor infers the dtype automatically, while torch.Tensor
        # returns a torch.FloatTensor.
        # https://discuss.pytorch.org/t/difference-between-torch-tensor-and-torch-tensor/30786/2
        lengths = torch.tensor([len(X) for X in X_seq])

        X = torch.nn.utils.rnn.pad_sequence(sequences=X_seq,
                                            batch_first=batch_first,
                                            padding_value=padding_value)

        Y = torch.nn.utils.rnn.pad_sequence(sequences=Y_seq,
                                            batch_first=batch_first,
                                            padding_value=padding_value)

        return (X, Y, lengths)
