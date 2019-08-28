"""This module implement the Hilbert Mapper Class."""
from typing import List, Tuple

import numpy as np


class HilbertMapper:
    """Hilbert Mapper Class."""

    def __init__(self, order: int, number_of_channels: int, **kwargs) -> None:
        """
        Mapper that transforms an input sequence to a hilbert space.

        Args:
            order (int) : Hilbert curve order.

            number_of_channels (int) : Number of output channels.

        """
        self.order = order
        self.number_of_channels = number_of_channels

        # Here we initialize the computed hilbert map
        self.hilbert_map = self.__class__.hilbert_curve(order=self.order)

        self._output_shape = (*self.hilbert_map.shape, self.number_of_channels)

    def __call__(self, sequence: List[int]) -> np.ndarray:
        """
        Call method that applies the sequence2hilbert method.

        Args:
            sequence (List[int]) : An input list of numbers to be mapped by
                                   the hilbert_map.

        Returns:
            (numpy.ndarray) : It returns the hilbert mapped sequence.

        """
        # Here we compute the mapped sequence using the computed hilbert map
        mapped_sequence = self.__class__.sequence2hilbert(
            sequence=sequence,
            hilbert_map=self.hilbert_map,
            number_of_channels=self.number_of_channels)

        # mapped_sequence = sequence2hilbert(sequence, **self.kwargs)

        return mapped_sequence

    @staticmethod
    def hilbert_curve(order: int) -> np.ndarray:
        """

        Compute the hilbert curve.

        Args:
            order (int) : Hilbert curve order.

        Returns:
            (numpy.ndarray) : It returns a numpy ndarray containing the
                              hilbert curve mapping of "n" order.

        """
        if order == 1:
            return np.zeros(shape=(1, 1), dtype=np.int32)

        t = HilbertMapper.hilbert_curve(order=order // 2)

        a = np.flipud(np.rot90(t))
        b = t + t.size
        c = t + t.size * 2
        d = np.flipud(np.rot90(t, -1)) + t.size * 3

        # TODO : It raises a Warning.
        # FutureWarning: arrays to stack must be passed as a "sequence" type
        # such as list or tuple. Support for non-sequence iterables such as
        # generators is deprecated as of NumPy 1.16 and will raise an error in
        # the future.
        hilbert_map = np.vstack(map(np.hstack, [[a, b], [d, c]]))

        return hilbert_map

    @staticmethod
    def sequence2hilbert(sequence: List[int], hilbert_map: np.ndarray,
                         number_of_channels: int) -> np.ndarray:
        """

        Transform an input sequence to a hilbert space.

        Args:
            sequence (List[int]) : An input list of numbers to be mapped by
                                   the hilbert_map.

            hilbert_map (numpy.ndarray) : Predefined hilbert curve mapping.

            number_of_channels (int) : Number of output channels.

        Returns:
            (numpy.ndarray) : It returns the sequence mapped by the given
                              hilbert map.

        """
        # TODO : It should recieve a sequence of embeddings instead of integers.

        output_shape = (*hilbert_map.shape, number_of_channels)

        mapped_sequence = np.zeros(shape=output_shape, dtype=np.int)

        for idx, row in enumerate(hilbert_map):

            for jdx, ii in enumerate(row):

                if len(sequence) <= ii:

                    continue

                element = sequence[ii]

                if element > number_of_channels:
                    # TODO : Perhaps we should raise a ValueError("Wrong ! . Got {} > {}".format(element, number_of_channels))
                    raise Exception("index > number of channels")

                mapped_sequence[idx, jdx, element] = 1

        return mapped_sequence

    @property
    def shape(self) -> Tuple[int, ...]:
        """

        Return the output shape.

        Returns:
            (Tuple[int, ...]) : Output shape.

        """
        return self._output_shape
