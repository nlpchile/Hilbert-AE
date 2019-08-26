from typing import List

import numpy as np


def hilbert_curve(order: int) -> np.ndarray:
    '''

    Method to compute the hilbert curve.

    Args:

        order (int) : Hilbert curve order.

    Returns:

        (numpy.ndarray) : It returns a numpy ndarray containing
                          the hilbert curve mapping of "n" order.

    '''

    if order == 1:
        return np.zeros((1, 1), np.int32)

    t = hilbert_curve(order // 2)

    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size * 2
    d = np.flipud(np.rot90(t, -1)) + t.size * 3

    hilbert_map = np.vstack(map(np.hstack, [[a, b], [d, c]]))

    return hilbert_map


def sequence2hilbert(sequence: List[int],
                     hilbert_map: np.ndarray,
                     number_of_channels: int = 20) -> np.ndarray:
    '''

    Helper method to transform an input sequence to a hilbert space.

    Args:

        sequence (list) : An input list of numbers to be mapped by
                          the hilbert_map.

        hilbert_map (numpy.ndarray) : Predefined hilbert curve mapping.

        number_of_channels (int) : Number of output channels.

    Returns:

        (numpy.ndarray) : It returns the sequence mapped by the
                          given hilbert map.

    '''

    # TODO : It should recieve a sequence of embeddings instead of integers.

    shape = [dim for dim in hilbert_map.shape] + [number_of_channels]

    mapped_sequence = np.zeros(shape=shape, dtype=np.int)

    for idx, row in enumerate(hilbert_map):

        for jdx, ii in enumerate(row):

            if len(sequence) <= ii:

                continue

            element = sequence[ii]

            if element > number_of_channels:
                raise Exception("index > number of channels")

            mapped_sequence[idx, jdx, element] = 1

    return mapped_sequence
