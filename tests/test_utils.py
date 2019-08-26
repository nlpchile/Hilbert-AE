import numpy as np
import pytest

from src.utils import hilbert_curve, sequence2hilbert


def test_01_hilbert_curve():

    order = 2
    hilbert_map = hilbert_curve(order=order)

    expected_hilbert_map = np.array([[0, 1], [3, 2]], dtype=np.int)
    expected_shape = (2, 2)

    assert isinstance(hilbert_map, np.ndarray)
    assert hilbert_map.shape == expected_shape
    assert (hilbert_map == expected_hilbert_map).all()


def test_02_sequence2hilbert():

    sequence = [4, 1, 2, 3, 2, 5]

    number_of_channels = 5

    # hilbert_map of order 2
    hilbert_map = np.array([[0, 1], [3, 2]], dtype=np.int)

    expected_shape = [dim for dim in hilbert_map.shape] + [number_of_channels]

    mapped_sequence = sequence2hilbert(sequence=sequence,
                                       hilbert_map=hilbert_map,
                                       number_of_channels=number_of_channels)

    assert isinstance(mapped_sequence, np.ndarray)
    assert list(mapped_sequence.shape) == expected_shape


def test_03_hilbert_curve_and_sequence_2_hilbert():

    # HILBERT - MAP

    #  0,  3,  4,  5
    #  1,  2,  7,  6
    # 14, 13,  8,  9
    # 15, 12, 11, 10
    #                   0 1 2 3 4 5 6 7 8 9 10 11

    order = 4
    hilbert_map = hilbert_curve(order=order)

    sequence = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    number_of_channels = 2

    mapped_sequence = sequence2hilbert(sequence=sequence,
                                       hilbert_map=hilbert_map,
                                       number_of_channels=number_of_channels)

    expected_mapped_sequence = np.array(
        [[[1, 0], [1, 0], [0, 1], [0, 1]], [[1, 0], [1, 0], [0, 1], [0, 1]],
         [[0, 0], [0, 0], [1, 0], [1, 0]], [[0, 0], [0, 0], [1, 0], [1, 0]]],
        dtype=np.int)

    assert (mapped_sequence == expected_mapped_sequence).all()
