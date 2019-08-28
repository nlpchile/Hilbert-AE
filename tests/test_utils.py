import numpy as np
import pytest

from src.utils import HilbertMapper


def test_01_hilbert_curve():

    order = 2
    hilbert_map = HilbertMapper.hilbert_curve(order=order)

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

    expected_shape = (*hilbert_map.shape, number_of_channels)

    mapped_sequence = HilbertMapper.sequence2hilbert(
        sequence=sequence,
        hilbert_map=hilbert_map,
        number_of_channels=number_of_channels)

    assert isinstance(mapped_sequence, np.ndarray)
    assert mapped_sequence.shape == expected_shape


def test_03_hilbert_curve_and_sequence_2_hilbert():

    # HILBERT - MAP

    #  0,  3,  4,  5
    #  1,  2,  7,  6
    # 14, 13,  8,  9
    # 15, 12, 11, 10
    #                   0 1 2 3 4 5 6 7 8 9 10 11

    order = 4
    number_of_channels = 2

    sequence = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

    expected_mapped_sequence = np.array(
        [[[1, 0], [1, 0], [0, 1], [0, 1]], [[1, 0], [1, 0], [0, 1], [0, 1]],
         [[0, 0], [0, 0], [1, 0], [1, 0]], [[0, 0], [0, 0], [1, 0], [1, 0]]],
        dtype=np.int)

    hilbert_map = HilbertMapper.hilbert_curve(order=order)

    mapped_sequence = HilbertMapper.sequence2hilbert(
        sequence=sequence,
        hilbert_map=hilbert_map,
        number_of_channels=number_of_channels)

    assert (mapped_sequence == expected_mapped_sequence).all()


def test_04_hilbert_mapper():

    order = 4
    number_of_channels = 2

    sequence = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

    # We initialize our mapper object
    mapper = HilbertMapper(order=order, number_of_channels=number_of_channels)

    # We invoke its "__call__" method
    mapped_sequence = mapper(sequence=sequence)

    expected_mapped_sequence = np.array(
        [[[1, 0], [1, 0], [0, 1], [0, 1]], [[1, 0], [1, 0], [0, 1], [0, 1]],
         [[0, 0], [0, 0], [1, 0], [1, 0]], [[0, 0], [0, 0], [1, 0], [1, 0]]],
        dtype=np.int)

    assert (mapped_sequence == expected_mapped_sequence).all()