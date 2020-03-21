from pathlib import Path

import numpy as np
import pytest

from src.HilbertMapper import HilbertMapper
from src.utils import create_folders


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

    assert hasattr(HilbertMapper, "sequence2hilbert")

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

    assert hasattr(HilbertMapper, "hilbert_curve")

    hilbert_map = HilbertMapper.hilbert_curve(order=order)

    assert hasattr(HilbertMapper, "sequence2hilbert")

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

    assert callable(mapper)

    # We invoke its "__call__" method
    mapped_sequence = mapper(sequence=sequence)

    expected_mapped_sequence = np.array(
        [[[1, 0], [1, 0], [0, 1], [0, 1]], [[1, 0], [1, 0], [0, 1], [0, 1]],
         [[0, 0], [0, 0], [1, 0], [1, 0]], [[0, 0], [0, 0], [1, 0], [1, 0]]],
        dtype=np.int)

    assert (mapped_sequence == expected_mapped_sequence).all()

    @pytest.skip(msg="WIP")
    def test_05_create_folders():

        path_to_folder = "./test_05/foo/boo/"
        parents = True
        exist_ok = True

        absolute_path_to_folder = create_folders(path=path_to_folder,
                                                 parents=parents,
                                                 exist_ok=exist_ok)

        assert isinstance(absolute_path_to_folder, str)

        path = Path(absolute_path_to_folder)

        assert path.is_dir()
