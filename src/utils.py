"""This module implement the utils classes and methods."""

import argparse
import os


def create_folders(path: str = "./output/") -> None:

    if not os.path.exists(path):
        os.mkdir(path)

    return


def get_args():
    parser = argparse.ArgumentParser('Train Hilbert AutoEncoder')
    parser.add_argument('--hdf5_file', type=str, help='Path to HDF5 file')
    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='Path to Checkpoint Model')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs')
    parser.add_argument('--early_stop',
                        type=int,
                        default=40,
                        help='Early stop limit')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-5,
                        help='weight decay to optimizer')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size')
    parser.add_argument('--nc',
                        type=int,
                        default=1,
                        help='Number of channels in data')
    parser.add_argument('--ld',
                        type=int,
                        default=256,
                        help='latent dimension size')

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    return args
