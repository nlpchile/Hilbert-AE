"""This module implement the utils classes and methods."""

import argparse
import os

import torch


def get_args():

    parser = argparse.ArgumentParser('Train Hilbert AutoEncoder')

    parser.add_argument('--device',
                        type=str,
                        default=None,
                        help="torch device as a string")

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


def load_from_checkpoint(model: torch.nn.Module, path_to_checkpoint: str):
    """
    Load from checkpoint.

    Args:
        model (torch.nn.Module) : A torch model.

        path_to_checkpoint (str) : A path to checkpoint files.

    Returns:
        (torch.nn.Module) : A torch model.

    """
    # TODO : We must first save model and optimizer states.
    # TODO : We must then load model and optimizer states from checkpoint.

    if path_to_checkpoint is not None and os.path.exists(path_to_checkpoint):
        model.load_state_dict(torch.load(path_to_checkpoint))

    # Still work in progress.
    raise NotImplementedError


def process_batch(batch: torch.Tensor):
    """
    Process a torch Tensor batch.

    Args:
        batch (torch.Tensor) : A torch Tensor.

    Returns:
        (torch.Tensor) : A processed torch Tensor.

    """
    # TODO : We must update this method if we update the dataloader outputs.
    x = torch.stack(batch).permute(dims=[0, 3, 1, 2]).float()

    return x


def create_folders(path: str = "./output/") -> None:
    # TODO : Document and extend this method.

    if not os.path.exists(path):
        os.mkdir(path)

    return
