"""This module implement the utils classes and methods."""

import argparse
import json
import os
import random
import typing
from typing import List, Tuple, Union

import numpy as np
import torch

Tensor = torch.Tensor


def get_kwargs():

    parser = argparse.ArgumentParser(description="Path to Config File")

    parser.add_argument("--config-file",
                        default="./config.json",
                        type=str,
                        help="path to JSON file with configs")

    args = parser.parse_args()

    kwargs = {}

    with open(args.config_file, "r") as json_file:
        kwargs = json.load(json_file)

    return kwargs


def set_config(seed: int, device: torch.device,
               enforce_reproducibility: bool) -> torch.device:

    # Use the GPU if possible.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
                          ) if device is None else device

    # Use CuDNN optimizations if possible,
    # At the expense of potentially having an impact on reproducibility.
    # https://pytorch.org/docs/stable/notes/randomness.html

    if torch.cuda.is_available():

        if enforce_reproducibility:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if not enforce_reproducibility:
            # torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    # I may be confused, but I think PyTorch by default uses half of the available threads,
    # So, if that is True, then this should make it possible to use all available threads.
    torch.set_num_threads(2 * os.cpu_count())

    # Set the seeds for reproducibility

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed=seed)

    # TODO : Replace prints with a logger
    print(" PyTorch Version      : {} ".format(torch.__version__))
    print(" Device               : {} ".format(device))
    print(" Number of Threads    : {} ".format(torch.get_num_threads()))
    print(" Seed                 : {} ".format(seed))

    return device


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    path: str = "./checkpoints/") -> str:
    """
    Save Model and Optimizer checkpoint.

    Args:
        model (torch.nn.Module) : A pytorch model.
        optimizer (torch.optim) : A pytor optimizer.
        path (str) : A path to save the checkpoints.

    Returns:
        () :

    """
    # TODO : Perhaps we need some identifiers or we could automatically create a folder by experiment.
    raise NotImplementedError


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


def process_batch(batch: Union[Tuple[Tensor, ...], List[Tensor]]):
    """
    Process a torch Tensor batch.

    Args:
        batch (Tensor) : A torch Tensor.

    Returns:
        (Tensor) : A processed torch Tensor.

    """

    # TODO : Uncomment this
    # batch = tuple(element for element in batch)

    # TODO : We must update this method if we update the dataloader outputs.
    x = torch.stack(tensors=batch, dim=0).permute(dims=[0, 3, 1, 2]).float()
    # x = batch.permute(dims=[0, 3, 1, 2]).float()

    return x


def create_folders(path: str = "./output/") -> None:
    """
    Create folder.

    Args :
        path (str): Path to the folder that's being created.

    """
    # TODO : Document and extend this method.
    # Try using pathlib.Path and its mkdir method.

    if not os.path.exists(path):
        os.mkdir(path)

    return


def save_as_binary_dataset(*args, **kwargs):
    """Save a dataset object as binary file."""
    raise NotImplementedError


def load_dataset(path: str, *args, **kwargs):
    """Load a dataset object from path."""
    # Maybe we can infer if binary or raw by looking at the file extension.
    raise NotImplementedError
