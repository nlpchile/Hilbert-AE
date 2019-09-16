"""This module implement the utils classes and methods."""

import argparse
import json
import os
import random
import typing
from pathlib import Path
from typing import Dict, List, Tuple, Union

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
                    path_to_checkpoints: str = "./checkpoints/",
                    prefix: str = "",
                    suffix: str = "",
                    extension: str = ".pth") -> Dict[str, str]:
    """
    Save Model and Optimizer checkpoint.

    Args:
        model (torch.nn.Module) : A pytorch model.

        optimizer (torch.optim) : A pytor optimizer.

        path_to_checkpoints (str) : A path to save the checkpoints.
            Default = "./checkpoints/"

        prefix (str) : A prefix to all checkpoint filenames.
            Default = ""

        suffix (str) : A suffix to all checkpoint filenames.
            Default = ""

        extension (str) : The checkpoint filename extension.
            Default = ".pth"

    Returns:
        (Dict[str, str]) : A dictionary containing the paths to the checkpoints.

                paths = {
                    "model": str(path_to_models),
                    "optimizer": str(path_to_optimizers)
                }

    """
    # TODO : Extend this method to multiple models and optimizers.
    # TODO : Decide where to create checkpoint folder if it doesn't exist.

    # Models paths
    path_to_models = Path(path_to_checkpoints) / "{}".format(prefix + "model" +
                                                             suffix +
                                                             extension)

    # Optimizers paths
    path_to_optimizers = Path(path_to_checkpoints) / "{}".format(
        prefix + "optimizer" + suffix + extension)

    # Save Models
    torch.save(model.state_dict(), str(path_to_models))

    # Save Optimizers
    torch.save(optimizer.state_dict(), str(path_to_optimizers))

    paths = {
        "model": str(path_to_models),
        "optimizer": str(path_to_optimizers)
    }

    return paths


def load_from_checkpoint(model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         device: torch.device, paths: Dict[str, str]):
    """
    Load from checkpoint.

    Args:
        model (torch.nn.Module) : A torch model.

        optimizer (torch.optim.Optimizer) : A torch optimizer

        device (torch.device) : A torch device

        paths (Dict[str, str]) : A dictionary containing the paths to checkpoint
            files.

            paths = {
                    "model": str(path_to_models),
                    "optimizer": str(path_to_optimizers)
                }

    Returns:
        (torch.nn.Module, torch.optim.Optimizer) : A tuple containing a torch model
            and a torch optimizer.

    """

    model.load_state_dict(torch.load(paths["model"], map_location=device))

    optimizer.load_state_dict(
        torch.load(paths["optimizer"], map_location=device))

    checkpoint = {"model": model, "optimizer": optimizer}

    return checkpoint


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

    return x


def create_folders(path: str = "./output/",
                   parents: bool = False,
                   exist_ok: bool = False) -> str:
    """
    Create folder.

    https://docs.python.org/3.7/library/pathlib.html#pathlib.Path.mkdir

    Args :
        path (str): Path to the folder that's being created.

    """

    absolute_path_to_folder = Path(path).absolute()

    absolute_path_to_folder.mkdir(parents=parents, exist_ok=exist_ok)

    return str(absolute_path_to_folder)


def save_as_binary_dataset(*args, **kwargs):
    """Save a dataset object as binary file."""
    raise NotImplementedError


def load_dataset(path: str, *args, **kwargs):
    """Load a dataset object from path."""
    # Maybe we can infer if binary or raw by looking at the file extension.
    raise NotImplementedError
