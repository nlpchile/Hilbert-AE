"""Training Stage."""
import os
from pathlib import Path
from typing import Dict, Union

import torch
from torch.utils.tensorboard import SummaryWriter

# TODO : Select the right tqdm for a jupyter notebook or a python console
from tqdm import tqdm

from src.AutoEncoder import (autoencoder, simple_autoencoder, training_step,
                             validation_step)
from src.dataloaders.dataloaders import build_dataloader_from_disk
from src.Meter import Accumulator
from src.utils import (create_folders, get_kwargs, load_from_checkpoint,
                       process_batch, save_checkpoint, set_config)

try:
    from apex import amp
    APEX_IS_AVAILABLE = True
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    opt_level = "O1"

except ImportError:
    APEX_IS_AVAILABLE = False
    pass

## HDF5 concurrent reads aren't safe
# Workarround still raises "TypeError: h5py objects cannot be pickled"
# try:
#     import torch.multiprocessing as mp
#     mp.set_start_method("spawn")
#     from src.dataloaders.dataloaders import build_dataloader_from_disk
# except RuntimeError:
#     from src.dataloaders.dataloaders import build_dataloader_from_disk
#     pass

# Install requirements :
#           pip3 install -r requirements.txt

# Install library in editable mode:
#           python3 setup.py develop

# Run this script
#           python3 train.py --config-file="./config.json"


def train(kwargs: Dict) -> None:
    """Train a Hilbert Autoencoder."""

    # Create tensorboard log folder if it doesn't exist
    path_to_tensorboard_logs = create_folders(
        path=kwargs["path_to_tensorboard_logs"], parents=True, exist_ok=True)

    # TODO : Replace this with a logger
    print("tensorboard --logdir={}".format(path_to_tensorboard_logs))

    # Tensorboard writer
    writer = SummaryWriter(log_dir=path_to_tensorboard_logs)

    # Config Device
    device = set_config(
        seed=kwargs["seed"],
        device=kwargs["device"],
        enforce_reproducibility=kwargs["enforce_reproducibility"])

    # Initialize Model
    # TODO : Define a get_models() method that outputs a dictionary

    # model = autoencoder(**kwargs["autoencoder"]).to(device)
    model = simple_autoencoder(**kwargs["simple_autoencoder"]).to(device)

    # TODO : Define a get_criterions() method that outputs a dictionary
    criterion = torch.nn.MSELoss(**kwargs["loss"])

    # TODO : Define a get_optimizers() method that outputs a dictionary
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 **kwargs["optimizer"])

    if APEX_IS_AVAILABLE:
        # https://nvidia.github.io/apex/amp.html#apex.amp.initialize
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=opt_level,
                                          enabled=True)

    # TODO : Define a get_dataloaders() method that outputs a dictionary
    train_loader, dev_loader = build_dataloader_from_disk(
        **kwargs["build_dataloader_from_disk"])

    ###############################################################################

    epochs = kwargs["epochs"]

    best_validation_loss = float("inf")
    best_epoch = -1

    training_loss_accumulator = Accumulator()
    validation_loss_accumulator = Accumulator()

    for epoch in tqdm(range(epochs), position=0, desc="epoch"):

        # Logging Model Weights
        # TODO : Add a "log_every" configurable parameter,
        # to avoid taking up all storage with the tensorboard logs.
        for module, module_name in zip([model.encoder, model.decoder],
                                       ["Encoder", "Decoder"]):

            for name, param in module.named_parameters():
                writer.add_histogram(tag=module_name + "/" + name,
                                     values=param,
                                     global_step=epoch,
                                     bins="tensorflow",
                                     walltime=None,
                                     max_bins=None)

        # We could save 1 batch specific batch in order to watch
        # the evolution of its latent representation over the
        # training epochs.

        # TRAINING
        training_loss_accumulator.reset()

        for idx, batch in enumerate(
                tqdm(train_loader, position=1, desc="train batch")):

            batch = batch.to(device)
            batch = process_batch(batch=batch)
            x = batch

            output = training_step(model=model,
                                   x=x,
                                   optimizer=optimizer,
                                   criterion=criterion)

            loss, output, latent = output["loss"], output["x_hat"], output["z"]

            training_loss_accumulator(value=loss)

        # Logging Training Loss
        writer.add_scalar(tag="train/loss",
                          scalar_value=training_loss_accumulator.avg,
                          global_step=epoch,
                          walltime=None)

        # VALIDATION
        validation_loss_accumulator.reset()

        for jdx, batch in enumerate(
                tqdm(dev_loader, position=2, desc="dev batch")):

            batch = batch.to(device)
            batch = process_batch(batch=batch)
            x = batch

            output = validation_step(model=model, x=x, criterion=criterion)

            loss, output, latent = output["loss"], output["x_hat"], output["z"]

            validation_loss_accumulator(value=loss)

        # Logging Validation Loss
        writer.add_scalar(tag="dev/loss",
                          scalar_value=validation_loss_accumulator.avg,
                          global_step=epoch,
                          walltime=None)

        # Saving Best Model according to the Validation Error
        # TODO : Support save checkpoints when using APEX : https://github.com/NVIDIA/apex#checkpointing
        if validation_loss_accumulator.avg < best_validation_loss:

            best_validation_loss = validation_loss_accumulator.avg
            best_epoch = epoch

            # suffix = "_epoch_{}".format(epoch)
            _ = save_checkpoint(
                model=model,
                optimizer=optimizer,
                path_to_checkpoints=kwargs["path_to_checkpoints"],
                prefix="",
                suffix="_best",
                extension=".pth")

    writer.close()


if __name__ == "__main__":

    kwargs = get_kwargs()

    train(kwargs=kwargs)
