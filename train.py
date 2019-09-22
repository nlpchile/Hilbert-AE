"""Training Stage."""
import os
from pathlib import Path
from typing import Dict, Union

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.AutoEncoder import autoencoder, training_step
from src.dataloaders.dataloaders import build_dataloader_from_disk
from src.Meter import Accumulator
from src.utils import (create_folders, get_kwargs, load_from_checkpoint,
                       process_batch, set_config)

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
    model = autoencoder(**kwargs["autoencoder"]).to(device)

    # TODO : Define a get_criterions() method that outputs a dictionary
    criterion = torch.nn.MSELoss(**kwargs["loss"])

    # TODO : Define a get_optimizers() method that outputs a dictionary
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 **kwargs["optimizer"])

    # TODO : Define a get_dataloaders() method that outputs a dictionary
    train_loader = build_dataloader_from_disk(
        **kwargs["build_dataloader_from_disk"])

    ###############################################################################

    epochs = kwargs["epochs"]

    iteration = 0

    training_loss_acumulator = Accumulator()

    for epoch in tqdm(range(epochs), position=0, desc="epoch"):

        training_loss_acumulator.reset()

        # We could save 1 batch specific batch in order to watch
        # the evolution of its latent representation over the
        # training epochs.

        for idx, batch in enumerate(
                tqdm(train_loader, position=1, desc="batch")):

            batch = batch.to(device)

            batch = process_batch(batch=batch)

            x = batch

            output = training_step(model=model,
                                   x=x,
                                   optimizer=optimizer,
                                   criterion=criterion)

            loss, output, latent = output["loss"], output["x_hat"], output["z"]

            training_loss_acumulator(value=loss)

            # # Logging Loss
            # writer.add_scalar(tag="train/loss",
            #                   scalar_value=loss,
            #                   global_step=iteration,
            #                   walltime=None)

            iteration += 1

        # Logging Loss
        writer.add_scalar(tag="train/loss",
                          scalar_value=training_loss_acumulator.avg,
                          global_step=epoch,
                          walltime=None)

    writer.close()


if __name__ == "__main__":

    kwargs = get_kwargs()

    train(kwargs=kwargs)
