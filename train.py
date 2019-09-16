"""Training Stage."""
import os
from pathlib import Path
from typing import Dict, Union

import torch
from torch.utils.tensorboard import SummaryWriter

from src.AutoEncoder import autoencoder, training_step
from src.dataloaders.dataloaders import build_dataloader_from_disk
from src.utils import (create_folders, get_kwargs, load_from_checkpoint,
                       process_batch, set_config)

# Just run
#           python3 train.py --config-file="./config.json"


def train(kwargs):
    """Train a Hilbert Autoencoder."""

    # Create tensorboard log folder if it doesn't exist
    path_to_tensorboard_logs = create_folders(
        path=kwargs["path_to_tensorboard_logs"], parents=True, exist_ok=True)

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

    # TODO : Add this in config.
    #
    # checkpoint_paths = {
    #     "model": path_to_model_checkpoint,
    #     "optimizer": path_to_optimizer_checkpoint
    # }
    #
    # TODO : Replace with this.
    # checkpoint = load_from_checkpoint(model=model,
    #                                   optimizer=optimizer,
    #                                   device=device,
    #                                   paths=paths)

    path_to_checkpoint = Path(kwargs["path_to_checkpoints"])
    if path_to_checkpoint is not None and path_to_checkpoint.exists():
        model.load_state_dict(torch.load(str(path_to_checkpoint)))

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

    early_stop_limit = kwargs["early_stop"]

    early_stop_count = 0

    train_loss = []

    create_folders()

    # TODO : Add this to config json
    best_path = "./output/HILBERT_AE_best.pth"

    for epoch in range(epochs):

        loss_train = 0

        for idx, batch in enumerate(train_loader):

            batch = batch.to(device)

            # TODO : We must update this method if we update the dataloader outputs.
            batch = process_batch(batch=batch)

            x = batch

            output = training_step(model=model,
                                   x=x,
                                   optimizer=optimizer,
                                   criterion=criterion)

            loss, output, latent = output["loss"], output["x_hat"], output["z"]

            # TODO : Add a tensorboard logger

            loss_train += loss

        # ===================log========================
        # TODO : Add a tensorboard logger.
        print("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, epochs,
                                                  loss_train.item() / idx))
        train_loss.append(loss_train.item() / idx)

        # TODO : We must also save the optimizer states.
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       "./output/HILBERT_AE_{}.pth".format(epoch))
        if len(train_loss) > 2 and train_loss[-1] == min(train_loss):
            torch.save(model.state_dict(), best_path)
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > early_stop_limit:
            break

    writer.close()
    print("AutoEncoder was trained !!")


if __name__ == "__main__":

    kwargs = get_kwargs()

    train(kwargs=kwargs)
