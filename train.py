"""Training Stage."""
import os
from typing import Dict, Union

import torch

from src.AutoEncoder import autoencoder, training_step
from src.dataloaders.dataloaders import build_dataloader_from_disk
from src.utils import create_folders, get_args, process_batch


def train(args):
    """Train a Hilbert Autoencoder."""
    # Config Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
                          ) if args.device is None else args.device

    ###########################################################################
    # Config JSONs
    kwargs: Dict[str, Union[str, Dict[str, str]]] = {}

    kwargs["build_dataloader_from_disk"] = {
        "filename": args.hdf5_file,
        "batch_size": args.batch_size,
        "shuffle": True  # TODO : Add it to args
    }

    kwargs["autoencoder"] = {"nc": args.nc, "ndf": args.ld}

    kwargs["optimizer"] = {
        "lr": args.lr,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": args.weight_decay,
        "amsgrad": False
    }

    kwargs["loss"] = {
        "size_average": None,
        "reduce": None,
        "reduction": "mean"
    }

    ###########################################################################

    # Initialize Model
    model = autoencoder(**kwargs["autoencoder"]).to(device)

    # TODO : Defined as "load_from_checkpoint" in utils.
    path_to_checkpoint = args.path_to_checkpoint
    if path_to_checkpoint is not None and os.path.exists(path_to_checkpoint):
        model.load_state_dict(torch.load(path_to_checkpoint))

    criterion = torch.nn.MSELoss(**kwargs["loss"])

    optimizer = torch.optim.Adam(model.parameters(), **kwargs["optimizer"])

    train_loader = build_dataloader_from_disk(
        **kwargs["build_dataloader_from_disk"])

    epochs = args.epochs

    early_stop_limit = args.early_stop

    early_stop_count = 0

    train_loss = []

    create_folders()

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

            loss_train += loss

        # ===================log========================
        # TODO : Add a tensorboard logger.
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs,
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

    print("AutoEncoder was trained !!")


if __name__ == '__main__':

    args = get_args()

    train(args)
