"""Training Stage."""
import os

import torch

from src.AutoEncoder import autoencoder
from src.DataLoader import contruct_dataloader_from_disk
from src.utils import create_folders, get_args


def train(args):

    # if args.device is None: (?)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input channels
    nc = args.nc

    # Output channels
    ndf = args.ld

    model = autoencoder(nc=nc, ndf=ndf).to(device)

    # path_to_checkpoint
    checkpoint = args.checkpoint

    if checkpoint is not None and os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint))

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_loader = contruct_dataloader_from_disk(args.hdf5_file,
                                                 args.batch_size)

    num_epochs = args.epochs

    early_stop_limit = args.early_stop

    early_stop_count = 0

    train_loss = []

    create_folders()

    best_path = "./output/HILBERT_AE_best.pth"

    for epoch in range(num_epochs):

        loss_train = 0

        for idx, batch in enumerate(train_loader):

            hilbert_map = batch

            hilbert_map = torch.stack(hilbert_map).permute(0, 3, 1, 2).type(
                torch.FloatTensor)

            hilbert_map = hilbert_map.to('cuda:1')

            # ===================forward=====================

            output, latent = model(hilbert_map)

            loss = criterion(output, hilbert_map)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs,
                                                  loss_train.item() / idx))

        train_loss.append(loss_train.item() / idx)

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
