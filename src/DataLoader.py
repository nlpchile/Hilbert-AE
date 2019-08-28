import h5py
import torch
import torch.utils.data


class Dataset_Hilbert(torch.utils.data.Dataset):
    """Hilbert Dataset Class."""

    def __init__(self, filename: str, **kwargs) -> None:
        """
        Hilbert Dataset.

        Args:
            filename (str) : A string containing the filename to the dataset.

        """
        super(Dataset_Hilbert, self).__init__()

        self.h5pyfile = h5py.File(filename, 'r')
        self.num_seq = self.h5pyfile['hilbert_map'].shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get one item at a time.

        Args:
            index (int) : An integer containing the index of
                          the item to retrieve.

        Returns:
            (torch.Tensor) :  It returns an item as a torch.Tensor.

        """
        seq_hilbert = torch.Tensor(
            self.h5pyfile['hilbert_map'][index, :, :, :]).type(
                dtype=torch.long)

        return seq_hilbert

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.num_seq


def DataLoader(filename: str, **kwargs) -> Dataset_Hilbert:
    """

    Return a Dataset_Hilbert given a filename.

    Args:
        filename (str) : A string containing the filename to the dataset.

    Returns:
        (Dataset_Hilbert) : A Dataset_Hilbert object.

    """
    return Dataset_Hilbert(filename=filename, **kwargs)


def contruct_dataloader_from_disk(filename: str, minibatch_size: int,
                                  **kwargs):
    """

    Build the dataloader from disk file.

    Args:
        filename (str):  A string containing the filename to the dataset.

    Returns:
        (torch.utils.data.Dataloader) : It returns a torch DataLoader.

    """
    return torch.utils.data.DataLoader(DataLoader(filename=filename),
                                       batch_size=minibatch_size,
                                       shuffle=True,
                                       **kwargs)
