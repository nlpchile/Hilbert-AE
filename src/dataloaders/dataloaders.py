"""This module implements the DataLoaders Classes."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from src.dataloaders.datasets import HilbertDataset, LanguageModelDataset
from src.dataloaders.PadCollate import PadCollate


# TODO : Check if having batch_first = False as default still makes sense.
class LanguageModelDataLoader():
    """Language Model Dataloader."""

    def __init__(self,
                 filename: str = "/text/data/raw/horoscopo_raw.txt",
                 separator: str = " ",
                 max_number_of_examples: int = -1,
                 batch_size: int = 32,
                 num_workers: int = torch.get_num_threads(),
                 shuffle: bool = True,
                 drop_last: bool = True,
                 batch_first: bool = False,
                 padding_value: int = -1,
                 dev_split: float = 0.1,
                 **kwargs) -> None:
        """

        Language Model Dataloader.

        Args:
            filename (str) : Path to the raw text.

            separator (str) : A string identifier used to split the string
                              into its respective tokens.

            max_number_of_examples (int) : Max number of examples to load. To
                                           load them all use
                                           max_number_of_examples=-1 .

            batch_size (int) : Number of items in a batch.

            num_workers (int) : Number of workers to be used by the DataLoader.

            shuffle (bool) : If True, it shuffles the dataset. Default = True.

            drop_last (bool) : If True, it drops the last batch if its size is
                               less than batch_size.

            batch_first (bool) : Default : False.

            padding_value (int) : Token index that represents the token used
                                  for padding.

            dev_split (float) : A positive float value in (0, 1) range that
                                represents the percentaje of the data used for
                                dev_set. Default = 0.1

        """
        # Language Model Dataset
        self.filename = filename
        self.separator = separator
        self.max_number_of_examples = max_number_of_examples

        # Language Model Data Loader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last

        # pad_collate
        self.batch_first = batch_first
        self.padding_value = padding_value

        # Random Split
        self.dev_split = dev_split

    def __dataset__(self) -> LanguageModelDataset:
        """Return a LanguageModelDatasets."""
        # Dataset
        dataset = LanguageModelDataset(
            filename=self.filename,
            separator=self.separator,
            max_number_of_examples=self.max_number_of_examples)

        return dataset

    def data_loader(self) -> Tuple[DataLoader, DataLoader]:
        """

        Get the dataloaders.

        Returns:
            Tuple[Dataloader, Dataloader]: A tuple containing the train and
                                           dev DataLoaders respectively.

        """
        dataset = self.__dataset__()

        # Random Split
        self.train_split = 1.0 - self.dev_split
        self.train_size = int(self.train_split * len(dataset))
        self.dev_size = len(dataset) - self.train_size

        # https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
        train_set, dev_set = random_split(
            dataset=dataset, lengths=[self.train_size, self.dev_size])

        # Data Loaders
        train_loader = DataLoader(train_set,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle,
                                  num_workers=self.num_workers,
                                  collate_fn=PadCollate(
                                      batch_first=self.batch_first,
                                      padding_value=self.padding_value),
                                  drop_last=self.drop_last)

        dev_loader = DataLoader(dev_set,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers,
                                collate_fn=PadCollate(
                                    batch_first=self.batch_first,
                                    padding_value=self.padding_value),
                                drop_last=self.drop_last)

        # num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None

        return train_loader, dev_loader

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.__dataset__())


def HilbertDataLoader(filename: str, **kwargs):
    """

    Return a HilbertDataset given a filename.

    Args:
        filename (str) : A string containing the filename to the dataset.

    Returns:
        (HilbertDataset) : A HilbertDataset object.

    """
    return HilbertDataset(filename=filename)


def build_dataloader_from_disk(filename: str,
                               batch_size: int,
                               shuffle: bool = True,
                               **kwargs) -> torch.utils.data.DataLoader:
    """

    Build the dataloader from disk file.

    Args:
        filename (str):  A string containing the filename to the dataset.

        batch_size (int) : Number of items in a batch.

         shuffle (bool) : If True, it shuffles the dataset. Default = True.

    Returns:
        (torch.utils.data.Dataloader) : It returns a torch DataLoader.

    """
    dataloader = torch.utils.data.DataLoader(
        HilbertDataLoader(filename=filename),
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader
