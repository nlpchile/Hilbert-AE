from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from src.dataloaders.datasets import LanguageModelDataset, HilbertDataset
from src.dataloaders.PadCollate import PadCollate


class LanguageModelDataLoader():
    def __init__(
            self,
            filename: str = "/text/data/raw/horoscopo_raw.txt",
            separator: str = " ",
            max_examples: int = -1,
            #start_token: str = '<SOS>',
            #end_token: str = '<EOS>',
            batch_size: int = 32,
            num_workers: int = torch.get_num_threads(),
            shuffle: bool = True,
            drop_last: bool = True,
            batch_first: bool = False,
            padding_value: int = -1,
            dev_split: float = 0.1) -> None:

        # Language Model Dataset
        self.filename = filename
        self.separator = separator
        self.max_examples = max_examples
        #self.start_token = start_token
        #self.end_token = end_token

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

        # Dataset
        dataset = LanguageModelDataset(
            filename=self.filename,
            separator=self.separator,
            max_examples=self.max_examples
            #start_token=self.start_token,
            #end_token=self.end_token
        )

        return dataset

    def data_loader(self) -> Tuple[DataLoader, DataLoader]:

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
        return len(self.__dataset__())


def HilbertDataLoader(filename: str):
    return HilbertDataset(filename=filename)


def build_dataloader_from_disk(filename: str,
                               minibatch_size: int,
                               shuffle: bool = True
                               ) -> torch.utils.data.DataLoader:

    dataloader = torch.utils.data.DataLoader(
        HilbertDataLoader(filename=filename),
        batch_size=minibatch_size,
        shuffle=shuffle,
    )

    return dataloader