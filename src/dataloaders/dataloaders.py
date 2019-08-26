import torch
from torch.utils.data import DataLoader, random_split

from src.dataloaders.datasets import LanguageModelDataset
from src.dataloaders.PadCollate import PadCollate


class LanguageModelDataLoader():
    def __init__(
            self,
            filename: str = '/text/data/raw/horoscopo_raw.txt',
            separator: str = ' ',
            start_token: str = '<SOS>',
            end_token: str = '<EOS>',
            batch_size: int = 32,
            num_workers=torch.get_num_threads(),  # TODO : add type hint
            shuffle: bool = True,
            drop_last: bool = True,
            batch_first: bool = False,
            padding_value: int = -1,
            dev_spli: float = 0.1) -> None:

        # Language Model Dataset
        self.filename = filename
        self.separator = separator
        self.start_token = start_token
        self.end_token = end_token

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

    def __dataset__(self):  # TODO : Add type hint

        # Dataset
        dataset = LanguageModelDataset(filename=self.filename,
                                       separator=self.separator,
                                       start_token=self.start_token,
                                       end_token=self.end_token)
        return dataset

    def data_loader(self):  # TODO : Add type hint

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
