from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Mapping, NoReturn

from torch.utils.data import DataLoader


class BaseScikitModel(ABC):
    @property
    @abstractmethod
    def data_type(self) -> str:
        pass

    @property
    @abstractmethod
    def run_type(self) -> str:
        pass

    @abstractmethod
    def fit(self, train_dataloader: DataLoader) -> NoReturn:
        pass

    @abstractmethod
    def test(self, test_dataloader: DataLoader) -> Mapping:
        pass

    @staticmethod
    @abstractmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        pass

    @classmethod
    @abstractmethod
    def from_argparse_args(cls, args, **kwargs):
        pass
