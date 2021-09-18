from argparse import SUPPRESS, ArgumentParser
from os import PathLike
from pathlib import Path
from typing import Optional

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import argparse_utils
from torch.utils.data import DataLoader

from ..datasets import ShanghaiTechDataset

ROOT_PATH = (Path(__file__).parent.parent.parent / 'data').resolve()


class ShanghaiTechDataModule(LightningDataModule):
    name = 'shanghaitech'

    def __init__(
            self,
            root_path: Optional[PathLike] = ROOT_PATH,
            img_size: Optional[int] = 224,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            without_background: bool = True,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.root_path = Path(root_path)
        self.shanghai_folder_path = Path(root_path, 'shanghaitech')
        self.img_size = img_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.without_background = without_background

        self.train_transforms = self.default_transforms()
        self.test_transforms = self.default_transforms()

        self.prepare_data()

    def prepare_data(self):
        """Download dataset if not exist"""
        # TODO
        if not Path(self.shanghai_folder_path / 'training').exists():
            raise FileNotFoundError("You need to download ShanghaiTech and unpack into 'data' folder.")

    def train_dataloader(self):
        dataset = ShanghaiTechDataset(
            root_path=self.shanghai_folder_path, is_train=True,
            without_background=self.without_background, transforms=self.train_transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):
        dataset = ShanghaiTechDataset(
            root_path=self.shanghai_folder_path, is_train=False,
            without_background=self.without_background, transforms=self.test_transforms)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        return loader

    def default_transforms(self):
        if self.img_size:  # use pre-specified image size
            transforms = Compose([
                Resize(self.img_size, self.img_size),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:  # use default image size
            transforms = Compose([
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        return transforms

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser,) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False, allow_abbrev=False)
        parser.add_argument('--without_background', type=lambda s: s.lower() == 'true', default=SUPPRESS,
                            help="remove the background with a MOG-based approach")

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return argparse_utils.from_argparse_args(cls, args, **kwargs)
