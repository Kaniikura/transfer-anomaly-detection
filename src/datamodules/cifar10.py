
# modified from [1]
# [1]: 'https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/datamodules/cifar10_datamodule.py'
from argparse import SUPPRESS, ArgumentParser
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import argparse_utils
from torch.utils.data import DataLoader

from ..datasets import CIFAR10

ROOT_PATH = (Path(__file__).parent.parent.parent / 'data').resolve()

CLASSES = {
    'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
}


class CIFAR10DataModule(LightningDataModule):
    name = "cifar10"

    def __init__(
        self,
        one_class: str,
        root_path: Optional[Union[Path, str]] = ROOT_PATH,
        img_size: int = 32,
        num_workers: int = 16,
        batch_size: int = 32,
        seed: int = 42,
        *args: Any,
        **kwargs: Any,
    ):
        assert one_class in CLASSES.keys(), f'class name: {one_class}, should be in {*CLASSES.keys(),}'
        self.normal_class_id = CLASSES[one_class]

        self.root_path = root_path
        self.img_size = img_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed

        self.train_transforms = self.default_transforms(is_train=True)
        self.test_transforms = self.default_transforms(is_train=False)

        self.prepare_data()

    def prepare_data(self):
        dataset = {}
        dataset['train'] = CIFAR10(root=self.root_path, train=True, download=True,
                                   transform=self.train_transforms)
        dataset['test'] = CIFAR10(root=self.root_path, train=False, download=True,
                                  transform=self.test_transforms)

        dataset['train'].data, dataset['train'].targets, \
            dataset['test'].data, dataset['test'].targets = self.get_cifar_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            nrm_cls_idx=self.normal_class_id,
        )

        self.train_dataset = dataset['train']
        self.test_dataset = dataset['test']

    def train_dataloader(self):

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        return loader

    def default_transforms(self, is_train: bool) -> Callable:
        if is_train:
            augs = [
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        else:
            augs = []

        if self.img_size:  # use pre-specified image size
            trfm = transforms.Compose([
                transforms.Resize(self.img_size),
                *augs,
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:  # use default image size
            trfm = transforms.Compose([
                *augs,
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        return trfm

    def add_argparse_args(parent_parser: ArgumentParser,) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False, allow_abbrev=False)
        parser.add_argument('--one_class', type=str, default=SUPPRESS,
                            help="name of the class to be treated as normal out of 10 CIFAR-10 classes")

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return argparse_utils.from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def get_cifar_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, nrm_cls_idx=0, manualseed=-1):
        # Changed 'https://github.com/samet-akcay/ganomaly/blob/master/lib/data.py' to one vs all protocol
        """[summary]
        Arguments:
            trn_img {np.array} -- Training images
            trn_lbl {np.array} -- Training labels
            tst_img {np.array} -- Test     images
            tst_lbl {np.array} -- Test     labels
        Keyword Arguments:
            nrm_cls_idx {int} -- Normal class index (default: {0})
        Returns:
            [np.array] -- New training-test images and labels.
        """
        # Convert train-test labels into numpy array.
        trn_lbl = np.array(trn_lbl)
        tst_lbl = np.array(tst_lbl)

        # --
        # Find idx, img, lbl for abnormal and normal on org dataset.
        nrm_trn_idx = np.where(trn_lbl == nrm_cls_idx)[0]
        abn_trn_idx = np.where(trn_lbl != nrm_cls_idx)[0]
        nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
        abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
        nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
        abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

        nrm_tst_idx = np.where(tst_lbl == nrm_cls_idx)[0]
        abn_tst_idx = np.where(tst_lbl != nrm_cls_idx)[0]
        nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
        abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
        nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
        abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

        # --
        # Assign labels to normal (0) and abnormals (1)
        nrm_trn_lbl[:] = 0
        nrm_tst_lbl[:] = 0
        abn_trn_lbl[:] = 1
        abn_tst_lbl[:] = 1

        new_trn_img = np.copy(nrm_trn_img)
        new_trn_lbl = np.copy(nrm_trn_lbl)
        new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
        new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

        return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl
