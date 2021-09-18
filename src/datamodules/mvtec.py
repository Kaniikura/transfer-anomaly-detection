import tarfile
import urllib
from argparse import SUPPRESS, ArgumentParser
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from PIL import Image
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import argparse_utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets import MVTecDataset
from ..transforms import mvtec_get_transforms

URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
ROOT_PATH = (Path(__file__).parent.parent.parent / 'data').resolve()
NUM_IMAGES = 3629 + 1725  # train + test


class MVTecDataModule(LightningDataModule):
    name = 'mvtec_ad'
    extra_args = {}

    def __init__(
            self,
            class_name: str,
            root_path: Optional[PathLike] = ROOT_PATH,
            img_size: int = 224,
            trfm_params: Dict[str, Any] = None,
            only_noise: bool = False,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            img_format: str = 'png',
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert class_name in CLASS_NAMES, f'class_name: {class_name}, should be in {CLASS_NAMES}'
        self.class_name = class_name
        self.root_path = Path(root_path)
        self.mvtec_folder_path = self.root_path / 'mvtec_ad'
        self.mvtec_folder_path.mkdir(exist_ok=True)
        self.img_size = img_size
        self.trfm_params = trfm_params or self._default_trfm_params(class_name=class_name, only_noise=only_noise)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.img_format = img_format

        self.train_transforms = mvtec_get_transforms(resize=self.img_size, **self.trfm_params)
        self.test_transforms = mvtec_get_transforms(resize=self.img_size, **self.trfm_params, is_train=False)

        self.prepare_data()
        if self.img_format == 'jpg':
            self.data_dir = self.root_path / 'mvtec_ad_jpeg'
            self.data_dir.mkdir(exist_ok=True)
            self.ensure_jpeg_data()
        else:
            self.data_dir = self.mvtec_folder_path

    def prepare_data(self):
        """Download dataset if not exist"""

        if not Path(self.mvtec_folder_path, self.class_name).exists():
            tar_file_path = self.mvtec_folder_path / '.tar.xz'
            if not tar_file_path.exists():
                self._download_url(URL, tar_file_path)
            print('unzip downloaded dataset: %s' % tar_file_path)
            tar = tarfile.open(tar_file_path, 'r:xz')
            tar.extractall(self.mvtec_folder_path)
            tar.close()

        else:
            print("dataset already downloaded. Did not download twice.")

    def ensure_jpeg_data(self):
        # HACK: need refactoring
        with tqdm(total=NUM_IMAGES, desc='[Create JPEG images]') as pbar:
            if len(list(self.data_dir.glob('*'))) != 15:
                for node in self.mvtec_folder_path.glob('*'):
                    if node.is_dir():
                        for phase in ['train', 'test']:
                            jpg_dir = self.data_dir / node.name / phase
                            jpg_dir.mkdir(exist_ok=True, parents=True)
                            img_types = sorted([x.name for x in (node / phase).iterdir()])
                            for img_type in img_types:
                                save_dir = jpg_dir / img_type
                                save_dir.mkdir(exist_ok=True)
                                for png_path in (node / phase / img_type).glob('*.png'):
                                    jpg_file_path = save_dir / f'{png_path.stem}.jpg'
                                    if not jpg_file_path.exists():
                                        image = Image.open(png_path)
                                        rgb_im = image.convert('RGB')
                                        rgb_im.save(str(jpg_file_path), 'JPEG', quality=90)
                                    pbar.update(1)

    def train_dataloader(self):
        dataset = MVTecDataset(
            class_name=self.class_name, root_path=self.data_dir,
            transforms=self.train_transforms, is_train=True,
            format=self.img_format, **self.extra_args)
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
        dataset = MVTecDataset(
            class_name=self.class_name, root_path=self.data_dir,
            transforms=self.test_transforms, is_train=False,
            format=self.img_format, **self.extra_args)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        return loader

    def _default_trfm_params(self, class_name: str, only_noise: bool = False):
        with open(Path(__file__).parent / 'aug_trfm.yaml') as f:
            trfm_params = yaml.safe_load(f)[class_name]
        trfm_params['only_noise'] = only_noise

        return trfm_params

    @staticmethod
    def _download_url(url, output_path):
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser,) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False, allow_abbrev=False)
        parser.add_argument('--class_name', type=str, default=SUPPRESS,
                            help="name of the class out of 15 MVTec AD classes")
        parser.add_argument('--only_noise', type=lambda s: s.lower() == 'true', default=SUPPRESS,
                            help="apply only noise augmentation")
        parser.add_argument('--img_format', type=str, default='png', choices=['png', 'jpg'],
                            help="format of images")

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return argparse_utils.from_argparse_args(cls, args, **kwargs)
