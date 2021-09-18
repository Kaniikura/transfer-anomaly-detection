# modified from https://github.com/byungjae89/SPADE-pytorch/blob/master/src/datasets/mvtec.py

from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class MVTecDataset(Dataset):
    def __init__(self, class_name: str, root_path: str = './data',
                 transforms: Callable[[np.ndarray], torch.Tensor] = None,
                 is_train: bool = True, format: Literal['png', 'jpg'] = 'png'):
        self.class_name = class_name
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.transforms = transforms
        self.is_train = is_train
        self.format = format

        # load dataset
        self.x, self.y = self.load_dataset_folder()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, y = self.x[idx], self.y[idx]
        image = np.array(Image.open(path))
        if image is None or image.size == 0:
            raise OSError("Could not read image: {}".format(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        else:
            image = T.functional.to_tensor(image)

        return {'image': image, 'label': y, 'id': idx}

    def __len__(self) -> int:
        return len(self.x)

    def load_dataset_folder(self) -> Tuple[List[PathLike], List[int]]:
        phase = 'train' if self.is_train else 'test'
        x, y = [], []
        img_dir = Path(self.root_path, self.class_name, phase)

        img_types = sorted([x.name for x in img_dir.iterdir()])
        for img_type in img_types:
            # load images
            img_type_dir = Path(img_dir, img_type)
            if not img_type_dir.is_dir():
                continue
            img_fpath_list = sorted([Path(img_type_dir, x.name)
                                     for x in img_type_dir.iterdir()
                                     if x.name.endswith(f'.{self.format}')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y)
