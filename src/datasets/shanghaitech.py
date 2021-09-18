# partly borrowed from 'https://github.com/aimagelab/novelty-detection/blob/master/datasets/shanghaitech.py'
from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import Callable, List, NoReturn, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage.morphology import binary_dilation
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as T
from tqdm.auto import tqdm

from ..utils import video_to_frames
from .base import VideoAnomalyDetectionDataset

NUM_FRAMES = {
    'training': 274_516,
    'testing': 40_791,
}
RESIZE = (256, 512)

try:
    from skvideo.io import vread
except ImportError:
    vread = video_to_frames


class ShanghaiTechDataset(VideoAnomalyDetectionDataset):
    def __init__(self, root_path: PathLike, is_train: bool = True,
                 without_background: bool = True, transforms: Callable[[np.ndarray], torch.Tensor] = None):
        super(ShanghaiTechDataset, self).__init__()

        self.root_path = root_path
        self.is_train = is_train
        self.without_background = without_background
        self.transforms = transforms

        self.phase = 'training' if is_train else 'testing'
        self.data_dir = Path(self.root_path) / self.phase

        if self.is_train & (len(list((self.data_dir / 'frames').glob('*/*.jpg'))) != NUM_FRAMES['training']):
            print(len(list((self.data_dir / 'frames').glob('*/*.jpg'))))
            self.convert_train_videos_to_frames()

        self.video_ids = self.load_video_ids()

        if self.without_background:
            self.img_dir = self.data_dir / 'wo_background'
            self.img_dir.mkdir(exist_ok=True)
            if len(list(self.img_dir.glob('*/*.jpg'))) != NUM_FRAMES[self.phase]:
                self.create_and_save_frames_without_background()
            else:
                print(f'[Success] {self.phase.capitalize()} image data without background exists.')
        else:
            self.img_dir = self.data_dir / 'frames'

        self.frame_paths = self.load_frame_paths()
        self.sequence_gts = self.load_sequence_gts()

        assert len(self.frame_paths) == len(self.sequence_gts)

    def load_video_ids(self) -> List[str]:
        return sorted([d.name for d in (self.data_dir / 'frames').glob('*') if d.is_dir()])

    def load_frame_paths(self) -> List[str]:
        frame_paths = []
        for vid in self.video_ids:
            frame_paths.extend(sorted((self.img_dir / vid).glob('*.jpg')))
        return frame_paths

    def load_sequence_gts(self) -> np.ndarray:
        agg_clip_gts = []
        for vid in self.video_ids:
            if self.is_train:
                sequence_dir = self.data_dir / 'frames' / vid
                img_list = sorted(sequence_dir.glob('*.jpg'))
                clip_gt = np.zeros(len(img_list), dtype=int)
            else:
                clip_gt = np.load(self.data_dir / 'test_frame_mask' / f'{vid}.npy')
            agg_clip_gts.append(clip_gt)
        agg_clip_gts = np.concatenate(agg_clip_gts)

        return agg_clip_gts

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.frame_paths[idx]
        img = cv2.imread(str(img_path))

        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        else:
            img = T.functional.to_tensor(img)

        label = self.sequence_gts[idx]

        return img, label, idx

    def convert_train_videos_to_frames(self):
        train_video_dir = Path(self.root_path) / 'training' / 'videos'
        video_paths = sorted(train_video_dir.glob('*.avi'))
        num_videos = len(video_paths)

        train_jpg_dir = Path(self.root_path) / 'training' / 'frames'
        train_jpg_dir.mkdir(exist_ok=True)

        print('Converting video data to jpeg and save it.')
        pbar = tqdm(enumerate(video_paths), total=num_videos)
        for i, path in pbar:
            video_id = path.stem
            save_dir = train_jpg_dir / video_id
            save_dir.mkdir(exist_ok=True)
            frames = vread(str(path))
            pbar.set_postfix(OrderedDict(id=video_id, num_frames=frames.shape[0]))
            for j, frame in enumerate(frames):
                img_name = f'{j:04}.jpg'
                resized = cv2.resize(frame, (RESIZE[1], RESIZE[0]))
                save_path = save_dir / img_name
                cv2.imwrite(str(save_path), resized)

    @staticmethod
    def create_background(video_frames: np.ndarray) -> np.ndarray:
        """
        Create the background of a video via MOGs.
        :param video_frames: list of ordered frames (i.e., a video).
        :return: the estimated background of the video.
        """
        mog = cv2.createBackgroundSubtractorMOG2()
        for frame in video_frames:
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mog.apply(img)

        # Get background
        background = mog.getBackgroundImage()

        return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    @staticmethod
    def remove_background(video_frames: np.ndarray, background: np.ndarray) -> np.ndarray:
        threshold = 128
        mask = np.uint8(np.sum(np.abs(np.int32(video_frames) - background), axis=-1) > threshold)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.stack([binary_dilation(mask_frame, iterations=5) for mask_frame in mask])

        video_frames *= mask

        return video_frames

    def create_and_save_frames_without_background(self) -> NoReturn:
        pbar = tqdm(self.video_ids, desc=f'[create {self.phase} frames w/o background]',
                    total=len(self.video_ids))
        for vid in pbar:
            video_path = self.data_dir / 'frames' / vid
            frames = vread(str(video_path))
            background = self.create_background(frames)
            wo_background = self.remove_background(frames, background)
            sequence_dir = self.data_dir / 'frames' / vid
            img_names = [p.name for p in sorted(sequence_dir.glob('*.jpg'))]
            save_dir = self.data_dir / 'wo_background' / vid
            save_dir.mkdir(exist_ok=True)
            for wo_bg, name in zip(wo_background, img_names):
                save_path = save_dir / name
                cv2.imwrite(str(save_path), wo_bg)

    @property
    def collate_fn(self):
        return default_collate
