from typing import Tuple

from albumentations import (CLAHE, Blur, Compose, HorizontalFlip,
                            HueSaturationValue, IAAAdditiveGaussianNoise,
                            IAAEmboss, IAASharpen, MedianBlur, MotionBlur,
                            Normalize, OneOf, RandomBrightnessContrast,
                            RandomRotate90, Resize, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2


def pixel_aug(p: float = 0.5, noise: bool = True,
              only_noise: bool = False) -> Compose:
    # borrowed from 'https://github.com/ORippler/gaussian-ad-mvtec/blob/main/src/common/augmentation.py'
    """Augmentation only on a pixel-level."""
    if not only_noise:
        augs = [
            OneOf(
                [
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ],
                p=0.2,
            ),
            OneOf(
                [
                    CLAHE(clip_limit=2),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomBrightnessContrast(brightness_limit=(-0.1, 0.2)),
                ],
                p=0.3,
            ),
            # Reduced hue shift to not change the color that much (purple
            # hazelnuts).
            # reduced val shift to not overly darken the image
            HueSaturationValue(
                hue_shift_limit=10, val_shift_limit=(-10, 20), p=0.3
            ),
        ]
    else:
        augs = []

    if noise or only_noise:
        augs.append(
            OneOf(
                [
                    # Slightly less aggressive:
                    IAAAdditiveGaussianNoise(
                        scale=(0.01 * 255, 0.03 * 255), per_channel=False
                    ),
                    IAAAdditiveGaussianNoise(
                        scale=(0.01 * 255, 0.03 * 255), per_channel=True
                    ),
                ],
                p=0.5 if only_noise else 0.2,
            )
        )
    return Compose(augs, p=p)


def get_transforms(resize: int, flip: bool, only_noise: bool, rotate90: bool,
                   shift_scale: bool, rotation_degrees: float, is_train: bool = True,
                   norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                   norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
    first_trfm = [Resize(resize, resize)]
    final_trfm = [Normalize(mean=norm_mean, std=norm_std), ToTensorV2()]

    if is_train:
        augs = [pixel_aug(p=1, noise=True, only_noise=only_noise)]
        if not only_noise:
            if flip:
                augs.append(HorizontalFlip())
            if rotate90:
                augs.append(RandomRotate90())

            shift_limit, scale_limit = (0.05, (-0.05, 0.1)) if shift_scale else (0, 0)
            augs += [ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit,
                                      rotate_limit=rotation_degrees, p=0.2)]

        augs = Compose(augs, p=1.0 if only_noise else 0.5)
        trfm = Compose([*first_trfm, augs, *final_trfm])
    else:
        trfm = Compose([*first_trfm, *final_trfm])

    return trfm
