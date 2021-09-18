import sys
from argparse import ArgumentParser
from functools import partialmethod
from typing import List, Mapping, Optional, Tuple, Union

import cupy
import cv2
import numpy as np
import torch
from cupy import fromDlpack as cupy_from_dlpack
from torch.utils.dlpack import to_dlpack as torch_to_dlpack


def int_or_str(x: str):
    if x.isdigit():
        return int(x)
    else:
        return x


def t2np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().data.numpy() if tensor is not None else None


def preprocess_batch(data: Union[List, Mapping], array_type: str = 'torch',
                     device: Union[str, torch.device] = 'cuda') -> Mapping:
    assert array_type in ['torch', 'cupy', 'numpy']
    if isinstance(data, list):
        inputs, labels, idx = data
    elif isinstance(data, dict):
        inputs, labels, idx = data['image'], data['label'], data['id']
    else:
        msg = "data must be provided as \"list\" or \"dict\""
        raise TypeError(msg)

    if array_type == 'torch':
        x, y = inputs.to(device), labels.to(device)
    elif array_type == 'cupy':
        dx, dy = torch_to_dlpack(inputs), torch_to_dlpack(labels)
        x, y = cupy_from_dlpack(dx), cupy_from_dlpack(dy)
    elif array_type == 'numpy':
        x, y = inputs.cpu().numpy(), labels.cpu().numpy()
    else:
        msg = "X,y matrix format " + array_type + " not supported"
        raise TypeError(msg)

    return {'input': x, 'label': y, 'id': idx}


def cupy_to_torch(x: cupy.ndarray) -> torch.Tensor:
    assert isinstance(x, cupy.ndarray)
    # the following code does not work in my env, for some reason
    # dx = x.toDlpack()
    # tx = torch.utils.dlpack.from_dlpack(dx)
    nx = cupy.asnumpy(x)
    tx = torch.from_numpy(nx)

    return tx


def torch_to_cupy(x: torch.Tensor) -> cupy.ndarray:
    assert isinstance(x, torch.Tensor)
    # the following code does not work in my env, for some reason
    # dx = torch.utils.dlpack.to_dlpack(x)
    # cx = cupy.fromDlpack(dx)
    cx = cupy.asarray(x.clone().cpu().numpy())  # this works fine
    return cx


def to_numpy(x: Union[torch.Tensor, cupy.ndarray, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.clone().cpu().numpy()
    elif isinstance(x, cupy.ndarray):
        return cupy.asnumpy(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        msg = "X matrix format " + str(x.__class__) + " not supported"
        raise TypeError(msg)


def video_to_frames(path: str, img_size: Optional[Tuple[int, int]] = None,
                    resize: Optional[Tuple[int, int]] = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if img_size is not None:
        assert (h, w) == img_size
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = []
    while(cap.isOpened()):
        success, frame = cap.read()
        if success:
            if resize is not None:
                resized = cv2.resize(frame, resize)
                frame_list.append(resized)
            else:
                frame_list.append(frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frames = np.stack(frame_list)
    assert frames.shape[0] == num_frames

    cv2.destroyAllWindows()
    cap.release()

    return frames


# borrowed from
# 'https://stackoverflow.com/questions/38911146/python-equivalent-of-functools-partial-for-a-class-constructor#answer-38911383'
def partialclass(name, cls, *args, **kwds):
    new_cls = type(name, (cls,), {
        '__init__': partialmethod(cls.__init__, *args, **kwds)
    })

    # The following is copied nearly ad verbatim from `namedtuple's` source.
    """
    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).
    """
    try:
        new_cls.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    return new_cls


class Null:
    # borrowed from 'https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s24.html'
    """ Null objects always and reliably "do nothing." """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __repr__(self):
        return "Null(  )"

    def __nonzero__(self):
        return 0

    def __getattr__(self, name):
        return self

    def __setattr__(self, name, value):
        return self

    def __delattr__(self, name):
        return self

    def add_argparse_args(parser: ArgumentParser) -> ArgumentParser:
        return parser
