__all__ = [
    'DATA_MODULE_REGISTRY',
]

from ..utils import Registry
from .cifar10 import CIFAR10DataModule
from .mvtec import MVTecDataModule
from .shanghaitech import ShanghaiTechDataModule

DATA_MODULE_REGISTRY = Registry('data_module')
DATA_MODULE_REGISTRY.register(CIFAR10DataModule, name='cifar10')
DATA_MODULE_REGISTRY.register(MVTecDataModule, name='mvtec')
DATA_MODULE_REGISTRY.register(ShanghaiTechDataModule, name='shanghai')
