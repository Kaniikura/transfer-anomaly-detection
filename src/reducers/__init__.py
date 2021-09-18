
__all__ = [
    'REDUCER_REGISTRY',
]

from ..utils import Null, Registry
from .auto_encoder import LitVAE, LitVanillaAE
from .pca import PCAReducer

REDUCER_REGISTRY = Registry('reducer')
REDUCER_REGISTRY.register(LitVAE, name='vae')
REDUCER_REGISTRY.register(LitVanillaAE, name='ae')
REDUCER_REGISTRY.register(PCAReducer, name='pca')
REDUCER_REGISTRY.register(Null, name='none')
