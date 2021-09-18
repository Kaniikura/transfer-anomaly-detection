__all__ = [
    'MODEL_REGISTRY',
]

from ..utils import Registry
from .flow import LitFlow
from .knn import ScikitKnn
from .mahalanobis import ScikitMaharanobis
from .ocsvm import ScikitOCSVM

MODEL_REGISTRY = Registry('model')
MODEL_REGISTRY.register(LitFlow, name='flow')
MODEL_REGISTRY.register(ScikitMaharanobis, name='mvg')
MODEL_REGISTRY.register(ScikitKnn, name='knn')
MODEL_REGISTRY.register(ScikitOCSVM, name='ocsvm')
