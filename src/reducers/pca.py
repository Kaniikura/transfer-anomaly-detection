from argparse import ArgumentParser
from typing import NoReturn

import numpy as np
from cuml import PCA
from pytorch_lightning.utilities import argparse_utils
from torch.utils.data import DataLoader

from ..utils import torch_to_cupy
from .base_reducer import BaseScikitReducer


class PCAReducer(BaseScikitReducer):
    def __init__(self, n_components: int = None, top_n_components: int = None,
                 cumulative_variance_threshold: float = None, percentile_components: float = None,
                 negated_pca: bool = False, **kwargs):
        self.n_components = n_components

        self.top_n_components = top_n_components
        self.percentile_components = percentile_components
        self.cumulative_variance_threshold = cumulative_variance_threshold
        pca_trfm_keywords = ['top_n_components', 'percentile_components', 'cumulative_variance_threshold']
        key_counts = sum([self.__dict__[k] is not None for k in pca_trfm_keywords])
        if key_counts != 1:
            raise TypeError(f'you must specify exactly one of {*pca_trfm_keywords,}')

        self.model = PCA(n_components=n_components)
        self.negated_pca = negated_pca
        self._data_type = 'cupy'

    @property
    def data_type(self) -> str:
        return self._data_type

    @classmethod
    def run_type(cls) -> str:
        return 'sklearn'

    def fit(self, train_dataloader: DataLoader) -> NoReturn:
        _X = train_dataloader.dataset[:][0]
        if self.n_components is None:
            self.n_components = min(_X.shape)
            self.model.n_components = self.n_components
        X = torch_to_cupy(_X)
        self.model.fit(X)
        evr = self.model.explained_variance_ratio_
        _arange = np.arange(self.n_components)
        if self.cumulative_variance_threshold is not None:
            idx = int(np.argmin(np.abs(np.cumsum(evr) - self.cumulative_variance_threshold)))
            self.indices = _arange[idx:] if self.negated_pca else _arange[:idx + 1]
        elif self.top_n_components is not None:
            idx = self.top_n_components
            self.indices = _arange[-idx:] if self.negated_pca else _arange[:idx]
        else:  # self.percentile_components is not None
            idx = int(round(self.percentile_components * self.n_components))
            self.indices = _arange[-idx:] if self.negated_pca else _arange[:idx]

    def __call__(self, x):
        return self.model.transform(x)[:, self.indices]

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False, allow_abbrev=False)
        parser.add_argument('--negated_pca', type=lambda s: s.lower() == 'true', default=False, required=False,
                            help="take the elements with the lowest cumulative variance in order")
        parser.add_argument('--cumulative_variance_threshold', type=float, default=None, required=False,
                            help="threshold of cumulative variance")
        parser.add_argument('--top_n_components', type=int, default=None, required=False,
                            help="number of components")
        parser.add_argument('--percentile_components', type=float, default=None, required=False,
                            help="percent of the number of components")
        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return argparse_utils.from_argparse_args(cls, args, **kwargs)
