from argparse import ArgumentParser
from typing import Callable, Mapping, NoReturn, Optional

from cuml.neighbors import KNeighborsRegressor as KNR
from pytorch_lightning.utilities import argparse_utils
from sklearn.utils.validation import check_is_fitted
from torch import Tensor
from torch.utils.data import DataLoader

from ...utils import (aggregate_by_index, calc_metrics, cupy_to_torch, t2np,
                      torch_to_cupy)
from ..base_model import BaseScikitModel


class ScikitKnn(BaseScikitModel):

    def __init__(self, n_neighbors: int, loss: Optional[Callable], tta: int,
                 input_dim: Optional[int] = None):
        super(ScikitKnn, self).__init__()
        self.n_neighbors = n_neighbors
        self.model = KNR(n_neighbors=n_neighbors)
        self.loss = loss
        self.tta = tta
        self.input_dim = input_dim

    @property
    def data_type(self) -> str:
        return 'cupy'

    @property
    def run_type(self) -> str:
        return 'sklearn'

    def fit(self, train_dataloader: DataLoader) -> NoReturn:
        X, y = train_dataloader.dataset[:][0:2]
        X, y = torch_to_cupy(X), torch_to_cupy(y)
        self.model.fit(X, y)

    def test(self, test_dataloader: DataLoader) -> Mapping:
        X, y_true, test_ids = test_dataloader.dataset[:][:]
        X, y_true = torch_to_cupy(X), y_true
        check_is_fitted(self.model)
        neigh_dist, _ = self.model.kneighbors(X)
        neigh_dist = cupy_to_torch(neigh_dist)

        is_anomaly = Tensor([0 if x == 0 else 1 for x in y_true])
        anomaly_score = neigh_dist.mean(dim=1)
        if self.tta > 1:
            is_anomaly = aggregate_by_index(is_anomaly, test_ids).int()
            anomaly_score = aggregate_by_index(anomaly_score, test_ids).float()
        metrics = calc_metrics(t2np(is_anomaly), t2np(anomaly_score))

        return metrics

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser,) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False, allow_abbrev=False)
        parser.add_argument('--n_neighbors', type=int, default=5,
                            help="number of neighbors to use by default for kneighbors queries")

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return argparse_utils.from_argparse_args(cls, args, **kwargs)
