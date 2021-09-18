from argparse import ArgumentParser
from typing import Callable, Mapping, NoReturn, Optional

import torch
from sklearn.svm import OneClassSVM
from pytorch_lightning.utilities import argparse_utils
from sklearn.utils.validation import check_is_fitted
from torch import Tensor
from torch.utils.data import DataLoader

from ...utils import aggregate_by_index, calc_metrics, t2np
from ..base_model import BaseScikitModel


class ScikitOCSVM(BaseScikitModel):

    def __init__(self, nu: float, loss: Optional[Callable], tta: int,
                 input_dim: Optional[int] = None):
        super(ScikitOCSVM, self).__init__()
        self.nu = nu
        self.model = OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
        self.loss = loss
        self.tta = tta
        self.input_dim = input_dim

    @property
    def data_type(self) -> str:
        return 'numpy'

    @property
    def run_type(self) -> str:
        return 'sklearn'

    def fit(self, train_dataloader: DataLoader) -> NoReturn:
        X, y = train_dataloader.dataset[:][0:2]
        X, y = t2np(X), t2np(y)
        # NOTE: OC-SVM treats negative values are outliers and non-negative ones are inliers
        y = 1 - 2 * y  # (0, 1) -> (1, -1)
        self.model.fit(X, y)

    def test(self, test_dataloader: DataLoader) -> Mapping:
        X, y_true, test_ids = test_dataloader.dataset[:][:]
        X, y_true = t2np(X), y_true
        check_is_fitted(self.model)
        dist = torch.from_numpy(self.model.decision_function(X)).float()

        is_anomaly = Tensor([0 if x == 0 else 1 for x in y_true])
        anomaly_score = - dist  # NOTE: negative values are outliers
        if self.tta > 1:
            is_anomaly = aggregate_by_index(is_anomaly, test_ids).int()
            anomaly_score = aggregate_by_index(anomaly_score, test_ids).float()
        metrics = calc_metrics(t2np(is_anomaly), t2np(anomaly_score))

        return metrics

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser,) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False, allow_abbrev=False)
        parser.add_argument('--nu', type=float, default=0.5,
                            help=r"""An upper bound on the fraction of training errors and a lower bound of the fraction of
                                     support vectors. Should be in the interval (0, 1]. By default 0.5 will be taken.""")

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return argparse_utils.from_argparse_args(cls, args, **kwargs)
