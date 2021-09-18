# taken some of the code from 'https://github.com/ORippler/gaussian-ad-mvtec/blob/main/src/gaussian/model.py'
from argparse import ArgumentParser
from typing import Callable, Mapping, NoReturn, Optional

import torch
from pytorch_lightning.utilities import argparse_utils
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf
from torch import Tensor
from torch.utils.data import DataLoader

from ...utils import aggregate_by_index, calc_metrics, t2np
from ..base_model import BaseScikitModel


class ScikitMaharanobis(BaseScikitModel):

    def __init__(self, input_dim: int, loss: Optional[Callable], tta: int,
                 device: Optional[str] = 'cuda'):
        super(ScikitMaharanobis, self).__init__()
        self.input_dim = input_dim
        self.loss = loss
        self.tta = tta
        self.inv_covariance = None
        self.device = device
        self.mean = None
        self.statistics_computed = False

    @property
    def run_type(self) -> str:
        return 'sklearn'

    @property
    def data_type(self) -> str:
        return 'tensor'

    @staticmethod
    def compute_mahalanobis_threshold(k: int, p: float = 0.9973) -> Tensor:
        """Compute a threshold on the mahalanobis distance.
        So that the probability of mahalanobis with k dimensions being less
        than the returned threshold is p.
        """
        # Mahalanobis² is Chi² distributed with k degrees of freedom.
        # So t is square root of the inverse cdf at p.
        return Tensor([chi2.ppf(p, k)]).sqrt()

    def compute_train_gaussian(self, input: Tensor) -> NoReturn:
        def fit_inv_covariance(input: Tensor):
            return Tensor(LedoitWolf().fit(input.cpu()).precision_)
        inv_covariance = fit_inv_covariance(input).to(self.device)
        mean = input.mean(dim=0).to(self.device)
        self.inv_covariance, self.mean = inv_covariance, mean

    def mahalanobis_distance(self, input: Tensor) -> Tensor:
        """Compute the batched mahalanobis distance.
        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.
        """
        assert input.dim() == 2
        inv_covariance = self.inv_covariance
        mean = self.mean
        assert 1 <= mean.dim() <= 2
        assert inv_covariance.dim() == 2
        assert input.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = input - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        return dist.sqrt()

    def fit(self, train_dataloader: DataLoader) -> NoReturn:
        X = train_dataloader.dataset[:][0]
        self.compute_train_gaussian(X.to(self.device))
        self.statistics_computed = True

    def test(self, test_dataloader: DataLoader) -> Mapping:
        X, y_true, test_ids = test_dataloader.dataset[:][:]
        X = X.to(self.device)

        assert self.statistics_computed

        dist = self.mahalanobis_distance(X)

        is_anomaly = torch.Tensor([0 if x == 0 else 1 for x in y_true])
        anomaly_score = dist
        if self.tta > 1:
            is_anomaly = aggregate_by_index(is_anomaly, test_ids).int()
            anomaly_score = aggregate_by_index(anomaly_score, test_ids).float()
        metrics = calc_metrics(t2np(is_anomaly), t2np(anomaly_score))

        return metrics

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser,) -> ArgumentParser:
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return argparse_utils.from_argparse_args(cls, args, **kwargs)
