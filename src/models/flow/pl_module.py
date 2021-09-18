from argparse import ArgumentParser
from typing import Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import argparse_utils
from torch import Tensor

from ...utils import aggregate_by_index, calc_metrics, t2np
from .model import Flow


class LitFlow(pl.LightningModule):

    def __init__(self, input_dim: int,
                 loss: Callable[[Tensor, Tensor], Tensor], tta: int,
                 lr_init: float):
        super(LitFlow, self).__init__()
        self.model = Flow(input_dim=input_dim)
        self.loss = loss
        self.tta = tta
        self.lr_init = lr_init

    def forward(self, x):
        z = self.model(x)
        return z

    @property
    def run_type(self) -> str:
        return 'lightning'

    def training_step(self, batch, batch_idx):
        X = batch[0]
        z = self.forward(X)
        loss_train = self.loss(z, self.model.nf.jacobian(run_forward=False))
        self.log('train_loss', loss_train, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss_train}

    def test_step(self, batch, batch_idx):
        X, y, idx = batch
        z = self.forward(X)
        loss_test = self.loss(z, self.model.nf.jacobian(run_forward=False))
        self.log_dict({'test_loss': loss_test})

        return {'loss': loss_test, 'y_true': y.detach().cpu(), 'idx': idx, 'z_test': z.detach().cpu()}

    def test_epoch_end(self, outputs):
        loss_test = torch.stack([o['loss'] for o in outputs]).mean()
        y_true = torch.cat([o['y_true'] for o in outputs]).numpy()
        test_ids = torch.cat([o['idx'] for o in outputs])
        z_test = torch.cat([o['z_test'] for o in outputs])

        is_anomaly = torch.Tensor([0 if x == 0 else 1 for x in y_true])
        anomaly_score = torch.mean(z_test ** 2, dim=-1)
        if self.tta > 1:
            is_anomaly = aggregate_by_index(is_anomaly, test_ids).int()
            anomaly_score = aggregate_by_index(anomaly_score, test_ids).float()
        metrics = calc_metrics(t2np(is_anomaly), t2np(anomaly_score))

        # logging
        self.log_dict({'test_loss': loss_test})
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.nf.parameters(), lr=self.lr_init,
                                     betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
        return optimizer

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser,) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False, allow_abbrev=False)
        parser.add_argument('--lr_init', type=float, default=2e-4,
                            help="initial learning rate")

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return argparse_utils.from_argparse_args(cls, args, **kwargs)
