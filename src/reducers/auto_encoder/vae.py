from argparse import ArgumentParser
from typing import Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import argparse_utils

from .networks.vae import VAE, Decoder, Encoder


class LitVAE(pl.LightningModule):

    def __init__(self, input_dim: int, z_dim: int,
                 hidden_dim1: Optional[int] = None,
                 hidden_dim2: Optional[int] = None,
                 loss: Optional[Callable] = None,
                 use_latent_loss: bool = True, tta: int = 1,
                 lr_init: float = 5e-04, **kwargs):
        super(LitVAE, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        _, _hd2, _hd1, _ = np.geomspace(self.z_dim, self.input_dim, num=4, dtype=int)
        self.hidden_dim1 = hidden_dim1 or _hd1
        self.hidden_dim2 = hidden_dim2 or _hd2
        self.loss = loss or torch.nn.MSELoss()
        self.use_latent_loss = use_latent_loss
        self.tta = tta
        self.lr_init = lr_init
        self.__data_type = 'torch'

        self.encoder = Encoder(self.input_dim, self.hidden_dim1, self.hidden_dim2)
        self.decoder = Decoder(self.z_dim, self.hidden_dim2, self.hidden_dim1, self.input_dim)
        self.vae = VAE(self.encoder, self.decoder)

    def forward(self, x):
        dec = self.vae(x)
        return dec

    def encode(self, x):
        self.encoder.eval()
        with torch.no_grad():
            enc = self.encoder(x)
            mu = self.vae._enc_mu(enc)

        return mu

    @property
    def data_type(self) -> str:
        return self.__data_type

    @classmethod
    def run_type(cls) -> str:
        return 'lightning'

    def training_step(self, batch, batch_idx):
        X = batch[0]
        dec = self.forward(X)
        if self.use_latent_loss:
            ll = self.latent_loss(self.vae.z_mean, self.vae.z_sigma)
            loss_train = self.loss(dec, X) + ll
        else:
            loss_train = self.loss(dec, X)
        self.log('train_loss', loss_train, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss_train}

    @staticmethod
    def latent_loss(z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lr_init, weight_decay=1e-5)
        return optimizer

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser,) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False, allow_abbrev=False)
        parser.add_argument('--z_dim', type=int, default=32,
                            help="dimensionality of the latent space")
        parser.add_argument('--reducer_train_epochs', type=int, default=5,
                            help="number of epochs for training dimensionality reduction model")
        parser.add_argument('--lr_init', type=float, default=5e-4,
                            help="initial learning rate")

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return argparse_utils.from_argparse_args(cls, args, **kwargs)
