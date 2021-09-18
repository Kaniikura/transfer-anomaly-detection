# partly borrowed from 'https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py'

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.output_dim = D_out
        self.linear1 = torch.nn.Linear(D_in, H)
        self.bn1 = torch.nn.BatchNorm1d(H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.bn2 = torch.nn.BatchNorm1d(D_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        return F.relu(self.bn2(self.linear2(x)))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Decoder, self).__init__()
        self.input_dim = D_in
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.bn1 = torch.nn.BatchNorm1d(H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.bn2 = torch.nn.BatchNorm1d(H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        return self.linear3(x)


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        enc_out_dim = self.encoder.output_dim
        dec_input_dim = self.decoder.input_dim
        self._enc_mu = torch.nn.Linear(enc_out_dim, dec_input_dim)
        self._enc_log_sigma = torch.nn.Linear(enc_out_dim, dec_input_dim)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(mu.device)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)
