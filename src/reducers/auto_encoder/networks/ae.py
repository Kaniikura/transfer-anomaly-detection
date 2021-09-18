# partly borrowed from 'https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py'

import torch
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H2, H1, D_out):
        super(Encoder, self).__init__()
        self.output_dim = D_out
        self.linear1 = torch.nn.Linear(D_in, H2)
        self.bn1 = torch.nn.BatchNorm1d(H2)
        self.linear2 = torch.nn.Linear(H2, H1)
        self.bn2 = torch.nn.BatchNorm1d(H1)
        self.linear3 = torch.nn.Linear(H1, D_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        return self.linear3(x)


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


class VanillaAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VanillaAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
