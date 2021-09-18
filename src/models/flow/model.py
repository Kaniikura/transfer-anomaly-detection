# modified from 'https://github.com/marco-rudolph/differnet/blob/master/model.py'
from torch import nn

from .freia_funcs import (F_fully_connected, InputNode, Node, OutputNode,
                          ReversibleGraphNet, glow_coupling_layer,
                          permute_layer)


def nf_head(input_dim, clamp_alpha=3, n_coupling_blocks=8, fc_internal=2048, dropout=0.0):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': clamp_alpha, 'F_class': F_fully_connected,
                           'F_args': {'internal_size': fc_internal, 'dropout': dropout}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder


class Flow(nn.Module):
    def __init__(self, input_dim):
        super(Flow, self).__init__()
        self.nf = nf_head(input_dim=input_dim, fc_internal=input_dim)

    def forward(self, x):
        z = self.nf(x)
        return z
