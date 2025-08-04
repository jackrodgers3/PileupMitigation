"""
Save models here
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import math
from math import pi
from torch_geometric.nn.models import GAT
from torch_geometric.nn import GCNConv

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.7, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none').view(-1)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss).view(-1)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class Block(nn.Module):
    def __init__(self, hidden_dim, act_fn, dropout):
        super(Block, self).__init__()
        self.hidden_dim = hidden_dim
        if act_fn == 'relu':
            self.act_fn = nn.ReLU()
        elif act_fn == 'leakyrelu':
            self.act_fn = nn.LeakyReLU()
        elif act_fn == 'gelu':
            self.act_fn = nn.GELU()
        self.dropout = dropout
        self.main = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.act_fn,
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.main(x)


class GATFull(nn.Module):
    def __init__(self, in_channels=9, hidden_channels=128, num_enc_layers=3, num_dec_layers=3, out_channels=1, dropout = 0.1, act_fn = 'relu', aggr='mean'):
        super(GATFull, self).__init__()
        self.encoder = (GAT
            (
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_enc_layers,
            out_channels=hidden_channels,
            dropout=dropout
        )
        )
        self.decoder = []
        for j in range(num_dec_layers - 1):
            self.decoder.append(Block(hidden_channels, act_fn, dropout))
        self.decoder.append(nn.Linear(hidden_channels, out_channels))
        self.decoder = nn.Sequential(*self.decoder)
        self.aggr = aggr

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.encoder(x, edge_index)
        x = self.decoder(x)
        if self.aggr == 'mean':
            x = torch.mean(x, dim=0)
        elif self.aggr == 'sum':
            x = torch.sum(x, dim=0)
        x = x.view(-1, 1)
        return x

