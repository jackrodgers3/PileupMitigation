"""
Save models here
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import math
from math import pi
from torch_geometric.data import DataLoader
import pickle
from torch_geometric.nn.models import GAT, GCN
from torch_geometric.nn import GCNConv, global_mean_pool

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
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.main(x)


class SupervisedDummy(nn.Module):
    def __init__(self, in_channels=11, hidden_channels=128, num_enc_layers=3, num_dec_layers=3, out_channels=1, dropout = 0.1, act_fn = 'relu', aggr='mean'):
        super(SupervisedDummy, self).__init__()
        self.encoder = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_enc_layers,
            out_channels=hidden_channels,
            dropout=dropout
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
        print(x.shape)
        if self.aggr == 'mean':
            x = torch.mean(x, dim=0)
        elif self.aggr == 'sum':
            x = torch.sum(x, dim=0)
        x = x.view(-1, 1)
        return x


class Final_GCN(nn.Module):
    def __init__(self, in_channels=11, hidden_channels=128, dropout = 0.1, n_encoder_layers=2, n_decoder_layers=2, act_fn='relu'):
        super(Final_GCN, self).__init__()
        self.n_encoder_layers = n_encoder_layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.decoder = []
        self.out = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        for _ in range(n_decoder_layers):
            self.decoder.append(Block(hidden_channels, act_fn, dropout))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        if self.n_encoder_layers > 4:
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.conv5(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.conv6(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        elif self.n_encoder_layers > 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.conv5(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        elif self.n_encoder_layers > 2:
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv4(x, edge_index)

        x = global_mean_pool(x, batch)

        x = self.decoder(x)
        x = self.out(x)

        return x
