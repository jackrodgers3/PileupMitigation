import torch_geometric
import torch_geometric.nn as pygnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import  global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Linear, Dropout


class GNN(nn.Module):
    def __init__(self, feature_size, d_model, dropout, act_fn):
        super(GNN, self).__init__()
        if act_fn == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif act_fn == 'relu':
            self.act = nn.ReLU()
        elif act_fn == 'tanh':
            self.act = nn.Tanh()
        elif act_fn == 'gelu':
            self.act = nn.GELU()
        self.init_conv = GCNConv(feature_size, d_model)
        self.conv1 = GCNConv(d_model, d_model)
        self.conv2 = GCNConv(d_model, d_model)
        self.conv3 = GCNConv(d_model, d_model)
        self.conv4 = GCNConv(d_model, d_model)
        self.dropout1 = Dropout(p=dropout)
        self.dropout2 = Dropout(p=dropout)
        self.dropout3 = Dropout(p=dropout)
        self.dropout4 = Dropout(p=dropout)
        self.dropout5 = Dropout(p=dropout)

        self.out1 = Linear(d_model*2, d_model)
        self.out2 = Linear(d_model, 1)

    def forward(self, x, edge_index, batch_index):
        # first conv
        hidden = self.init_conv(x, edge_index)
        hidden = self.act(hidden)
        hidden = self.dropout1(hidden)
        hidden = self.conv1(hidden, edge_index)
        hidden = self.act(hidden)
        hidden = self.dropout2(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = self.act(hidden)
        hidden = self.dropout3(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = self.act(hidden)
        hidden = self.dropout4(hidden)
        hidden = self.conv4(hidden, edge_index)
        hidden = self.act(hidden)

        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)

        out = self.out1(hidden)
        out = self.act(out)
        out = self.dropout5(out)
        out = self.out2(out)
        return out