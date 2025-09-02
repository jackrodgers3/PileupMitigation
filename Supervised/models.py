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
            #nn.InstanceNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.main(x)


class GNNStack(nn.Module):
    def __init__(self, args):
        super(GNNStack, self).__init__()
        args.input_dim = args.input_dim - 2
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(args.input_dim, args.hidden_dim))
        self.norm = pyg_nn.LayerNorm(args.hidden_dim, mode="graph")
        assert (args.num_enc_layers >= 1) or (args.num_dec_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_enc_layers):
            self.convs.append(conv_model(args.hidden_dim, args.hidden_dim))
        self.post_mp = []
        for j in range(args.num_dec_layers):
            self.post_mp.append(Block(args.hidden_dim, args.act_fn, args.dropout))
        self.post_mp.append(nn.Linear(args.hidden_dim, args.output_dim))
        self.post_mp = nn.Sequential(*self.post_mp)
        self.dropout = args.dropout
        self.num_enc_layers = args.num_enc_layers
        self.num_dec_layers = args.num_dec_layers

    def build_conv_model(self, model_type):
        if model_type == 'Gated':
            return Gated_Model

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        eta_pos = 0
        phi_pos = 1
        pt_pos = 2
        eta = x[:, eta_pos]
        phi = x[:, phi_pos]
        pt = x[:, pt_pos]
        x = x[:, pt_pos:]

        for layer in self.convs:
            x = layer(x, edge_index, eta, phi)
            x = F.relu(x)
            x = self.norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)
        
        #print(x.shape)
        # force weights between 0, 1
        x = torch.sigmoid(x)

        # pt_i *= x, pt_i -> (px_i, py_i) -> (px, py) -> pt  
        pt = pt * x
        px, py = pt * torch.cos(phi), pt * torch.sin(phi)
        pxpy = torch.cat((px, py), dim=0)
        vec_sum = torch.sum(pxpy, dim=-1)
        final_pt = torch.hypot(vec_sum[0], vec_sum[1]).view(1, 1)
        return final_pt, x

class Gated_Model(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Gated_Model, self).__init__(aggr='mean')

        new_x_input = 3 * (in_channels) + 3 + 1
        self.x_dim = new_x_input
        self.lin_m2 = torch.nn.Linear(new_x_input, 1)

        self.lin_m5 = torch.nn.Linear(new_x_input + 2 * in_channels + 1, 1)
        self.lin_m5_g1 = torch.nn.Linear(in_channels, out_channels)
        self.lin_m5_g2 = torch.nn.Linear(new_x_input, out_channels)

    def forward(self, x, edge_index, eta, phi):
        num_nodes = x.size(0)
        x = torch.cat((x, eta.view(-1, 1)), dim=1)
        x = torch.cat((x, phi.view(-1, 1)), dim=1)
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    def message(self, x_j, x_i, edge_index, size, x):
        self.node_count = torch.tensor(size[0]).type(torch.float32).to('cuda')

        d_eta_phi = x_j[:, -2: x_j.size()[1]] - x_i[:, -2: x_i.size()[1]]

        # restrict phi
        indices = d_eta_phi[:, 1] > pi
        temp = torch.ceil((d_eta_phi[:, 1][indices] - pi) / (2 * pi)) * (2 * pi)
        d_eta_phi[:, 1][indices] = d_eta_phi[:, 1][indices] - temp

        d_r = torch.sum(torch.sqrt(d_eta_phi ** 2), dim=1).reshape(-1, 1)

        # get rid of eta, phi now
        x = x[:, 0:-2]
        x_i = x_i[:, 0:-2]
        x_j = x_j[:, 0:-2]
        x_g = torch.mean(x, dim=0)
        log_count = torch.log(self.node_count)
        log_count = log_count.repeat(x_i.size()[0], 1)
        x_g = x_g.repeat(x_i.size()[0], 1)
        x_j = torch.cat((x_j, x_i, x_g, d_eta_phi, d_r, log_count), dim=1)
        M_1 = self.lin_m2(x_j)
        M_2 = torch.sigmoid(M_1)
        x_j = x_j * M_2
        return x_j

    def update(self, aggr_out, x):
        x = x[:, 2:]
        x_g = torch.mean(x, dim=0)
        log_count = torch.log(self.node_count)
        log_count = log_count.repeat(x.size()[0], 1)
        x_g = x_g.repeat(x.size()[0], 1)
        aggr_out_temp = aggr_out
        aggr_out = torch.cat((aggr_out, x, x_g, log_count), dim=1)
        aggr_out = torch.sigmoid(self.lin_m5(aggr_out))
        aggr_out = F.relu(aggr_out * self.lin_m5_g1(x) +
                          (1 - aggr_out) * self.lin_m5_g2(aggr_out_temp))
        return aggr_out
