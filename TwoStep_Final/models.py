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


class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args, disc=True):
        # since we do not need phi and eta in x
        input_dim = input_dim - 2

        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, args.hidden_dim))
        assert (args.num_enc_layers >= 1) or (args.num_dec_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_enc_layers - 1):
            self.convs.append(conv_model(args.hidden_dim, args.hidden_dim))
        self.disc = disc

        # post-message-passing
        if self.disc:
            self.post_mp = []
            self.post_da = []
            for j in range(args.num_dec_layers - 1):
                self.post_mp.append(Block(args.hidden_dim, args.act_fn, args.dropout))
                self.post_da.append(Block(args.hidden_dim, args.act_fn, args.dropout))
            self.post_mp.append(nn.Linear(args.hidden_dim, output_dim))
            self.post_da.append(nn.Linear(args.hidden_dim, output_dim))


            self.post_da = nn.Sequential(*self.post_da)

            self.grl = GradientReverseLayer()

            self.dropout = args.dropout
            self.beta_one = nn.parameter.Parameter(torch.rand(1))
            self.num_enc_layers = args.num_enc_layers
            self.num_dec_layers = args.num_dec_layers

            self.lamb = args.lamb
        else:
            self.post_mp = []
            for j in range(args.num_dec_layers - 1):
                self.post_mp.append(Block(args.hidden_dim, args.act_fn, args.dropout))
            self.post_mp.append(nn.Linear(args.hidden_dim, output_dim))
            self.post_mp = nn.Sequential(*self.post_mp)
            self.dropout = args.dropout
            self.num_enc_layers = args.num_enc_layers
            self.num_dec_layers = args.num_dec_layers

    def build_conv_model(self, model_type):
        if model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'Gated':
            return Gated_model

    def forward(self, data):
        if self.disc:
            num_feature = data.num_feature_actual[0].item()
            original_x = data.x[:, 0:num_feature]
            train_mask = data.x[:, num_feature]
            train_default = data.x[:, (num_feature + 1):]

            x = torch.transpose(original_x, 0, 1) * (1 - train_mask) + \
                torch.transpose(train_default, 0, 1) * train_mask
            x = torch.transpose(x, 0, 1)
        else:
            x = data.x

        edge_index = data.edge_index
        eta_pos = 0
        phi_pos = 1
        eta = x[:, eta_pos]
        phi = x[:, phi_pos]
        # x = x[:, 2:-1]
        x[:, 2] = torch.log10(x[:, 2])
        x = x[:, 2:]

        # x = self.before_mp(x)

        for layer in self.convs:
            x = F.dropout(F.leaky_relu(layer(x, edge_index, eta, phi)), p=self.dropout)

        if self.disc:
            x_cl = self.post_mp(x)
            x_da = self.grl(x)
            x_da = self.post_da(x_da)

            x_cl = torch.sigmoid(x_cl)
            x_da = torch.sigmoid(x_da)

            return x_cl, x_da
        else:
            x_dec = self.post_mp(x)
            return x_dec

    def loss(self, pred, label, domain_discriminant, domain_label):
        # loss = nn.CrossEntropyLoss()
        """
        weight = torch.zeros_like(label)
        weight[label == 1] = 2
        weight[label == 0] = 0.2
        """
        loss = nn.BCELoss()
        eps = 1e-9
        temp = loss(pred + eps, label)
        temp_da = loss(domain_discriminant + eps, domain_label)
        return temp + self.lamb*temp_da

    def loss_reco(self, pred, label):
        loss = nn.MSELoss()
        temp = loss(pred, label)
        return temp


class GraphSage(pyg_nn.MessagePassing):

    def __init__(self, in_channels, out_channels, reducer='meansacct',
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')
        in_channels = in_channels
        self.lin = torch.nn.Linear(2 * in_channels + 1, out_channels)
        self.agg_lin = torch.nn.Linear(
            out_channels + 2 * in_channels + 1, out_channels)

        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index, eta, phi):
        # x = torch.cat((x, eta.view(-1, 1)), dim=1)
        # x = torch.cat((x, phi.view(-1, 1)), dim=1)
        num_nodes = x.size(0)
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    def message(self, x_j, edge_index, size, x_i, x):
        self.node_count = torch.tensor(size[0]).type(torch.float32).to('cuda')
        x_g = torch.mean(x, dim=0)
        log_count = torch.log(self.node_count)
        log_count = log_count.repeat(x_i.size()[0], 1)
        x_g = x_g.repeat(x_i.size()[0], 1)
        x_j = torch.cat((x_j, x_g, log_count), dim=1)
        x_j = self.lin(x_j)
        return x_j

    def update(self, aggr_out, x):
        x_g = torch.mean(x, dim=0)
        x_g = x_g.repeat(x.size()[0], 1)
        log_count = torch.log(self.node_count)
        log_count = log_count.repeat(x.size()[0], 1)
        aggr_out = torch.cat((aggr_out, x, x_g, log_count), dim=-1)
        aggr_out = self.agg_lin(aggr_out)
        aggr_out = F.relu(aggr_out)
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, x, coeff = 1.):
        ctx.coeff = coeff
        output = x * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, x):
        return GradientReverseFunction.apply(x)


class Gated_model(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, normalize_embedding=True):
        super(Gated_model, self).__init__(aggr='mean')
        # last sclar = d_eta, d_phi, d_R, append x_i, x_j, x_g so * 3，+1 for log count
        new_x_input = 3 * (in_channels) + 3 + 1
        self.x_dim = new_x_input
        self.lin_m2 = torch.nn.Linear(new_x_input, 1)
        # also append x and x_g, so + 2 * in_channels, +1 for log count in the global node
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
        # eta at position 0
        dif_eta_phi = x_j[:, -2: x_j.size()[1]] - x_i[:, -2: x_i.size()[1]]

        # make sure delta within 2pi
        indices = dif_eta_phi[:, 1] > pi
        temp = torch.ceil(
            (dif_eta_phi[:, 1][indices] - pi) / (2 * pi)) * (2 * pi)
        dif_eta_phi[:, 1][indices] = dif_eta_phi[:, 1][indices] - temp

        delta_r = torch.sum(torch.sqrt(dif_eta_phi ** 2), dim=1).reshape(-1, 1)

        x = x[:, 0:-2]
        x_i = x_i[:, 0:-2]
        x_j = x_j[:, 0:-2]
        x_g = torch.mean(x, dim=0)
        log_count = torch.log(self.node_count)
        log_count = log_count.repeat(x_i.size()[0], 1)
        x_g = x_g.repeat(x_i.size()[0], 1)
        x_j = torch.cat((x_j, x_i, x_g, dif_eta_phi,
                        delta_r, log_count), dim=1)
        M_1 = self.lin_m2(x_j)
        M_2 = torch.sigmoid(M_1)
        x_j = x_j * M_2
        return x_j

    def update(self, aggr_out, x):
        x = x[:, 0:-2]
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

