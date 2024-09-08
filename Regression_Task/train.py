import torch
import torch_geometric as pyg
import torch.nn as nn
import wandb
import torch_geometric.nn as pygnn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import argparse
import pickle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.spatial import distance
from math import pi
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dR = 0.4
with open(r"D:\Data\Research\PUPPI\Regression_Data\W\train700_reco", 'rb') as f:
    train_data = pickle.load(f)
f.close()

with open(r"D:\Data\Research\PUPPI\Regression_Data\W\valid150_reco", 'rb') as f:
    valid_data = pickle.load(f)
f.close()

with open(r"D:\Data\Research\PUPPI\Regression_Data\W\test150_reco", 'rb') as f:
    test_data = pickle.load(f)
f.close()


def convert_data(pre_data, dR):
    new_data = []
    for h in tqdm(range(len(pre_data))):
        x = []
        label = []
        # filter out PU from data
        for i in range(len(pre_data[h].x)):
            for j in range(len(pre_data[h].GenPart_nump)):
                if abs(pre_data[h].x[i][0] - pre_data[h].GenPart_nump[j][0]) < 0.003 and abs(
                        pre_data[h].x[i][1] - pre_data[h].GenPart_nump[j][1]) < 0.003:
                    x.append(np.array(pre_data[h].x[i]))
                    label.append(np.array(pre_data[h].GenPart_nump[j][2]))
        x = np.array(x)
        x[:, 2] = np.log(x[:, 2])
        label = np.array(np.log(label))

        phi = x[:, 1]
        eta = x[:, 0]
        phi = phi.reshape((-1, 1))
        eta = eta.reshape((-1, 1))

        dist_phi = distance.cdist(phi, phi, 'cityblock')
        indices = np.where(dist_phi > pi)
        temp = np.ceil((dist_phi[indices] - pi) / (2 * pi)) * (2 * pi)
        dist_phi[indices] = dist_phi[indices] - temp
        dist_eta = distance.cdist(eta, eta, 'cityblock')
        dist = np.sqrt(dist_phi ** 2 + dist_eta ** 2)

        edge_source = np.where((dist < dR) & (dist != 0))[0]
        edge_target = np.where((dist < dR) & (dist != 0))[1]

        edge_index = np.array([edge_source, edge_target])
        edge_index = torch.from_numpy(edge_index)
        edge_index = edge_index.type(torch.long)

        x = torch.from_numpy(x)
        x = x.type(torch.float32)
        label = torch.from_numpy(label)
        label = label.type(torch.float32)

        graph = Data(x=x, edge_index=edge_index, y=label)
        new_data.append(graph)
    return new_data

train_data = convert_data(train_data, dR)
valid_data = convert_data(valid_data, dR)
test_data = convert_data(test_data, dR)

lr = 1e-5
epochs = 20
batch_size = 16
hidden_dim = 256
n_layers = 4
dropout = 0.15

model = pygnn.GAT(
    in_channels=11,
    hidden_channels=hidden_dim,
    num_layers=n_layers,
    out_channels=1,
    dropout=dropout
)
model = model.to(device)


opt = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='mean')
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

best_valid_loss = float('inf')

config = {
    'lr': lr,
    'n_epochs': epochs,
    'batch_size': batch_size,
    'dropout': dropout,
    'hidden_dim': hidden_dim,
    'n_layers': n_layers
}

wandb.init(
    project='GP_MSE',
    config= config
)

for i in range(epochs):
    model.train()
    for batch_id, batch in enumerate(train_dataloader):
        opt.zero_grad()
        batch = batch.to(device)
        out = model.forward(x=batch.x, edge_index=batch.edge_index)
        out = out.squeeze()
        loss = criterion(out, batch.y)
        wandb.log({"train_loss": loss.item()})
        loss.backward()
        opt.step()

    model.eval()
    avg_valid_loss = 0.0
    with torch.no_grad():
        for v_batch_id, v_batch in enumerate(valid_dataloader):
            v_batch = v_batch.to(device)
            v_out = model.forward(x=v_batch.x, edge_index=v_batch.edge_index)
            v_out = v_out.squeeze()
            v_loss = criterion(v_out, v_batch.y)
            wandb.log({"valid_loss": v_loss.item()})
            avg_valid_loss += v_loss.item()
        avg_valid_loss /= v_batch_id
        wandb.log({"avg_valid_loss": avg_valid_loss})
        if avg_valid_loss < best_valid_loss:
            torch.save(model.state_dict(), r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\model.pt')
            best_valid_loss = avg_valid_loss

best_model = pygnn.GAT(
    in_channels=11,
    hidden_channels=hidden_dim,
    num_layers=n_layers,
    out_channels=1,
    dropout=dropout
)
best_model = best_model.to(device)

predictions = []
truths = []
model.eval()
with torch.no_grad():
    avg_test_loss = 0.0
    with torch.no_grad():
        for t_batch_id, t_batch in enumerate(test_dataloader):
            t_batch = t_batch.to(device)
            t_out = model.forward(x=t_batch.x, edge_index=t_batch.edge_index)
            t_out = t_out.squeeze()
            t_loss = criterion(t_out, t_batch.y)
            wandb.log({"test_loss": t_loss.item()})
            avg_test_loss += t_loss.item()
            predictions.append(t_out.detach().cpu().numpy())
            truths.append(batch.y.detach().cpu().numpy())
        avg_test_loss /= t_batch_id
wandb.log({"avg_test_loss": avg_test_loss})

wandb.finish()

torch.save(np.array(predictions), r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\predictions.pt')
torch.save(np.array(truths), r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\truths.pt')

