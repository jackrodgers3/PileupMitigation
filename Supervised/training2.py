import math
import sys
from tqdm import tqdm
#import wandb
from collections import OrderedDict
from timeit import default_timer as timer
import pickle
import random
import numpy as np
import scipy.stats
import argparse
from scipy.stats import binned_statistic
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import models
#import utils
#import test_physics_metrics as phym
#from test_physics_metrics import Args
import matplotlib
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import mplhep as hep
import faulthandler
import optuna
import wandb
from torch_geometric.nn.models import GAT

hep.set_style(hep.style.CMS)
wandb.login(key='8b998ffdd7e214fa724dd5cf67eafb36b111d2a7')
linewidth = 1.5
fontsize = 18
def get_args():
    parser = argparse.ArgumentParser(description='GAT Arguments')
    parser.add_argument("--data_path")
    parser.add_argument("--save_dir")
    parser.add_argument("--bs", help="batch size")
    parser.add_argument("--lr", help="learning rate")
    parser.add_argument("--hd", help="hidden dimension")
    parser.add_argument("--nl", help="num layers")
    parser.add_argument("--dr", help="dropout rate")
    parser.add_argument("--ne", help="number of epochs")
    parser.set_defaults(
        data_path = r"C:\Users\jackm\PycharmProjects\PileupMitigation\Supervised\data/",
        save_dir = r"C:\Users\jackm\PycharmProjects\PileupMitigation\Supervised\save_dir/",
        bs = 32,
        lr=3e-4,
        hd = 128,
        nl = 3,
        dr = 0.1,
        ne = 20
    )

    args = parser.parse_args()
    return args

def getResol(input):
    return (np.quantile(input, 0.84) - np.quantile(input, 0.16))/2

def getStat(input):
    return float(np.median(input)), float(getResol(input))

args = get_args()
wandb_config = {
    "dropout": args.dr,
    "hidden_dim": args.hd,
    "lr": args.lr,
    "num_layers": args.nl,
    "batch_size": args.bs
}

# wandb initialization
wandb.init(
    project='GP_Supervised_Test_Scratch',
    config=wandb_config
)
with open(args.data_path + "train5000", "rb") as fp:
    dataset_train = pickle.load(fp)
fp.close()
with open(args.data_path + "valid5000", "rb") as fp:
    dataset_valid = pickle.load(fp)
fp.close()
with open(args.data_path + "test5000", "rb") as fp:
    dataset_test = pickle.load(fp)
fp.close()

# normalizing tvt
# train
x_train_total = torch.cat([dataset_train[i].x for i in range(len(dataset_train))])
x_train_mean, x_train_std = torch.mean(x_train_total, dim=0), torch.std(x_train_total, dim=0)
y_train_total = torch.cat([dataset_train[i].y for i in range(len(dataset_train))])
y_train_mean, y_train_std = torch.mean(y_train_total, dim=0), torch.std(y_train_total, dim=0)
for graph in dataset_train:
    graph.x = (graph.x - x_train_mean) / x_train_std
    graph.y = (graph.y - y_train_mean) / y_train_std
# valid
x_valid_total = torch.cat([dataset_valid[i].x for i in range(len(dataset_valid))])
x_valid_mean, x_valid_std = torch.mean(x_valid_total, dim=0), torch.std(x_valid_total, dim=0)
y_valid_total = torch.cat([dataset_valid[i].y for i in range(len(dataset_valid))])
y_valid_mean, y_valid_std = torch.mean(y_valid_total, dim=0), torch.std(y_valid_total, dim=0)
for graph in dataset_valid:
    graph.x = (graph.x - x_valid_mean) / x_valid_std
    graph.y = (graph.y - y_valid_mean) / y_valid_std
# test
x_test_total = torch.cat([dataset_test[i].x for i in range(len(dataset_test))])
x_test_mean, x_test_std = torch.mean(x_test_total, dim=0), torch.std(x_test_total, dim=0)
y_test_total = torch.cat([dataset_test[i].y for i in range(len(dataset_test))])
y_test_mean, y_test_std = torch.mean(y_test_total, dim=0), torch.std(y_test_total, dim=0)
for graph in dataset_test:
    graph.x = (graph.x - x_test_mean) / x_test_std
    graph.y = (graph.y - y_test_mean) / y_test_std


B, I = dataset_train[0].x.shape
model = GAT(
    in_channels=I,
    hidden_channels=args.hd,
    out_channels=1,
    num_layers=args.nl,
    dropout=args.dr
)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()
train_dataloader = DataLoader(dataset_train, args.bs, shuffle=True)
valid_dataloader = DataLoader(dataset_valid, args.bs, shuffle=True)
test_dataloader = DataLoader(dataset_test, args.bs, shuffle=True)
best_validation_loss = float("inf")
for epoch in tqdm(range(args.ne)):
    model.train()
    for batch in train_dataloader:
        opt.zero_grad()
        pred = torch.sum(model.forward(batch.x, batch.edge_index)).view(-1, 1)
        label = batch.y
        label = label.type(torch.float)
        label = label.view(-1, 1)
        loss = criterion(pred, label)
        wandb.log({"train_loss": loss.item()})
        loss.backward()
        opt.step()
    with torch.no_grad():
        model.eval()
        avg_val_loss = 0.0
        val_steps = 0
        for batch in valid_dataloader:
            val_steps += 1
            pred = torch.sum(model.forward(batch.x, batch.edge_index)).view(-1, 1)
            label = batch.y
            label = label.type(torch.float)
            label = label.view(-1, 1)
            loss = criterion(pred, label)
            wandb.log({"valid_loss": loss.item()})
            avg_val_loss += loss.item()
        avg_val_loss /= val_steps
        wandb.log({"avg_valid_loss": avg_val_loss})
        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            torch.save(model.state_dict(), args.save_dir + "best_model.pth")

model.load_state_dict(torch.load(args.save_dir + "best_model.pth"))
pred_jet_masses = []
truth_jet_masses = []
with torch.no_grad():
    model.eval()
    avg_test_loss = 0.0
    test_steps = 0
    for batch in test_dataloader:
        test_steps += 1
        pred = torch.sum(model.forward(batch.x, batch.edge_index)).view(-1, 1)
        label = batch.y
        label = label.type(torch.float)
        label = label.view(-1, 1)
        loss = criterion(pred, label)
        wandb.log({"test_loss": loss})
        avg_test_loss += loss.item()
        pred_us = (y_test_std * pred) + y_test_mean
        label_us = (y_test_std * label) + y_test_mean
        pred_jet_masses.append(pred_us.item())
        truth_jet_masses.append(label_us.item())
    avg_test_loss /= test_steps
    wandb.log({"avg_test_loss": avg_test_loss})

pred_jet_masses = np.array(pred_jet_masses)
truth_jet_masses = np.array(truth_jet_masses)
rel_diff_masses = (pred_jet_masses - truth_jet_masses) / truth_jet_masses
pf_jet_masses = []
for graph in dataset_test:
    pf_jet_masses.append(graph.pf_jm.item())
pf_jet_masses = np.array(pf_jet_masses)
rel_diff_pf_masses = (pf_jet_masses - truth_jet_masses) / truth_jet_masses
N_BINS = 40
filter_mass_top = 200
filter_mass_bottom = 1.

mass_pred = np.array([m for m in pred_jet_masses if
                           filter_mass_bottom <= m <= filter_mass_top])
mass_pred_pf = np.array([m for m in pf_jet_masses if
                           filter_mass_bottom <= m <= filter_mass_top])
mass_truth = np.array([m for m in truth_jet_masses if
                           filter_mass_bottom <= m <= filter_mass_top])

mass_rel_diff_SSL = np.array([rel_diff_masses[i]
                              for i in range(len(rel_diff_masses)) if
                              filter_mass_bottom <= truth_jet_masses[i] <= filter_mass_top])


#mass_rel_diff_PF = np.array([rel_diff_pf_masses[i]
#                              for i in range(len(rel_diff_pf_masses)) if
#                              filter_mass_bottom <= truth_jet_masses[i] <= filter_mass_top])

# rel mass diff
masses_binned = binned_statistic(x=mass_truth, values=mass_truth, statistic='mean', bins=N_BINS)
ssl_masses_reldiff_means_binned = binned_statistic(x=mass_truth, values=mass_rel_diff_SSL, statistic='mean',
                                                   bins=N_BINS)
ssl_masses_reldiff_stds_binned = binned_statistic(x=mass_truth, values=mass_rel_diff_SSL, statistic='std',
                                                  bins=N_BINS)
#pf_masses_reldiff_means_binned = binned_statistic(x=mass_truth, values=rel_diff_pf_masses, statistic='mean',
#                                                   bins=N_BINS)
#pf_masses_reldiff_stds_binned = binned_statistic(x=mass_truth, values=rel_diff_pf_masses, statistic='std',
#                                                  bins=N_BINS)

fig = plt.figure(figsize=(10, 8))
plt.plot(masses_binned.statistic, np.absolute(ssl_masses_reldiff_means_binned.statistic), 'go', label='SSL')
#plt.plot(masses_binned.statistic, np.absolute(pf_masses_reldiff_means_binned.statistic), 'bo', label='PF')
plt.title('Mean mass vs. mean relative difference in mass')
plt.xlabel(r'mean mass [GeV]')
plt.ylabel(r'mean of $(m_{reco} - m_{truth})/m_{truth}$')
plt.yscale('log')
plt.legend()
plt.savefig(args.save_dir + 'mean_mass_rel_diff_comp_log.png')
plt.close()

fig = plt.figure(figsize=(10, 8))
plt.plot(masses_binned.statistic, np.absolute(ssl_masses_reldiff_means_binned.statistic), 'go', label='SSL')
#plt.plot(masses_binned.statistic, np.absolute(pf_masses_reldiff_means_binned.statistic), 'bo', label='PF')
plt.title('Mean mass vs. mean relative difference in mass')
plt.xlabel(r'mean mass [GeV]')
plt.ylabel(r'mean of $(m_{reco} - m_{truth})/m_{truth}$')
plt.legend()
plt.savefig(args.save_dir + 'mean_mass_rel_diff_comp_nolog.png')
plt.close()

fig = plt.figure(figsize=(10, 8))
plt.plot(masses_binned.statistic, np.absolute(ssl_masses_reldiff_stds_binned.statistic), 'go', label='SSL')
#plt.plot(masses_binned.statistic, np.absolute(pf_masses_reldiff_stds_binned.statistic), 'bo', label='PF')
plt.title('Mean mass vs. std relative difference in mass')
plt.xlabel(r'mean mass [GeV]')
plt.ylabel(r'std of $(m_{reco} - m_{truth})/m_{truth}$')
plt.yscale('log')
plt.legend()
plt.savefig(args.save_dir + 'std_mass_rel_diff_comp_log.png')
plt.close()

fig = plt.figure(figsize=(10, 8))
plt.plot(masses_binned.statistic, np.absolute(ssl_masses_reldiff_stds_binned.statistic), 'go', label='SSL')
#plt.plot(masses_binned.statistic, np.absolute(pf_masses_reldiff_stds_binned.statistic), 'bo', label='PF')
plt.title('Mean mass vs. std relative difference in mass')
plt.xlabel(r'mean mass [GeV]')
plt.ylabel(r'std of $(m_{reco} - m_{truth})/m_{truth}$')
plt.legend()
plt.savefig(args.save_dir + 'std_mass_rel_diff_comp_nolog.png')
plt.close()

fig = plt.figure(figsize=(10, 8))
plt.hist(mass_rel_diff_SSL, bins=40, range=(-2, 2), histtype='step', color='blue', linewidth=linewidth,
         density=True, label=r'Supervised GNN, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(mass_rel_diff_SSL)))+str(len(mass_rel_diff_SSL)))

#plt.hist(mass_rel_diff_PF, bins=40, range=(-2, 2), histtype='step', color='red', linewidth=linewidth,
#         density=True, label=r'PF, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(mass_rel_diff_PF)))+str(len(mass_rel_diff_PF)))
# plt.xlim(-1.0,1.3)
plt.xlabel(r"Jet Mass $(m_{reco} - m_{truth})/m_{truth}$")
plt.ylabel('density')
plt.ylim(0, 3.6)
plt.rc('legend', fontsize=fontsize)
plt.legend()
plt.savefig(args.save_dir + "Jet_mass_diff.pdf")
plt.close(fig)