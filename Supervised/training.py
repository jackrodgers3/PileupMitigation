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
import optuna
from scipy.stats import binned_statistic
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
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
from models import *
from sklearn.utils import compute_sample_weight
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hep.set_style(hep.style.CMS)
wandb.login(key='8b998ffdd7e214fa724dd5cf67eafb36b111d2a7')
linewidth = 1.5
fontsize = 18
def get_args():
    parser = argparse.ArgumentParser(description='GAT Arguments')
    parser.add_argument("--data_path")
    parser.add_argument("--save_dir")
    parser.add_argument("--batch_size", help="batch size")
    parser.add_argument("--lr", help="learning rate")
    parser.add_argument("--hidden_dim", help="hidden dimension")
    parser.add_argument("--dropout", help="dropout rate")
    parser.add_argument("--ne", help="number of epochs")
    parser.add_argument("--act_fn", help="activation function in decoder")
    parser.add_argument("--num_enc_layers", help="num layers in encoder")
    parser.add_argument("--num_dec_layers", help="num layers in decoder")
    parser.add_argument("--model_type", help="model type")
    parser.add_argument("--input_dim", help="input dimension")
    parser.add_argument("--output_dim", help="output dimension")
    parser.set_defaults(
        data_path = "/depot/cms/users/jprodger/PUPPI/Physics_Optimization/supervised_jet_pt/data/",
        save_dir = "/depot/cms/users/jprodger/PUPPI/Physics_Optimization/supervised_jet_pt/save_dir/",
        batch_size = 1,
        lr=1e-3,
        hidden_dim = 64,
        dropout = 0.1,
        ne = 5,
        act_fn = 'gelu',
        num_enc_layers = 3,
        num_dec_layers = 3,
        model_type = 'Gated',
        input_dim = 11,
        output_dim = 1,
    )

    return parser.parse_args()

def getResol(input):
    return (np.quantile(input, 0.84) - np.quantile(input, 0.16))/2

def getStat(input):
    return float(np.median(input)), float(getResol(input))


def train_ind():
    args = get_args()
    version = "no_norm_p_weights"
    args.save_dir = args.save_dir + version + "/"
    os.makedirs(args.save_dir, exist_ok=True)

    wandb_config = {
        "dropout": args.dropout,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "num_enc_layers": args.num_enc_layers,
        "num_dec_layers": args.num_dec_layers,
        "batch_size": args.batch_size,
        "num_epochs": args.ne,
        "act_fn": args.act_fn
    }

    # wandb initialization
    wandb.init(
        project='GP_Supervised_jet_pt_sum_debug',
        config=wandb_config,
        notes = version
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
    print("Data Loaded")
    # normalizing tvt
    # train
    '''
    x_train_total = torch.cat([dataset_train[i].x for i in range(len(dataset_train))])
    x_train_mean, x_train_std = torch.mean(x_train_total, dim=0), torch.std(x_train_total, dim=0)
    y_train_total = torch.cat([dataset_train[i].y for i in range(len(dataset_train))])
    y_train_mean, y_train_std = torch.mean(y_train_total, dim=0), torch.std(y_train_total, dim=0)
    x_train_mean[0] = 0.0
    x_train_std[0] = 1.0
    x_train_mean[1] = 0.0
    x_train_std[1] = 1.0
    for graph in dataset_train:
        graph.x = (graph.x - x_train_mean) / x_train_std
        graph.y = (graph.y - y_train_mean) / y_train_std
    # valid
    x_valid_total = torch.cat([dataset_valid[i].x for i in range(len(dataset_valid))])
    x_valid_mean, x_valid_std = torch.mean(x_valid_total, dim=0), torch.std(x_valid_total, dim=0)
    y_valid_total = torch.cat([dataset_valid[i].y for i in range(len(dataset_valid))])
    y_valid_mean, y_valid_std = torch.mean(y_valid_total, dim=0), torch.std(y_valid_total, dim=0)
    x_valid_mean[0] = 0.0
    x_valid_std[0] = 1.0
    x_valid_mean[1] = 0.0
    x_valid_std[1] = 1.0
    for graph in dataset_valid:
        graph.x = (graph.x - x_valid_mean) / x_valid_std
        graph.y = (graph.y - y_valid_mean) / y_valid_std
    # test
    x_test_total = torch.cat([dataset_test[i].x for i in range(len(dataset_test))])
    x_test_mean, x_test_std = torch.mean(x_test_total, dim=0), torch.std(x_test_total, dim=0)
    y_test_total = torch.cat([dataset_test[i].y for i in range(len(dataset_test))])
    y_test_mean, y_test_std = torch.mean(y_test_total, dim=0), torch.std(y_test_total, dim=0)
    x_test_mean[0] = 0.0
    x_test_std[0] = 1.0
    x_test_mean[1] = 0.0
    x_test_std[1] = 1.0
    for graph in dataset_test:
        graph.x = (graph.x - x_test_mean) / x_test_std
        graph.y = (graph.y - y_test_mean) / y_test_std
    
    n_bins = 50
    y_train = []
    y_valid = []
    y_test = []
    # Example: continuous target variable
    for graph in dataset_train:
        y_train.append(graph.y.item())
    # Bin the targets
    bins_train = np.linspace(min(y_train), max(y_train), num=n_bins)
    y_binned_train = np.digitize(y_train, bins_train)
    for graph in dataset_valid:
        y_valid.append(graph.y.item())
    bins_valid = np.linspace(min(y_valid), max(y_valid), num=n_bins)
    y_binned_valid = np.digitize(y_valid, bins_valid)
    for graph in dataset_test:
        y_test.append(graph.y.item())
    bins_test = np.linspace(min(y_test), max(y_test), num=n_bins)
    y_binned_test = np.digitize(y_test, bins_test)

    # Compute weights: inverse frequency
    sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_binned_train)
    sample_weights_train = sample_weights_train / np.mean(sample_weights_train)
    sample_weights_valid = compute_sample_weight(class_weight='balanced', y=y_binned_valid)
    sample_weights_valid = sample_weights_valid / np.mean(sample_weights_valid)
    sample_weights_test = compute_sample_weight(class_weight='balanced', y=y_binned_test)
    sample_weights_test = sample_weights_test / np.mean(sample_weights_test)

    sample_weights_train = torch.tensor(sample_weights_train, dtype=torch.float)
    sample_weights_valid = torch.tensor(sample_weights_valid, dtype=torch.float)
    sample_weights_test = torch.tensor(sample_weights_test, dtype=torch.float)
    for i, graph in enumerate(dataset_train):
        graph.weight = sample_weights_train[i]
    for i, graph in enumerate(dataset_valid):
        graph.weight = sample_weights_valid[i]
    for i, graph in enumerate(dataset_test):
        graph.weight = sample_weights_test[i]
    '''
    print("Graph Preprocessing complete.")
    B, I = dataset_train[0].x.shape
    args.input_dim = I
    model = GNNStack(args).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0, factor=0.9)
    criterion = nn.MSELoss()
    train_dataloader = DataLoader(dataset_train, args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset_valid, args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test, 1, shuffle=True)
    best_validation_loss = float("inf")
    for epoch in tqdm(range(args.ne)):
        model.train()
        for batch in tqdm(train_dataloader):
            batch = batch.to(device)
            opt.zero_grad()
            pred, w = model.forward(batch)
            label = batch.y
            label = label.type(torch.float)
            label = label.view(-1, 1)
            #loss = criterion(pred, label) * batch.weight
            loss = criterion(pred, label)
            loss = loss.mean()
            wandb.log({"train_loss": loss.item()})
            loss.backward()
            opt.step()
        with torch.no_grad():
            model.eval()
            avg_val_loss = 0.0
            val_steps = 0
            for batch in valid_dataloader:
                val_steps += 1
                batch = batch.to(device)
                pred, w = model.forward(batch)
                label = batch.y
                label = label.type(torch.float)
                label = label.view(-1, 1)
                #loss = criterion(pred, label) * batch.weight
                loss = criterion(pred, label)
                loss = loss.mean()
                wandb.log({"valid_loss": loss.item()})
                avg_val_loss += loss.item()
            avg_val_loss /= val_steps
            wandb.log({"avg_valid_loss": avg_val_loss})
            if avg_val_loss < best_validation_loss:
                best_validation_loss = avg_val_loss
                torch.save(model.state_dict(), args.save_dir + "best_model.pth")
        scheduler.step(avg_val_loss)

    model.load_state_dict(torch.load(args.save_dir + "best_model.pth"))
    pred_jet_pts = []
    truth_jet_pts = []
    model_weights = []
    with torch.no_grad():
        model.eval()
        avg_test_loss = 0.0
        test_steps = 0
        for batch in test_dataloader:
            test_steps += 1
            batch = batch.to(device)
            pred, w = model.forward(batch)
            label = batch.y
            label = label.type(torch.float)
            label = label.view(-1, 1)
            #loss = criterion(pred, label) * batch.weight
            loss = criterion(pred, label)
            loss = loss.mean()
            wandb.log({"test_loss": loss})
            avg_test_loss += loss.item()
            #pred_us = (y_test_std * pred) + y_test_mean
            #label_us = (y_test_std * label) + y_test_mean
            pred_us = pred
            label_us = label
            pred_jet_pts.append(pred_us.item())
            truth_jet_pts.append(label_us.item())
            model_weights.append(w.detach().cpu().numpy())
        avg_test_loss /= test_steps
        wandb.log({"avg_test_loss": avg_test_loss})

    pred_jet_pts = np.array(pred_jet_pts)
    truth_jet_pts = np.array(truth_jet_pts)
    rel_diff_pts = (pred_jet_pts - truth_jet_pts) / truth_jet_pts
    pf_jet_pts = []
    for graph in dataset_test:
        pf_jet_pts.append(graph.pf_jm.item())
    pf_jet_pts = np.array(pf_jet_pts)
    rel_diff_pf_pts = (pf_jet_pts - truth_jet_pts) / truth_jet_pts
    N_BINS = 40
    filter_pt_top = 10000
    filter_pt_bottom = 1.

    pt_pred = np.array([m for m in pred_jet_pts if
                               filter_pt_bottom <= m <= filter_pt_top])
    pt_pred_pf = np.array([m for m in pf_jet_pts if
                               filter_pt_bottom <= m <= filter_pt_top])
    pt_truth = np.array([m for m in truth_jet_pts if
                               filter_pt_bottom <= m <= filter_pt_top])

    pts_rel_diff_sup = np.array([rel_diff_pts[i]
                                  for i in range(len(rel_diff_pts)) if
                                  filter_pt_bottom <= truth_jet_pts[i] <= filter_pt_top])
    
    pts_rel_diff_pf = np.array([rel_diff_pf_pts[i]
                                  for i in range(len(rel_diff_pf_pts)) if
                                  filter_pt_bottom <= truth_jet_pts[i] <= filter_pt_top])


    #mass_rel_diff_PF = np.array([rel_diff_pf_masses[i]
    #                              for i in range(len(rel_diff_pf_masses)) if
    #                              filter_mass_bottom <= truth_jet_masses[i] <= filter_mass_top])

    # rel pt diff
    pts_binned = binned_statistic(x=pt_truth, values=pt_truth, statistic='mean', bins=N_BINS)
    sup_pts_reldiff_means_binned = binned_statistic(x=pt_truth, values=pts_rel_diff_sup, statistic='mean',
                                                       bins=N_BINS)
    sup_pts_reldiff_stds_binned = binned_statistic(x=pt_truth, values=pts_rel_diff_sup, statistic='std',
                                                      bins=N_BINS)
    pf_pts_reldiff_means_binned = binned_statistic(x=pt_truth, values = pts_rel_diff_pf, statistic = 'mean',
                                            bins = N_BINS)
    pf_pts_reldiff_stds_binned = binned_statistic(x=pt_truth, values = pts_rel_diff_pf, statistic = 'std',
                                            bins = N_BINS)
    
    #pf_masses_reldiff_means_binned = binned_statistic(x=mass_truth, values=rel_diff_pf_masses, statistic='mean',
    #                                                   bins=N_BINS)
    #pf_masses_reldiff_stds_binned = binned_statistic(x=mass_truth, values=rel_diff_pf_masses, statistic='std',
    #                                                  bins=N_BINS)
    if model_weights[0].ndim == 2:
        model_weights = np.concatenate([w for w in model_weights], axis=0)
        fig = plt.figure(figsize=(10, 8))
        plt.hist(model_weights, bins= 40, histtype='step')
        plt.yscale('log')
        plt.title('Model Weights')
        plt.savefig(args.save_dir + 'model_weights.png')
        plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(pts_binned.statistic, np.absolute(sup_pts_reldiff_means_binned.statistic), 'go', label='Supervised')
    plt.plot(pts_binned.statistic, np.absolute(pf_pts_reldiff_means_binned.statistic), 'bo', label='PF')
    plt.title('Mean pT vs. mean relative difference in pT')
    plt.xlabel(r'mean pT [GeV]')
    plt.ylabel(r'mean of $(pT_{reco} - pT_{truth})/pT_{truth}$')
    plt.yscale('log')
    plt.legend()
    plt.savefig(args.save_dir + 'mean_pt_rel_diff_comp_log.png')
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(pts_binned.statistic, sup_pts_reldiff_means_binned.statistic, 'go', label='Supervised')
    plt.plot(pts_binned.statistic, pf_pts_reldiff_means_binned.statistic, 'bo', label='PF')
    plt.title('Mean pT vs. mean relative difference in pT')
    plt.xlabel(r'mean pT [GeV]')
    plt.ylabel(r'mean of $(pT_{reco} - pT_{truth})/pT_{truth}$')
    plt.legend()
    plt.savefig(args.save_dir + 'mean_pt_rel_diff_comp_nolog.png')
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(pts_binned.statistic, np.absolute(sup_pts_reldiff_stds_binned.statistic), 'go', label='SSL')
    plt.plot(pts_binned.statistic, np.absolute(pf_pts_reldiff_stds_binned.statistic), 'bo', label='PF')
    plt.title('Mean pT vs. std relative difference in pT')
    plt.xlabel(r'mean pT [GeV]')
    plt.ylabel(r'std of $(pT_{reco} - pT_{truth})/pT_{truth}$')
    plt.yscale('log')
    plt.legend()
    plt.savefig(args.save_dir + 'std_pt_rel_diff_comp_log.png')
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(pts_binned.statistic, sup_pts_reldiff_stds_binned.statistic, 'go', label='SSL')
    plt.plot(pts_binned.statistic, pf_pts_reldiff_stds_binned.statistic, 'bo', label='PF')
    plt.title('Mean pT vs. std relative difference in pT')
    plt.xlabel(r'mean pT [GeV]')
    plt.ylabel(r'std of $(pT_{reco} - pT_{truth})/pT_{truth}$')
    plt.legend()
    plt.savefig(args.save_dir + 'std_pt_rel_diff_comp_nolog.png')
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    plt.hist(pts_rel_diff_sup, bins=40, range=(-2, 2), histtype='step', color='blue', linewidth=linewidth,
             density=True, label=r'Supervised GNN, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(pts_rel_diff_sup)))+str(len(pts_rel_diff_sup)))
    plt.hist(pts_rel_diff_pf, bins=40, range=(-2, 2), histtype='step', color='green', linewidth=linewidth,
             density=True, label=r'PF, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(pts_rel_diff_pf)))+str(len(pts_rel_diff_pf)))
    # plt.xlim(-1.0,1.3)
    plt.xlabel(r"Jet pT $(pT_{reco} - pT_{truth})/pT_{truth}$")
    plt.ylabel('density')
    #plt.ylim(0, 3.6)
    plt.rc('legend', fontsize=fontsize)
    plt.legend()
    plt.savefig(args.save_dir + "Jet_pt_diff.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    plt.hist(pt_pred, bins=40, histtype='step', color='blue', linewidth=linewidth,
             density=True, label=r'Supervised GNN, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(
            *(getStat(pt_pred))) + str(len(pt_pred)))
    plt.hist(pt_truth, bins=40, histtype='step', color='red', linewidth=linewidth,
             density=True, label=r'Truth, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(
            *(getStat(pt_truth))) + str(len(pt_truth)))
    plt.hist(pt_pred_pf, bins=40, histtype='step', color='green', linewidth=linewidth,
             density=True, label=r'PF, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(
            *(getStat(pt_pred_pf))) + str(len(pt_pred_pf)))

    # plt.hist(mass_rel_diff_PF, bins=40, range=(-2, 2), histtype='step', color='red', linewidth=linewidth,
    #         density=True, label=r'PF, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(mass_rel_diff_PF)))+str(len(mass_rel_diff_PF)))
    # plt.xlim(-1.0,1.3)
    plt.xlabel(r"Jet pT")
    plt.ylabel('density')
    #plt.ylim(0, 3.6)
    plt.rc('legend', fontsize=fontsize)
    plt.legend()
    plt.savefig(args.save_dir + "Jet_pt.pdf")
    plt.close(fig)

    mu, sig = getStat(pts_rel_diff_sup)
    vm = sig / (1 - mu)
    wandb.log({"val_metric": vm})
    wandb.finish()
    return vm


def train(trial):
    args = get_args()

    args.save_dir = args.save_dir + f"trial{str(trial.number)}/"
    os.makedirs(args.save_dir, exist_ok=True)
    args.lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    args.hidden_dim = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    args.num_enc_layers = trial.suggest_categorical("num_enc_layers", [2, 3, 4, 5])
    args.num_dec_layers = trial.suggest_categorical("num_dec_layers", [2, 3, 4, 5])
    args.dropout = trial.suggest_float("dropout", 0, 0.3)
    args.act_fn = trial.suggest_categorical("act_fn", ["gelu", "relu", "leakyrelu"])


    wandb_config = {
        "dropout": args.dropout,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "num_enc_layers": args.num_enc_layers,
        "num_dec_layers": args.num_dec_layers,
        "batch_size": args.batch_size,
        "num_epochs": args.ne,
        "act_fn": args.act_fn,
        "trial_num": trial.number
    }

    # wandb initialization
    wandb.init(
        project='GP_Supervised_jet_pt_sum',
        config=wandb_config
    )

    with open(args.data_path + "train10000", "rb") as fp:
        dataset_train = pickle.load(fp)
    fp.close()
    with open(args.data_path + "valid10000", "rb") as fp:
        dataset_valid = pickle.load(fp)
    fp.close()
    with open(args.data_path + "test10000", "rb") as fp:
        dataset_test = pickle.load(fp)
    fp.close()
    print("Data Loaded")
    # normalizing tvt
    # train
    x_train_total = torch.cat([dataset_train[i].x for i in range(len(dataset_train))])
    x_train_mean, x_train_std = torch.mean(x_train_total, dim=0), torch.std(x_train_total, dim=0)
    y_train_total = torch.cat([dataset_train[i].y for i in range(len(dataset_train))])
    y_train_mean, y_train_std = torch.mean(y_train_total, dim=0), torch.std(y_train_total, dim=0)
    x_train_mean[0] = 0.0
    x_train_std[0] = 1.0
    x_train_mean[1] = 0.0
    x_train_std[1] = 1.0
    for graph in dataset_train:
        graph.x = (graph.x - x_train_mean) / x_train_std
        graph.y = (graph.y - y_train_mean) / y_train_std
    # valid
    x_valid_total = torch.cat([dataset_valid[i].x for i in range(len(dataset_valid))])
    x_valid_mean, x_valid_std = torch.mean(x_valid_total, dim=0), torch.std(x_valid_total, dim=0)
    y_valid_total = torch.cat([dataset_valid[i].y for i in range(len(dataset_valid))])
    y_valid_mean, y_valid_std = torch.mean(y_valid_total, dim=0), torch.std(y_valid_total, dim=0)
    x_valid_mean[0] = 0.0
    x_valid_std[0] = 1.0
    x_valid_mean[1] = 0.0
    x_valid_std[1] = 1.0
    for graph in dataset_valid:
        graph.x = (graph.x - x_valid_mean) / x_valid_std
        graph.y = (graph.y - y_valid_mean) / y_valid_std
    # test
    x_test_total = torch.cat([dataset_test[i].x for i in range(len(dataset_test))])
    x_test_mean, x_test_std = torch.mean(x_test_total, dim=0), torch.std(x_test_total, dim=0)
    y_test_total = torch.cat([dataset_test[i].y for i in range(len(dataset_test))])
    y_test_mean, y_test_std = torch.mean(y_test_total, dim=0), torch.std(y_test_total, dim=0)
    x_test_mean[0] = 0.0
    x_test_std[0] = 1.0
    x_test_mean[1] = 0.0
    x_test_std[1] = 1.0
    for graph in dataset_test:
        graph.x = (graph.x - x_test_mean) / x_test_std
        graph.y = (graph.y - y_test_mean) / y_test_std
    n_bins = 50
    y_train = []
    y_valid = []
    y_test = []
    # Example: continuous target variable
    for graph in dataset_train:
        y_train.append(graph.y.item())
    # Bin the targets
    bins_train = np.linspace(min(y_train), max(y_train), num=n_bins)
    y_binned_train = np.digitize(y_train, bins_train)
    for graph in dataset_valid:
        y_valid.append(graph.y.item())
    bins_valid = np.linspace(min(y_valid), max(y_valid), num=n_bins)
    y_binned_valid = np.digitize(y_valid, bins_valid)
    for graph in dataset_test:
        y_test.append(graph.y.item())
    bins_test = np.linspace(min(y_test), max(y_test), num=n_bins)
    y_binned_test = np.digitize(y_test, bins_test)

    # Compute weights: inverse frequency
    sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_binned_train)
    sample_weights_train = sample_weights_train / np.mean(sample_weights_train)
    sample_weights_valid = compute_sample_weight(class_weight='balanced', y=y_binned_valid)
    sample_weights_valid = sample_weights_valid / np.mean(sample_weights_valid)
    sample_weights_test = compute_sample_weight(class_weight='balanced', y=y_binned_test)
    sample_weights_test = sample_weights_test / np.mean(sample_weights_test)

    sample_weights_train = torch.tensor(sample_weights_train, dtype=torch.float)
    sample_weights_valid = torch.tensor(sample_weights_valid, dtype=torch.float)
    sample_weights_test = torch.tensor(sample_weights_test, dtype=torch.float)
    for i, graph in enumerate(dataset_train):
        graph.weight = sample_weights_train[i]
    for i, graph in enumerate(dataset_valid):
        graph.weight = sample_weights_valid[i]
    for i, graph in enumerate(dataset_test):
        graph.weight = sample_weights_test[i]
    print("Graph Preprocessing complete.")

    B, I = dataset_train[0].x.shape
    args.input_dim = I
    model = GNNStack(args).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.9)
    criterion = nn.MSELoss()
    train_dataloader = DataLoader(dataset_train, args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset_valid, args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test, 1, shuffle=True)
    best_validation_loss = float("inf")
    for epoch in tqdm(range(args.ne)):
        model.train()
        for batch in tqdm(train_dataloader):
            batch = batch.to(device)
            opt.zero_grad()
            pred = model.forward(batch)
            label = batch.y
            label = label.type(torch.float)
            label = label.view(-1, 1)
            loss = criterion(pred, label) * batch.weight
            loss = loss.mean()
            wandb.log({"train_loss": loss.item()})
            loss.backward()
            opt.step()
        with torch.no_grad():
            model.eval()
            avg_val_loss = 0.0
            val_steps = 0
            for batch in valid_dataloader:
                val_steps += 1
                batch = batch.to(device)
                pred = model.forward(batch)
                label = batch.y
                label = label.type(torch.float)
                label = label.view(-1, 1)
                loss = criterion(pred, label) * batch.weight
                loss = loss.mean()
                wandb.log({"valid_loss": loss.item()})
                avg_val_loss += loss.item()
            avg_val_loss /= val_steps
            wandb.log({"avg_valid_loss": avg_val_loss})
            if avg_val_loss < best_validation_loss:
                best_validation_loss = avg_val_loss
                torch.save(model.state_dict(), args.save_dir + "best_model.pth")
        scheduler.step(avg_val_loss)

    model.load_state_dict(torch.load(args.save_dir + "best_model.pth"))
    pred_jet_pts = []
    truth_jet_pts = []
    with torch.no_grad():
        model.eval()
        avg_test_loss = 0.0
        test_steps = 0
        for batch in test_dataloader:
            test_steps += 1
            batch = batch.to(device)
            pred = model.forward(batch)
            label = batch.y
            label = label.type(torch.float)
            label = label.view(-1, 1)
            loss = criterion(pred, label) * batch.weight
            loss = loss.mean()
            wandb.log({"test_loss": loss})
            avg_test_loss += loss.item()
            #pred_us = (y_test_std * pred) + y_test_mean
            #label_us = (y_test_std * label) + y_test_mean
            #pred_jet_pts.append(pred_us.item())
            #truth_jet_pts.append(label_us.item())
            pred_jet_pts.append(pred.item())
            truth_jet_pts.append(label.item())
        avg_test_loss /= test_steps
        wandb.log({"avg_test_loss": avg_test_loss})

    pred_jet_pts = np.array(pred_jet_pts)
    truth_jet_pts = np.array(truth_jet_pts)
    rel_diff_pts = (pred_jet_pts - truth_jet_pts) / truth_jet_pts
    pf_jet_pts = []
    for graph in dataset_test:
        pf_jet_pts.append(graph.pf_jm.item())
    pf_jet_pts = np.array(pf_jet_pts)
    rel_diff_pf_pts = (pf_jet_pts - truth_jet_pts) / truth_jet_pts
    N_BINS = 40
    filter_pt_top = 10000
    filter_pt_bottom = 1.

    pt_pred = np.array([m for m in pred_jet_pts if
                               filter_pt_bottom <= m <= filter_pt_top])
    pt_pred_pf = np.array([m for m in pf_jet_pts if
                               filter_pt_bottom <= m <= filter_pt_top])
    pt_truth = np.array([m for m in truth_jet_pts if
                               filter_pt_bottom <= m <= filter_pt_top])

    pts_rel_diff_sup = np.array([rel_diff_pts[i]
                                  for i in range(len(rel_diff_pts)) if
                                  filter_pt_bottom <= truth_jet_pts[i] <= filter_pt_top])


    #mass_rel_diff_PF = np.array([rel_diff_pf_masses[i]
    #                              for i in range(len(rel_diff_pf_masses)) if
    #                              filter_mass_bottom <= truth_jet_masses[i] <= filter_mass_top])

    # rel pt diff
    pts_binned = binned_statistic(x=pt_truth, values=pt_truth, statistic='mean', bins=N_BINS)
    sup_pts_reldiff_means_binned = binned_statistic(x=pt_truth, values=pts_rel_diff_sup, statistic='mean',
                                                       bins=N_BINS)
    sup_pts_reldiff_stds_binned = binned_statistic(x=pt_truth, values=pts_rel_diff_sup, statistic='std',
                                                      bins=N_BINS)
    #pf_masses_reldiff_means_binned = binned_statistic(x=mass_truth, values=rel_diff_pf_masses, statistic='mean',
    #                                                   bins=N_BINS)
    #pf_masses_reldiff_stds_binned = binned_statistic(x=mass_truth, values=rel_diff_pf_masses, statistic='std',
    #                                                  bins=N_BINS)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(pts_binned.statistic, np.absolute(sup_pts_reldiff_means_binned.statistic), 'go', label='Supervised')
    #plt.plot(masses_binned.statistic, np.absolute(pf_masses_reldiff_means_binned.statistic), 'bo', label='PF')
    plt.title('Mean pT vs. mean relative difference in pT')
    plt.xlabel(r'mean pT [GeV]')
    plt.ylabel(r'mean of $(pT_{reco} - pT_{truth})/pT_{truth}$')
    plt.yscale('log')
    plt.legend()
    plt.savefig(args.save_dir + 'mean_pt_rel_diff_comp_log.png')
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(pts_binned.statistic, sup_pts_reldiff_means_binned.statistic, 'go', label='Supervised')
    #plt.plot(masses_binned.statistic, np.absolute(pf_masses_reldiff_means_binned.statistic), 'bo', label='PF')
    plt.title('Mean pT vs. mean relative difference in pT')
    plt.xlabel(r'mean pT [GeV]')
    plt.ylabel(r'mean of $(pT_{reco} - pT_{truth})/pT_{truth}$')
    plt.legend()
    plt.savefig(args.save_dir + 'mean_pt_rel_diff_comp_nolog.png')
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(pts_binned.statistic, np.absolute(sup_pts_reldiff_stds_binned.statistic), 'go', label='SSL')
    #plt.plot(masses_binned.statistic, np.absolute(pf_masses_reldiff_stds_binned.statistic), 'bo', label='PF')
    plt.title('Mean pT vs. std relative difference in pT')
    plt.xlabel(r'mean pT [GeV]')
    plt.ylabel(r'std of $(pT_{reco} - pT_{truth})/pT_{truth}$')
    plt.yscale('log')
    plt.legend()
    plt.savefig(args.save_dir + 'std_pt_rel_diff_comp_log.png')
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(pts_binned.statistic, sup_pts_reldiff_stds_binned.statistic, 'go', label='SSL')
    #plt.plot(masses_binned.statistic, np.absolute(pf_masses_reldiff_stds_binned.statistic), 'bo', label='PF')
    plt.title('Mean pT vs. std relative difference in pT')
    plt.xlabel(r'mean pT [GeV]')
    plt.ylabel(r'std of $(pT_{reco} - pT_{truth})/pT_{truth}$')
    plt.legend()
    plt.savefig(args.save_dir + 'std_pt_rel_diff_comp_nolog.png')
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    plt.hist(pts_rel_diff_sup, bins=40, range=(-2, 2), histtype='step', color='blue', linewidth=linewidth,
             density=True, label=r'Supervised GNN, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(pts_rel_diff_sup)))+str(len(pts_rel_diff_sup)))

    #plt.hist(mass_rel_diff_PF, bins=40, range=(-2, 2), histtype='step', color='red', linewidth=linewidth,
    #         density=True, label=r'PF, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(mass_rel_diff_PF)))+str(len(mass_rel_diff_PF)))
    # plt.xlim(-1.0,1.3)
    plt.xlabel(r"Jet pT $(pT_{reco} - pT_{truth})/pT_{truth}$")
    plt.ylabel('density')
    #plt.ylim(0, 3.6)
    plt.rc('legend', fontsize=fontsize)
    plt.legend()
    plt.savefig(args.save_dir + "Jet_pt_diff.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    plt.hist(pt_pred, bins=40, histtype='step', color='blue', linewidth=linewidth,
             density=True, label=r'Supervised GNN, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(
            *(getStat(pt_pred))) + str(len(pt_pred)))
    plt.hist(pt_truth, bins=40, histtype='step', color='red', linewidth=linewidth,
             density=True, label=r'Truth, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(
            *(getStat(pt_truth))) + str(len(pt_truth)))

    # plt.hist(mass_rel_diff_PF, bins=40, range=(-2, 2), histtype='step', color='red', linewidth=linewidth,
    #         density=True, label=r'PF, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(mass_rel_diff_PF)))+str(len(mass_rel_diff_PF)))
    # plt.xlim(-1.0,1.3)
    plt.xlabel(r"Jet pT")
    plt.ylabel('density')
    #plt.ylim(0, 3.6)
    plt.rc('legend', fontsize=fontsize)
    plt.legend()
    plt.savefig(args.save_dir + "Jet_pt.pdf")
    plt.close(fig)

    mu, sig = getStat(pts_rel_diff_sup)
    vm = sig / (1 - mu)
    wandb.log({"val_metric": vm})
    wandb.finish()
    return vm

def tune():
    study = optuna.create_study(study_name='Bayesian Optimization with Optuna for Pileup Mitigation', direction="minimize")
    study.optimize(train, n_trials=30)

train_ind()