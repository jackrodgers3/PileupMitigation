import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
import torch_geometric
from torch_geometric.data import Dataset, DataLoader
from models import GNN
import sys
import itertools
import pickle
import numpy as np
import torch.optim as optim
import math
import pandas as pd
from pyjet import DTYPE_PTEPM, cluster
import os
import argparse
import wandb
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')

    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--lr', type=float,
                        help='learning rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--data_dir', type=str,
                        help='directory to load graph datasets from')
    parser.add_argument('--save_dir', type=str,
                        help='directory to save trained model and plots')
    parser.add_argument('--act_fn', type=str,
                        help = 'activation function')
    parser.add_argument('--n_layers', type=int,
                        help='number of gcn layers')
    parser.add_argument('--standardize', type=bool,
                        help='standardization argument')
    parser.add_argument('--batch_size', type=int,
                        help='batch size')

    parser.set_defaults(
                        hidden_dim=128,
                        dropout=0.1,
                        lr = 1e-3,
                        epochs = 10,
                        data_dir = r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\jet_level_regression\dat/',
                        save_dir=r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\jet_level_regression\output/',
                        act_fn = 'relu',
                        n_layers = 3,
                        standardize = True,
                        batch_size = 8
                        )

    return parser.parse_args()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def standardize(dataset):
    x0 = []
    x1 = []
    x2 = []
    j_level_pt = []
    # making standardization params
    for i in range(len(dataset)):
        num_nodes, feature_size = dataset[i].x.shape
        j_level_pt.append(dataset[i].y.item())
        for j in range(num_nodes):
            x0.append(dataset[i].x[j][0].item())
            x1.append(dataset[i].x[j][1].item())
            x2.append(dataset[i].x[j][2].item())
    x0 = np.array(x0)
    x1 = np.array(x1)
    x2 = np.array(x2)
    jlp = np.array(j_level_pt)

    x0_m = np.mean(x0)
    x1_m = np.mean(x1)
    x2_m = np.mean(x2)
    jlp_m = np.mean(jlp)

    x0_s = np.std(x0)
    x1_s = np.std(x1)
    x2_s = np.std(x2)
    jlp_s = np.std(jlp)

    # applying standardization
    for i in range(len(dataset)):
        num_nodes, feature_size = dataset[i].x.shape
        dataset[i].y[0] = (dataset[i].y.item() - jlp_m) / jlp_s
        for j in range(num_nodes):
            dataset[i].x[j][0] = (dataset[i].x[j][0].item() - x0_m) / x0_s
            dataset[i].x[j][1] = (dataset[i].x[j][1].item() - x1_m) / x1_s
            dataset[i].x[j][2] = (dataset[i].x[j][2].item() - x2_m) / x2_s

    return dataset, (jlp_m, jlp_s)


def train():
    args = arg_parse()
    with open(args.data_dir + 'train2800', 'rb') as f:
        dataset_train = pickle.load(f)
    f.close()
    with open(args.data_dir + 'valid600', 'rb') as f:
        dataset_valid = pickle.load(f)
    f.close()
    with open(args.data_dir + 'test600', 'rb') as f:
        dataset_test = pickle.load(f)
    f.close()

    # standardization
    dataset_train, _ = standardize(dataset_train)
    dataset_valid, _ = standardize(dataset_valid)
    dataset_test, standard_params = standardize(dataset_test)

    config = {"hidden_dim": args.hidden_dim, "batch_size": args.batch_size}

    wandb.init(
        project='GP_JETLEVEL',
        config=config
    )

    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    predictions = []
    truths = []

    model = GNN(11, args.hidden_dim, args.dropout, args.act_fn)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=15, gamma=0.999)
    criterion = nn.MSELoss(reduction='mean')

    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        print(f'EPOCH {epoch}\n')
        model.train()
        for batch_id, batch in tqdm(enumerate(train_dataloader), desc='TRAINING'):
            optimizer.zero_grad()
            batch = batch.to(device)
            y_pred = model(batch.x, batch.edge_index, batch.batch).squeeze(dim=-1)
            loss = criterion(y_pred, batch.y)
            wandb.log({"train_loss": loss.item()})
            loss.backward()
            optimizer.step()
            scheduler.step()
        model.eval()
        avg_valid_loss = 0
        with torch.no_grad():
            for valid_batch_id, valid_batch in tqdm(enumerate(valid_dataloader), desc='VALIDATING'):
                valid_batch = valid_batch.to(device)
                y_pred_valid = model(valid_batch.x, valid_batch.edge_index, valid_batch.batch).squeeze(dim=-1)
                valid_loss = criterion(y_pred_valid, valid_batch.y)
                wandb.log({"valid_loss": valid_loss.item()})
                avg_valid_loss += valid_loss.item()

        avg_valid_loss /= (valid_batch_id + 1)
        wandb.log({"avg_valid_loss": avg_valid_loss})

        if avg_valid_loss > best_valid_loss:
            break

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), args.save_dir + 'best_model.pt')

    best_model = GNN(11, args.hidden_dim, args.dropout, args.act_fn)
    best_model.load_state_dict(torch.load(args.save_dir + 'best_model.pt'))
    best_model.eval()
    avg_test_loss = 0
    with torch.no_grad():
        for test_batch_id, test_batch in tqdm(enumerate(test_dataloader), desc='TESTING'):
            test_batch = test_batch.to(device)
            y_pred_test = model(test_batch.x, test_batch.edge_index, test_batch.batch).squeeze(dim=-1)
            predictions.append(y_pred_test.detach().cpu().numpy())
            truths.append(test_batch.y.detach().cpu().numpy())
            test_loss = criterion(y_pred_test, test_batch.y)
            wandb.log({"test_loss": test_loss.item()})
            avg_test_loss += test_loss.item()

    avg_test_loss /= (test_batch_id + 1)
    wandb.log({"avg_test_loss": avg_test_loss})
    torch.save(truths, args.save_dir + 'truths.pt')
    torch.save(predictions, args.save_dir + 'predictions.pt')

    plot(save_dir=args.save_dir, sparams=standard_params)

    wandb.finish()


def plot(save_dir, sparams):
    truths = torch.load(save_dir + 'truths.pt')
    predictions = torch.load(save_dir + 'predictions.pt')
    mu, sig = sparams
    # plotting
    total_predictions = []
    total_truths = []
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            total_predictions.append((predictions[i][j]*sig) + mu)
            total_truths.append((truths[i][j]*sig) + mu)

    n_bins = 40
    fig = plt.figure(figsize=(10, 8))

    _, bins, _ = plt.hist(total_truths, bins=n_bins, histtype='step', label='truth')
    _ = plt.hist(total_predictions, bins=bins, histtype='step', label='pred')
    plt.legend()
    plt.yscale('log')
    plt.savefig(save_dir + 'hist.png')
    plt.close(fig=fig)


if __name__ == '__main__':
    train()