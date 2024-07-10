import sys
import itertools
import wandb
import torch
import pickle
import numpy as np
import models
import torch.optim as optim
import torch.nn as nn
import math
import pandas as pd
from tqdm import tqdm
from pyjet import DTYPE_PTEPM, cluster
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clusterJets(pt, eta, phi, ptcut=0.5, deltaR=0.8):
    """
    cluster the jets based on the array of pt, eta, phi,
    of all particles (masses are assumed to be zero),
    with pyjet clustering algo
    """
    # cleaning zero pt-ed objects
    pt_wptcut = pt[pt > ptcut]
    eta_wptcut = eta[pt > ptcut]
    phi_wptcut = phi[pt > ptcut]
    mass_wptcut = np.zeros(pt_wptcut.shape[0])

    event = np.column_stack((pt_wptcut, eta_wptcut, phi_wptcut, mass_wptcut))
    event.dtype = DTYPE_PTEPM
    sequence = cluster(event, R=deltaR, p=-1)
    jets = sequence.inclusive_jets(ptmin=150)
    #charged only
    #jets = sequence.inclusive_jets(ptmin=20)

    return jets

def deltaPhi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    return dphi


def deltaR(eta1, phi1, eta2, phi2):
    """
    calculate the deltaR between two jets/particles
    """
    deta = eta1 - eta2
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    return np.hypot(deta, dphi)

def deltaRJet(jet1, jet2):
    """
    calculate the deltaR of the two PseudoJet
    """
    return deltaR(jet1.eta, jet1.phi, jet2.eta, jet2.phi)

def matchJets(jets_truth, jets_reco, dRcut=0.1):
    """
    match the jets in jets_reco to jets_truth,
    based on the deltaR
    """
    matched_indices = []

    jets_truth_indices = list(range(len(jets_truth)))
    jets_reco_indices = list(range(len(jets_reco)))

    for ijet_reco in jets_reco_indices:
        for ijet_truth in jets_truth_indices:
            # print("deltR between {} and {} is {}".format(ijet_truth, ijet_reco, deltaRJet(jets_truth[ijet_truth], jets_reco[ijet_reco])))
            if deltaRJet(jets_truth[ijet_truth], jets_reco[ijet_reco]) < dRcut:
                matched_indices.append((ijet_truth, ijet_reco))
                jets_truth_indices.remove(ijet_truth)
                break

    return matched_indices


def Part_in_jet(pt, eta, phi, jet):
    part_in_jet = 0
    constituents = jet.constituents_array()
    for constit in constituents:
        if ((abs(pt - constit[0]) < 0.001) & (abs(eta - constit[1]) < 0.001) & (abs(phi - constit[2]) < 0.001)):
            part_in_jet = 1
            return part_in_jet

    return part_in_jet


def get_jetidx(pt, eta, phi, jets):
    jetidx = []
    for i in range(len(pt)):
        if (len(jets) == 0):
            jetidx.append(-1)
            continue
        for j in range(len(jets)):
            PartJetFlag = Part_in_jet(pt[i], eta[i], phi[i], jets[j])
            Jetidx_ = -1
            if PartJetFlag > 0:
                Jetidx_ = j
                break
        jetidx.append(Jetidx_)

    return jetidx


def convert_data_to_jet(dataset, standardize=False):
    reco_jet_pts = []
    reco_jet_etas = []
    reco_jet_phis = []

    truth_jet_pts = []
    truth_jet_etas = []
    truth_jet_phis = []

    event_nums = []
    node_nums = []

    data_list = []

    df = pd.DataFrame(columns=['node_num', 'event_num', 'pt', 'eta', 'phi', 'pt_truth', 'eta_truth', 'phi_truth'])

    counter = 0
    # compile jet level data data
    for i in range(len(dataset)):
        # for i in range(100):
        if dataset[i - 1].event_num < dataset[i].event_num and i != 0:
            counter = 0
        elif i == 0:
            counter = 0
        else:
            counter += 1
        pt = np.array(dataset[i].x[:, 2].cpu().detach())
        eta = np.array(dataset[i].x[:, 0].cpu().detach())
        phi = np.array(dataset[i].x[:, 1].cpu().detach())

        pt_truth = np.array(dataset[i].GenPart_nump[:, 2].cpu().detach())
        eta_truth = np.array(dataset[i].GenPart_nump[:, 0].cpu().detach())
        phi_truth = np.array(dataset[i].GenPart_nump[:, 1].cpu().detach())

        jet_reco = clusterJets(pt, eta, phi)
        jet_truth = clusterJets(pt_truth, eta_truth, phi_truth)

        reco_jet_pts.append(jet_reco[0].pt)
        reco_jet_etas.append(jet_reco[0].eta)
        reco_jet_phis.append(jet_reco[0].phi)

        truth_jet_pts.append(jet_truth[0].pt)
        truth_jet_etas.append(jet_truth[0].eta)
        truth_jet_phis.append(jet_truth[0].phi)

        event_nums.append(dataset[i].event_num)
        node_nums.append(counter)

    df['node_num'] = node_nums
    df['event_num'] = event_nums
    df['pt'] = reco_jet_pts
    df['eta'] = reco_jet_etas
    df['phi'] = reco_jet_phis
    df['pt_truth'] = truth_jet_pts
    df['eta_truth'] = truth_jet_etas
    df['phi_truth'] = truth_jet_phis

    if standardize:
        df['pt'] = df['pt'].transform(lambda x: math.pow(x, 0.25))
        df['pt_truth'] = df['pt_truth'].transform(lambda x: math.pow(x, 0.25))

    n_unique_events = len(pd.unique(df['event_num']))

    # generating jet level regression graphs

    for i in range(n_unique_events):
        cur_df = df[df['event_num'] == i]
        # make node features
        node_features = cur_df.drop(cur_df.loc[:, ['node_num']], axis=1).drop(
            cur_df.loc[:, ['event_num']], axis=1).drop(
            cur_df.loc[:, ['pt_truth']], axis=1).drop(
            cur_df.loc[:, ['eta_truth']], axis=1).drop(
            cur_df.loc[:, ['phi_truth']], axis=1).to_numpy()
        node_features = torch.from_numpy(node_features)
        node_features = node_features.type(torch.float32)

        # make node labels
        label = cur_df.loc[:, ['pt_truth']].to_numpy()
        label = torch.from_numpy(label).view(-1)
        label = label.type(torch.float32)

        # make edge index
        node_nums = cur_df['node_num'].values
        permutations = list(itertools.permutations(node_nums, 2))
        edges_source = [e[0] for e in permutations]
        edges_target = [e[1] for e in permutations]
        edge_index = np.array([edges_source, edges_target])
        edge_index = torch.from_numpy(edge_index)
        edge_index = edge_index.type(torch.long)
        graph = Data(x=node_features, edge_index=edge_index, y=label)
        data_list.append(graph)
    return data_list


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--num_enc_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--num_dec_layers', type=int,
                        help='Number of decoder layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--lr', type=float,
                        help='learning rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--pulevel', type=int,
                        help='pileup level for the dataset')
    parser.add_argument('--lamb', type=float,
                        help='lambda for domain adaptation')
    parser.add_argument('--save_dir', type=str,
                        help='directory to save trained model and plots')
    parser.add_argument('--jet_type', type=str,
                        help='jet type to cluster')
    parser.add_argument('--act_fn', type=str,
                        help = 'activation function')

    parser.set_defaults(model_type='Gated',
                        num_enc_layers=5,
                        num_dec_layers = 5,
                        hidden_dim=64,
                        dropout=0.1,
                        lr = 3e-4,
                        epochs = 10,
                        pulevel=80,
                        save_dir=r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\output/',
                        jet_type = "W",
                        act_fn = 'relu'
                        )

    return parser.parse_args()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train():
    args = arg_parse()
    with open(r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\data\train2800', 'rb') as f:
        dataset_train = pickle.load(f)
    f.close()

    with open(r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\data\valid600', 'rb') as f:
        dataset_valid = pickle.load(f)
    f.close()

    with open(r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\data\test600', 'rb') as f:
        dataset_test = pickle.load(f)
    f.close()

    standardize = True

    jet_dataset_train = convert_data_to_jet(dataset_train, standardize=standardize)
    jet_dataset_valid = convert_data_to_jet(dataset_valid, standardize=standardize)
    jet_dataset_test = convert_data_to_jet(dataset_test, standardize=standardize)

    train_dataloader = DataLoader(jet_dataset_train)
    valid_dataloader = DataLoader(jet_dataset_valid)
    test_dataloader = DataLoader(jet_dataset_test)

    save_dir = args.save_dir

    config = {
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "n_epochs": args.epochs
    }
    in_channels = 3
    out_channels = 1
    wandb.init(
        project='jet_level_prediction',
        config= config,
        notes = "trying gated model"
    )
    model = models.GCN(in_channels=in_channels, hidden_channels=args.hidden_dim, out_channels=out_channels,
                       dropout=args.dropout)
    '''
    model = models.GNNStack(
        input_dim=in_channels,
        output_dim=out_channels,
        args=args
    )
    '''

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=15, gamma=0.99)
    criterion = nn.MSELoss(reduction='mean')

    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        print(f'EPOCH {epoch}\n')
        model.train()
        for batch_id, batch in tqdm(enumerate(train_dataloader), desc='TRAINING'):
            optimizer.zero_grad()
            batch = batch.to(device)
            wandb.log({"cur_lr": get_lr(optimizer)})
            y_pred = model(batch.x, batch.edge_index).squeeze(dim=-1)
            label = batch.y
            loss = criterion(y_pred, label)
            wandb.log({"train_loss": loss.item()})
            loss.backward()
            optimizer.step()
            scheduler.step(batch_id)
        model.eval()
        avg_valid_loss = 0
        with torch.no_grad():
            for valid_batch_id, valid_batch in tqdm(enumerate(valid_dataloader), desc='VALIDATING'):
                valid_batch = valid_batch.to(device)
                y_pred_valid = model(valid_batch.x, valid_batch.edge_index).squeeze(dim=-1)
                valid_label = valid_batch.y
                valid_loss = criterion(y_pred_valid, valid_label)
                avg_valid_loss += valid_loss.item()
                wandb.log({"valid_loss": valid_loss.item()})

        avg_valid_loss /= (valid_batch_id + 1)
        wandb.log({"avg_valid_loss": avg_valid_loss})

        if avg_valid_loss > best_valid_loss:
            break

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), save_dir + 'best_model.pt')

    best_model = models.GCN(in_channels=in_channels, hidden_channels=args.hidden_dim, out_channels=out_channels,
                       dropout=args.dropout)
    best_model.load_state_dict(torch.load(save_dir + 'best_model.pt'))
    predictions = []
    pf_recos = []
    truths = []
    model.eval()
    avg_test_loss = 0
    with torch.no_grad():
        for test_batch_id, test_batch in tqdm(enumerate(test_dataloader), desc='TESTING'):
            test_batch = test_batch.to(device)
            y_pred_test = model(test_batch.x, test_batch.edge_index).squeeze(dim=-1)
            test_label = test_batch.y
            test_loss = criterion(y_pred_test, test_label)
            avg_test_loss += test_loss.item()
            wandb.log({"test_loss": test_loss.item()})
            predictions.append(y_pred_test.detach().cpu().numpy())
            truths.append(test_label.detach().cpu().numpy())
            pf_recos.append(test_batch.x[:, 0].detach().cpu().numpy())
    avg_test_loss /= (test_batch_id + 1)
    wandb.log({"avg_test_loss": avg_test_loss})

    torch.save(np.array(predictions), save_dir + 'predictions.pt')
    torch.save(np.array(truths), save_dir + 'truths.pt')
    torch.save(np.array(pf_recos), save_dir + 'pf_recos.pt')
    wandb.finish()


def plotting_predictions(save_dir):
    predictions = torch.load(save_dir + 'predictions.pt')
    truths = torch.load(save_dir + 'truths.pt')
    pf_recos = torch.load(save_dir + 'pf_recos.pt')
    total_predictions = []
    total_truths = []
    total_pfrs = []
    n_bins = 30
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            '''
            total_predictions.append(math.pow(10, predictions[i][j]))
            total_truths.append(math.pow(10, truths[i][j]))
            total_pfrs.append(math.pow(10, pf_recos[i][j]))
            '''
            total_predictions.append(math.pow(predictions[i][j], 4))
            total_truths.append(math.pow(truths[i][j], 4))
            total_pfrs.append(math.pow(pf_recos[i][j], 4))
    fig = plt.figure(figsize=(10, 8))
    _, bins, _ = plt.hist(x=total_truths, bins=n_bins, histtype='step', label='truth')
    _ = plt.hist(x=total_predictions, bins=bins, histtype='step', label='pred')
    _ = plt.hist(x=total_pfrs, bins=bins, histtype='step', label='pf_reco')
    plt.legend()
    plt.savefig(save_dir + 'pt_comparison.png')
    plt.close(fig=fig)


if __name__ == '__main__':
    train()
    save_dir = r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\output/'
    plotting_predictions(save_dir)