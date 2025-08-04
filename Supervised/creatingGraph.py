from timeit import default_timer as timer
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import math
from math import pi
import torch
import random
from torch_geometric.data import Data
import pickle
from pyjet import cluster, DTYPE_PTEPM
from scipy.spatial import distance
import sys
np.random.seed(0)

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

def r_gen_dataframe(rfilename, num_event, num_start=0):
    """
    select pfcands from original root and convert to a pandas dataframe.
    Returned is a list of dataframes, with one dataframe for one event.
    """
    print(f"reading events from {num_start} to {num_start + num_event}")
    tree = uproot.open(rfilename)["Events"]
    pfcands = tree.arrays(filter_name="PF_*", entry_start=num_start,
                          entry_stop=num_event + num_start)
    genparts = tree.arrays(filter_name="packedGenPart_*",
                           entry_start=num_start, entry_stop=num_event + num_start)
    print(tree.num_entries)

    df_pf_list = []
    df_gen_list = []
    pf_jetindices_list = []
    gen_jetindices_list = []
    njets_matched_list = []

    #
    # todo: this loop can probably be removed;
    # no need to convert to dataframe for each event
    #
    for i in range(num_event):
        if i % 10 == 0:
            print(f"processed {i} events")
        event = pfcands[i]
        selected_features = ['PF_eta', 'PF_phi', 'PF_pt',
                             'PF_pdgId', 'PF_charge', 'PF_puppiWeight', 'PF_puppiWeightChg', 'PF_dz',
                             'PF_fromPV'
                             ]
        pf_chosen = event[selected_features]
        df_pfcands = ak.to_dataframe(pf_chosen)
        df_pfcands = df_pfcands[abs(df_pfcands['PF_eta']) < 2.5]
        df_pfcands = df_pfcands[df_pfcands['PF_pt'] > 0.5]
        df_pfcands.index = range(len(df_pfcands))
        if i == 0:
            print(df_pfcands)
        # df_pfcands['PF_pt'] = np.log(df_pfcands['PF_pt'])

        df_pf_list.append(df_pfcands)

        genevent = genparts[i]
        selected_features_gen = ['packedGenPart_eta', 'packedGenPart_phi',
                                 'packedGenPart_pt', 'packedGenPart_pdgId', 'packedGenPart_charge']
        gen_chosen = genevent[selected_features_gen]
        df_genparts = ak.to_dataframe(gen_chosen)
        # eliminate those with eta more than 2.5 and also neutrinos
        selection = (abs(df_genparts['packedGenPart_eta']) < 2.5) & (abs(df_genparts['packedGenPart_pdgId']) != 12) & (
                abs(df_genparts['packedGenPart_pdgId']) != 14) & (abs(df_genparts['packedGenPart_pdgId']) != 16)
        df_genparts = df_genparts[selection]
        df_genparts.index = range(len(df_genparts))
        df_gen_list.append(df_genparts)

        jets = clusterJets(df_pfcands['PF_pt'], df_pfcands['PF_eta'], df_pfcands['PF_phi'])
        jets_truth = clusterJets(df_genparts['packedGenPart_pt'], df_genparts['packedGenPart_eta'],
                                 df_genparts['packedGenPart_phi'])
        matched_indices = matchJets(jets_truth, jets)
        if i == 0:
            print(matched_indices)
        jets_matched = []
        jets_truth_matched = []
        for matchidx in matched_indices:
            jets_matched.append(jets[matchidx[1]])
            jets_truth_matched.append(jets_truth[matchidx[0]])
        if i == 0:
            print(jets_matched)
        # print(df_pfcands['PF_pt'])
        pf_jetindices = get_jetidx(df_pfcands['PF_pt'], df_pfcands['PF_eta'], df_pfcands['PF_phi'], jets_matched)
        if i == 0:
            print(pf_jetindices)
        gen_jetindices = get_jetidx(df_genparts['packedGenPart_pt'], df_genparts['packedGenPart_eta'],
                                    df_genparts['packedGenPart_phi'], jets_truth_matched)
        njets_matched = len(jets_matched)
        pf_jetindices_list.append(pf_jetindices)
        gen_jetindices_list.append(gen_jetindices)
        njets_matched_list.append(njets_matched)

    return df_pf_list, df_gen_list, pf_jetindices_list, gen_jetindices_list, njets_matched_list


def r_prepare_dataset(rfilename, num_event, dR, num_start=0):
    """
    process each dataframe, prepare the ingredients for graphs (edges, node features, labels, etc).
    Returned is a list of graphs (torch.geometric data), with one graph for one event.
    """
    data_list = []

    df_pf_list, df_gen_list, pf_jetindices_list, gen_jetindices_list, njets_matched_list = r_gen_dataframe(rfilename,
                                                                                                         num_event,num_start)
    PTCUT = 0.5

    # defination of max edge distance
    # deltaR < 0.8/0.4 are recognized as an edge
    deltaRSetting = dR
    # deltaRSetting = 0.4

    pre_data = []

    for num in range(len(df_pf_list)):
        if num % 10 == 0:
            print(f"processed {num} events")
        #
        # toDO: this for loop can probably be also removed
        # and the gen_dataframe can be merged inside this function
        #
        df_pfcands_all = df_pf_list[num]
        df_gencands_all = df_gen_list[num]
        if njets_matched_list[num] == 0:
            continue
        for ijet in range(0, njets_matched_list[num]):
            df_pfcandsidx = []
            df_gencandsidx = []
            for kp in range(len(pf_jetindices_list[num])):
                # print(pf_jetindices_list[num][kp])
                if (pf_jetindices_list[num][kp] == ijet):
                    df_pfcandsidx.append(kp)
            for kg in range(len(gen_jetindices_list[num])):
                if gen_jetindices_list[num][kg] == ijet:
                    df_gencandsidx.append(kg)
            df_pfcands = df_pfcands_all.loc[df_pfcandsidx]
            df_gencands = df_gencands_all.loc[df_gencandsidx]
            # fromPV > 2 or < 1 is a really strict cut
            LV_index = np.where((df_pfcands['PF_puppiWeight'] > 0.99) & (df_pfcands['PF_charge'] != 0) & (
                    df_pfcands['PF_pt'] > PTCUT) & (df_pfcands['PF_fromPV'] > 2))[0]
            PU_index = np.where((df_pfcands['PF_puppiWeight'] < 0.01) & (df_pfcands['PF_charge'] != 0) & (
                    df_pfcands['PF_pt'] > PTCUT) & (df_pfcands['PF_fromPV'] < 1))[0]
            # print("LV index", LV_index)
            # print("PU index", PU_index)
            # if LV_index.shape[0] < 5 or PU_index.shape[0] < 50:
            # continue
            Neutral_index = np.where(df_pfcands['PF_charge'] == 0)[0]
            Charge_index = np.where(df_pfcands['PF_charge'] != 0)[0]

            '''
            # label of samples
            label = df_pfcands.loc[:, ['PF_puppiWeight']].to_numpy()
            label = torch.from_numpy(label).view(-1)
            label = label.type(torch.long)
            '''

            # node features
            node_features = df_pfcands.drop(df_pfcands.loc[:, ['PF_charge']], axis=1).drop(
                df_pfcands.loc[:, ['PF_fromPV']], axis=1).to_numpy()

            node_features = torch.from_numpy(node_features)
            node_features = node_features.type(torch.float32)

            jet = clusterJets(
                df_gencands['packedGenPart_pt'].to_numpy(),
                df_gencands['packedGenPart_eta'].to_numpy(),
                df_gencands['packedGenPart_phi'].to_numpy()
                              )
            pf_jet = clusterJets(
                df_pfcands['PF_pt'].to_numpy(),
                df_pfcands['PF_eta'].to_numpy(),
                df_pfcands['PF_phi'].to_numpy()
            )

            label = torch.tensor([jet[0].mass]).view(-1)
            label = label.type(torch.float32)
            pf_jm = torch.tensor([pf_jet[0].mass]).view(-1)
            pf_jm = pf_jm.type(torch.float32)
            # set the charge pdgId for one hot encoding later
            # ToDO: fix for muons and electrons
            index_pdgId = 3
            node_features[[Charge_index.tolist()], index_pdgId] = 0
            # get index for mask training and testing samples for pdgId(3) and puppiWeight(4)
            # one hot encoding for pdgId and puppiWeight
            pdgId = node_features[:, index_pdgId]
            photon_indices = (pdgId == 22)
            pdgId[photon_indices] = 1
            hadron_indices = (pdgId == 130)
            pdgId[hadron_indices] = 2
            pdgId = pdgId.type(torch.long)
            # print(pdgId)
            pdgId_one_hot = torch.nn.functional.one_hot(pdgId)
            pdgId_one_hot = pdgId_one_hot.type(torch.float32)
            if pdgId_one_hot.shape[1] != 3: continue
            assert pdgId_one_hot.shape[1] == 3, "pdgId_one_hot.shape[1] != 3"
            # print ("pdgID_one_hot", pdgId_one_hot)
            # set the neutral puppiWeight to default
            index_puppi = 4
            index_puppichg = 5
            pWeight = node_features[:, index_puppi].clone()
            pWeightchg = node_features[:, index_puppichg].clone()
            node_features[[Neutral_index.tolist()], index_puppi] = 2
            puppiWeight = node_features[:, index_puppi]
            puppiWeight = puppiWeight.type(torch.long)
            puppiWeight_one_hot = torch.nn.functional.one_hot(puppiWeight)
            puppiWeight_one_hot = puppiWeight_one_hot.type(torch.float32)
            # columnsNamesArr = df_pfcands.columns.values
            node_features = torch.cat(
                (node_features[:, 0:5], pdgId_one_hot, puppiWeight_one_hot), 1)
            #    (node_features[:, 0:3], pdgId_one_hot, node_features[:, -1:], puppiWeight_one_hot), 1)
            # i(node_features[:, 0:3], pdgId_one_hot,node_features[:,5:6], puppiWeight_one_hot), 1)
            # (node_features[:, 0:4], pdgId_one_hot, puppiWeight_one_hot), 1)
            if num == 0:
                print("NODE FEATURES: ", node_features)
                print("pdgId dimensions: ", pdgId_one_hot.shape)
                print("puppi weights dimensions: ", puppiWeight_one_hot.shape)
                print("last dimention: ", node_features[:, -1:].shape)
                print("node_features dimension: ", node_features.shape)
                print("node_features[:, 0:3] dimention: ",
                      node_features[:, 0:3].shape)
                print("node_features dimension: ", node_features.shape)
                print("node_features[:, 6:7]",
                      node_features[:, 6:7].shape)  # dz values
                # print("columnsNamesArr", columnsNamesArr)
                # print ("pdgId_one_hot " , pdgId_one_hot)
                # print("node_features[:,-1:]",node_features[:,-1:])
                # print("puppi weights", puppiWeight_one_hot)
                # print("node features: ", node_features)

            # node_features = node_features.type(torch.float32)
            # construct edge index for graph

            phi = df_pfcands['PF_phi'].to_numpy().reshape((-1, 1))
            eta = df_pfcands['PF_eta'].to_numpy().reshape((-1, 1))

            # df_gencands = df_gen_list[num]
            gen_features = df_gencands.to_numpy()
            gen_features = torch.from_numpy(gen_features)
            gen_features = gen_features.type(torch.float32)

            dist_phi = distance.cdist(phi, phi, 'cityblock')
            # deal with periodic feature of phi
            indices = np.where(dist_phi > pi)
            temp = np.ceil((dist_phi[indices] - pi) / (2 * pi)) * (2 * pi)
            dist_phi[indices] = dist_phi[indices] - temp
            dist_eta = distance.cdist(eta, eta, 'cityblock')

            dist = np.sqrt(dist_phi ** 2 + dist_eta ** 2)

            edge_source = np.where((dist < deltaRSetting) & (dist != 0))[0]
            edge_target = np.where((dist < deltaRSetting) & (dist != 0))[1]

            edge_index = np.array([edge_source, edge_target])
            edge_index = torch.from_numpy(edge_index)
            edge_index = edge_index.type(torch.long)

            graph = Data(x=node_features, edge_index=edge_index, y=label)

            graph.LV_index = LV_index
            graph.PU_index = PU_index
            graph.event_num = num
            graph.Neutral_index = Neutral_index
            graph.Charge_index = Charge_index
            graph.GenPart_nump = gen_features
            graph.pWeight = pWeight
            graph.pWeightchg = pWeightchg
            graph.pf_jm = pf_jm

            data_list.append(graph)

    return data_list


def main_regression(dR = 0.4):
    start = timer()

    iname = r"D:\Data\Research\PUPPI\WJetsToQQ_HT-800toInf\output_1.root"
    num_events_train = 10000
    oname = r"C:\Users\jackm\PycharmProjects\PileupMitigation\Supervised\data/train" + str(num_events_train)
    dataset_train = r_prepare_dataset(iname, num_events_train, dR)
    # save outputs in pickle format
    with open(oname, "wb") as fp:
        pickle.dump(dataset_train, fp)
    fp.close()
    num_events_valid = 10000
    oname = r"C:\Users\jackm\PycharmProjects\PileupMitigation\Supervised\data/valid" + str(num_events_valid)
    dataset_test = r_prepare_dataset(iname, num_events_valid, dR, num_events_train)
    with open(oname, "wb") as fp:
        pickle.dump(dataset_test, fp)
    fp.close()
    num_events_test = 10000
    oname = r"C:\Users\jackm\PycharmProjects\PileupMitigation\Supervised\data/test" + str(num_events_test)
    dataset_valid = r_prepare_dataset(
        iname, num_events_test, dR, num_events_train + num_events_valid)
    with open(oname, "wb") as fp:
        pickle.dump(dataset_valid, fp)
    fp.close()
    end = timer()
    program_time = end - start
    print("generating graph time " + str(program_time))


if __name__ == '__main__':
    dR = 0.3
    main_regression(dR=dR)
