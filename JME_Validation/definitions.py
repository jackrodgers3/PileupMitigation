from bamboo import treefunctions as op
from bamboo import treedecorators as btd

# Object definitions


def muonDef(mu, iso = False):
    return op.AND(
        mu.pt >= 20.,
        op.abs(mu.eta) <= 2.4,
        op.abs(mu.dxy) <= 0.05,
        op.abs(mu.dz) <= 0.1,
        mu.miniPFRelIso_all <= 0.4 if iso else 99,
        mu.sip3d <= 8,
        # mu.looseId
    )


def muonConePt(muons):
    return op.map(muons, lambda lep: op.multiSwitch(
        (op.AND(op.abs(lep.pdgId) != 11, op.abs(lep.pdgId) != 13), lep.pt),
        (op.AND(op.abs(lep.pdgId) == 13, lep.mvaTTH > 0.50), lep.pt),
        0.9*lep.pt*(1.+lep.jetRelIso)
    ))


def elDef(el):
    return op.AND(
        el.pt >= 20.,
        op.abs(el.eta) <= 2.5,
        op.abs(el.dxy) <= 0.05,
        op.abs(el.dz) <= 0.1,
        el.miniPFRelIso_all <= 0.4,
        el.sip3d <= 8,
        # el.mvaNoIso_WPL,
        el.lostHits <= 1
    )


def elConePt(electrons):
    return op.map(electrons, lambda lep: op.multiSwitch(
        (op.AND(op.abs(lep.pdgId) != 11, op.abs(lep.pdgId) != 13), lep.pt),
        # (op.AND(op.abs(lep.pdgId) == 11, lep.mvaTTH > 0.30), lep.pt), # run3 MC don't have mvaTTH for electrons
        (op.AND(op.abs(lep.pdgId) == 11), lep.pt),
        0.9*lep.pt*(1.+lep.jetRelIso)
    ))


def cleanElectrons(electrons, muons):
    cleanedElectrons = op.select(electrons, lambda el: op.NOT(
        op.rng_any(
            muons, lambda mu: op.deltaR(el.p4, mu.p4) <= 0.3))
    )
    return cleanedElectrons




def ak4jetDef(jet):
    return op.AND(
        jet.jetId & 0x2, # any jetID, i.e. > loose selection 
        jet.pt > 20.,
        op.abs(jet.eta) <= 5.2
    )

def ak8jetDef(jet):
    return op.AND(
        # jet.jetId & 2, 
        jet.nConstituents>0,
        jet.pt > 150.,
        op.abs(jet.eta) <= 2.5
    )



def cleanJets(jets, genjets, muons, electrons, deltaRcut = 0.4, sort=True):
    jets = op.select(jets, lambda jet: op.AND(
            op.NOT(op.rng_any(electrons, lambda ele: op.deltaR(jet.p4, ele.p4) < deltaRcut)),
            op.NOT(op.rng_any(muons, lambda mu: op.deltaR(jet.p4, mu.p4) < deltaRcut))
        ))
    genjets = op.select(genjets, lambda jet: op.AND(
            op.NOT(op.rng_any(electrons, lambda ele: op.deltaR(jet.p4, ele.p4) < deltaRcut)),
            op.NOT(op.rng_any(muons, lambda mu: op.deltaR(jet.p4, mu.p4) < deltaRcut))
        ))

    if sort:
        jets = op.sort(jets, lambda j: -j.pt)
    return jets,genjets


def effjets(jets):
    return op.select(jets,lambda jet: op.AND(
                op.deltaR(jet.p4,jet.genJet.p4) < 0.2,
                jet.genJet.pt > 30
            ))

def purityjets(jets):
    return op.select(jets, lambda jet: op.AND(
                op.deltaR(jet.p4,jet.genJet.p4) < 0.2,
                jet.genJet.pt > 20
            ))

def pujets(jets):
    return op.select(jets, lambda jet: 
                op.deltaR(jet.p4,jet.genJet.p4) > 0.4
            )

def matchedjets(tree, electrons, muons, redo_match = False):
    
    if redo_match:
        # mapping reco jet index to genjet with smalles deltaR
        index = op.map(tree.GenJet,lambda gj: op.rng_min_element_index(tree.Jet, lambda rj: op.deltaR(gj.p4,rj.p4)))
        # adding index variable to tree for genjet branch
        tree.GenJet.valueType.MyRJ = btd.itemProxy(index)
        # creating pair of genjet and reco jet based on index from above
        gjrj_pairs = op.combine((tree.GenJet, tree.Jet),pred=lambda gj,rj: gj.MyRJ == rj.idx)
    else:
        # take matching from nanoAOD with recojet genindex
        gjrj_pairs = op.combine((tree.GenJet, tree.Jet),pred=lambda gj,rj: rj.genJet.idx == gj.idx)
    
    sort_jets = op.sort(tree.GenJet, lambda gjet: -gjet.pt)
    return  op.select(gjrj_pairs, lambda pair: op.AND( 
        pair[0].pt >= sort_jets[2].pt,
        op.deltaR(pair[0].p4,pair[1].p4) < 0.2,
        op.NOT(op.rng_any(electrons, lambda ele: op.deltaR(pair[1].p4, ele.p4) < 0.4)),
        op.NOT(op.rng_any(muons, lambda mu: op.deltaR(pair[1].p4, mu.p4) < 0.4))
    ))

    


def defineObjects(tree):
    # Muons
    muons = op.sort(
        op.select(tree.Muon, lambda mu: muonDef(mu)),
        #lambda mu: -muonConePt(tree.Muon)[mu.idx]
        lambda mu: -mu.pt
    )

    isomuons = op.sort(
        op.select(tree.Muon, lambda mu: muonDef(mu,iso=True)),
        #lambda mu: -muonConePt(tree.Muon)[mu.idx]
        lambda mu: -mu.pt
    )

    # Electrons
    electrons = op.sort(
        op.select(tree.Electron, lambda el: elDef(el)),
        #lambda el: -elConePt(tree.Electron)[el.idx]
        lambda el: -el.pt
    )
    # Cleaned Electrons
    clElectrons = cleanElectrons(electrons, muons)
    
    # AK4 Jets
    ak4Jets = op.sort(
        op.select(tree.Jet, lambda jet: ak4jetDef(jet)), lambda jet: -jet.pt)

    # AK8 Jets
    ak8Jets = op.sort(
        op.select(tree.FatJet, lambda jet: ak8jetDef(jet)), lambda jet: -jet.pt)
    
    ## jet - lepton cleaning
    clak4Jets, clak4genjets = cleanJets(ak4Jets, tree.GenJet, muons, clElectrons)
    clak8Jets,_ = cleanJets(ak8Jets, tree.GenJet, muons, clElectrons, deltaRcut = 0.8)
    
    ## jet ID & pT recommendations
    ak4JetsID = op.select(
        clak4Jets, lambda jet: jet.jetId & 2)
    
    ak4Jetspt40 = op.select(
        ak4JetsID, lambda jet: jet.pt > 40)

    ak4Jetspt100 = op.select(
        ak4JetsID, lambda jet: jet.pt > 100)
    
    ak4Jetsetas2p4 = op.select(
        ak4JetsID, lambda jet: op.abs(jet.eta) < 2.4)
    
    ak4Jetsetag2p4 = op.select(
        ak4JetsID, lambda jet: op.abs(jet.eta) > 2.4)


    return isomuons, electrons, clElectrons, ak4Jets, clak4Jets, ak4JetsID, ak4Jetspt40, ak4Jetspt100, ak4Jetsetas2p4, ak4Jetsetag2p4, ak8Jets, clak8Jets, clak4genjets
    
