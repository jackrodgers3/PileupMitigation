from bamboo import treefunctions as op
from bamboo.plots import Plot, SummedPlot
from bamboo.plots import EquidistantBinning as EqBin
from bamboo.plots import VariableBinning as VarBin
from binnings import eta_binning, pt_binning, response_pt_binning

def muonPlots(muons, sel, sel_tag, maxMuons = 4):
    plots = []

    plots.append(Plot.make1D(f"{sel_tag}_Muon_nMuons", op.rng_len(muons), sel, EqBin(15, 0., 15.),xTitle = "Number of muons"))
    
    #### do a for loop through all Muons
    for i in range(maxMuons):
        plots.append(Plot.make1D(f"{sel_tag}_Muon{i+1}_pt", op.switch(op.rng_len(muons)>i,muons[i].pt,-99.), sel, EqBin(20, 0., 500.), xTitle=f"muon_{{{i+1}}} p_{{T}} [GeV]"))
        plots.append(Plot.make1D(f"{sel_tag}_Muon{i+1}_eta", op.switch(op.rng_len(muons)>i,muons[i].eta,-99.), sel, EqBin(20, -2.5, 2.5), xTitle=f"muon_{{{i+1}}} #eta"))
        plots.append(Plot.make1D(f"{sel_tag}_Muon{i+1}_phi", op.switch(op.rng_len(muons)>i,muons[i].phi,-99.), sel, EqBin(20, -2.5, 2.5), xTitle=f"muon_{{{i+1}}} #phi"))
 
    return plots


def electronPlots(electrons, sel, sel_tag, maxElectrons = 4):
    plots = []

    plots.append(Plot.make1D(f"{sel_tag}_Electron_nElectrons", op.rng_len(electrons), sel, EqBin(15, 0., 15.), xTitle=f"Number of Electrons"))
    
    #### do a for loop through all Electrons
    for i in range(maxElectrons):
        plots.append(Plot.make1D(f"{sel_tag}_Electron{i+1}_pt", op.switch(op.rng_len(electrons)>i,electrons[i].pt,-99.), sel, EqBin(20, 0., 500.), xTitle=f"electron_{{{i+1}}} p_{{T}} [GeV]"))
        plots.append(Plot.make1D(f"{sel_tag}_Electron{i+1}_eta", op.switch(op.rng_len(electrons)>i,electrons[i].eta,-99.), sel, EqBin(20, -2.5, 2.5), xTitle=f"electron_{{{i+1}}} #eta"))
        plots.append(Plot.make1D(f"{sel_tag}_Electron{i+1}_phi", op.switch(op.rng_len(electrons)>i,electrons[i].phi,-99.), sel, EqBin(20, -2.5, 2.5), xTitle=f"electron_{{{i+1}}} #phi"))
 
    return plots


def AK4jetPlots(jets, sel, sel_tag, maxJets=4):
    plots = []

    plots.append(Plot.make1D(f"{sel_tag}_AK4Jets_nJets",op.rng_len(jets),sel,EqBin(15,0.,15.), xTitle=f"Number of Jets"))

    etas = op.map(jets, lambda j: j.eta)
    plots.append(Plot.make1D(f"{sel_tag}_AK4Jets_eta",etas,sel,EqBin(100,-5.,5.), xTitle=f"#eta"))

    pts = op.map(jets, lambda j: j.pt)
    plots.append(Plot.make1D(f"{sel_tag}_AK4Jets_pt",pts,sel,EqBin(250,0.,500.), xTitle=f"p_{{T}}"))

    #### do a for loop through all Jets
    for i in range(maxJets):
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_pt", op.switch(op.rng_len(jets)>i,jets[i].pt,-99.), sel, EqBin(250, 0., 500.), xTitle=f"jet_{{{i+1}}} p_{{T}} [GeV]"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_eta", op.switch(op.rng_len(jets)>i,jets[i].eta,-99.), sel, EqBin(100, -5., 5.), xTitle=f"jet_{{{i+1}}} #eta"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_phi", op.switch(op.rng_len(jets)>i,jets[i].phi,-99.), sel, EqBin(63, -3.5, 3.5), xTitle=f"jet_{{{i+1}}} #phi"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_nConst", op.switch(op.rng_len(jets)>i,jets[i].nConstituents,-99.), sel, EqBin(20, 0, 20), xTitle=f"jet_{{{i+1}}} number of constituents"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_nConst_eta2p0to3p0",op.switch(op.AND(op.rng_len(jets)>i,jets[i].eta < 3,jets[i].eta > 2,jets[i].pt < 50),jets[i].nConstituents,-99.), sel, EqBin(20, 0, 20), xTitle=f"jet_{{{i+1}}} number of constituents"))
        # DeepJet variable
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_btagDeepFlavB", op.switch(op.rng_len(jets)>i,jets[i].btagDeepFlavB,-99.), sel, EqBin(50, 0, 1), xTitle=f"jet_{{{i+1}}} btagDeepFlavB"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_btagDeepFlavCvB", op.switch(op.rng_len(jets)>i,jets[i].btagDeepFlavCvB,-99.), sel, EqBin(50, 0, 1), xTitle=f"jet_{{{i+1}}} btagDeepFlavCvB"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_btagDeepFlavCvL", op.switch(op.rng_len(jets)>i,jets[i].btagDeepFlavCvL,-99.), sel, EqBin(50, 0, 1), xTitle=f"jet_{{{i+1}}} btagDeepFlavCvL"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_btagDeepFlavQG", op.switch(op.rng_len(jets)>i,jets[i].btagDeepFlavQG,-99.), sel, EqBin(50, 0, 1), xTitle=f"jet_{{{i+1}}} btagDeepFlavQG"))
        # DeepJet variable
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_btagPNetB", op.switch(op.rng_len(jets)>i,jets[i].btagPNetB,-99.), sel, EqBin(50, 0, 1), xTitle=f"jet_{{{i+1}}} btagPNetB"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_btagPNetCvB", op.switch(op.rng_len(jets)>i,jets[i].btagPNetCvB,-99.), sel, EqBin(50, 0, 1), xTitle=f"jet_{{{i+1}}} btagPNetCvB"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_btagPNetCvL", op.switch(op.rng_len(jets)>i,jets[i].btagPNetCvL,-99.), sel, EqBin(50, 0, 1), xTitle=f"jet_{{{i+1}}} btagPNetCvL"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_btagPNetQvG", op.switch(op.rng_len(jets)>i,jets[i].btagPNetQvG,-99.), sel, EqBin(50, 0, 1), xTitle=f"jet_{{{i+1}}} btagPNetQvG"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_PNetRegPtRawCorr", op.switch(op.rng_len(jets)>i,jets[i].PNetRegPtRawCorr,-99.), sel, EqBin(50, 0, 10), xTitle=f"jet_{{{i+1}}} PNetRegPtRawCorr"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_PNetRegPtRawCorrNeutrino", op.switch(op.rng_len(jets)>i,jets[i].PNetRegPtRawCorrNeutrino,-99.), sel, EqBin(50, 0, 10), xTitle=f"jet_{{{i+1}}} PNetRegPtRawCorrNeutrino"))
        plots.append(Plot.make1D(f"{sel_tag}_Jet{i+1}_PNetRegPtRawRes", op.switch(op.rng_len(jets)>i,jets[i].PNetRegPtRawRes,-99.), sel, EqBin(50, 0, 10), xTitle=f"jet_{{{i+1}}} PNetRegPtRawRes"))


    chEmEF = op.map(jets, lambda j: j.chEmEF)
    plots.append(Plot.make1D(f"{sel_tag}_AK4Jets_chEmEF",chEmEF,sel,EqBin(20,0.,1.), xTitle=f"chEmEF"))

    chHEF = op.map(jets, lambda j: j.chHEF)
    plots.append(Plot.make1D(f"{sel_tag}_AK4Jets_chHEF",chHEF,sel,EqBin(20,0.,1.), xTitle=f"chHEF"))
    
    # hfEmEF = op.map(jets, lambda j: j.hfEmEF)
    # plots.append(Plot.make1D(f"{sel_tag}_AK4Jets_hfEmEF",hfEmEF,sel,EqBin(20,0.,1.), xTitle=f"hfEmEF"))

    # hfHEF = op.map(jets, lambda j: j.hfHEF)
    # plots.append(Plot.make1D(f"{sel_tag}_AK4Jets_hfHEF",hfHEF,sel,EqBin(20,0.,1.), xTitle=f"hfHEF"))

    muEF = op.map(jets, lambda j: j.muEF)
    plots.append(Plot.make1D(f"{sel_tag}_AK4Jets_muEF",muEF,sel,EqBin(20,0.,1.), xTitle=f"muEF"))

    neEmEF = op.map(jets, lambda j: j.neEmEF)
    plots.append(Plot.make1D(f"{sel_tag}_AK4Jets_neEmEF",neEmEF,sel,EqBin(20,0.,1.), xTitle=f"neEmEF"))

    neHEF = op.map(jets, lambda j: j.neHEF)
    plots.append(Plot.make1D(f"{sel_tag}_AK4Jets_neHEF",neHEF,sel,EqBin(20,0.,1.), xTitle=f"neHEF"))

    if op.rng_len(jets)>1:
        plots.append(Plot.make1D(f"{sel_tag}_AK4Jets_DeltaPhi12",op.deltaPhi(jets[0].p4,jets[1].p4),sel,EqBin(20,0.,4.), xTitle=f"#delta Phi(jet_1,jet_2)"))


    return plots

def AK8jetPlots(jets, sel, sel_tag, maxJets=4):
    plots = []

    plots.append(Plot.make1D(f"{sel_tag}_AK8Jets_nJets",op.rng_len(jets),sel,EqBin(15,0.,15.), xTitle=f"Number of Jets"))

    etas = op.map(jets, lambda j: j.eta)
    plots.append(Plot.make1D(f"{sel_tag}_AK8Jets_eta",etas,sel,EqBin(100,-5.,5.), xTitle=f"#eta"))

    pts = op.map(jets, lambda j: j.pt)
    plots.append(Plot.make1D(f"{sel_tag}_AK8Jets_pt",pts,sel,EqBin(500,0.,1000.), xTitle=f"p_{{T}}"))

    #### do a for loop through all Jets
    for i in range(maxJets):
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_pt",  op.switch(op.rng_len(jets)>i,jets[i].pt, -99.), sel, EqBin(500, 0., 1000.), xTitle=f"jet_{{{i+1}}} p_{{T}} [GeV]"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_eta", op.switch(op.rng_len(jets)>i,jets[i].eta, -99.), sel, EqBin(100, -5., 5.), xTitle=f"jet_{{{i+1}}} #eta"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_phi", op.switch(op.rng_len(jets)>i,jets[i].phi, -99.), sel, EqBin(63, -3.5, 3.5), xTitle=f"jet_{{{i+1}}} #phi"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_nConst", op.switch(op.rng_len(jets)>i,jets[i].nConstituents, -99.), sel, EqBin(20, 0, 20), xTitle=f"jet_{{{i+1}}} number of constituents"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_tau21", op.switch(op.rng_len(jets)>i,jets[i].tau2/jets[i].tau1, -99.), sel, EqBin(50, 0, 1), xTitle=f"jet_{{{i+1}}} tau21"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_tau32", op.switch(op.rng_len(jets)>i,jets[i].tau3/jets[i].tau2, -99.), sel, EqBin(50, 0, 1), xTitle=f"jet_{{{i+1}}} tau32"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_mSD", op.switch(op.rng_len(jets)>i,jets[i].msoftdrop, -99.), sel, EqBin(150, 0, 300), xTitle=f"jet_{{{i+1}}} mSD"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNetWithMass_QCD", op.switch(op.rng_len(jets)>i,jets[i].particleNetWithMass_QCD, -99.), sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNet_QCD"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNetWithMass_TvsQCD", op.switch(op.rng_len(jets)>i,jets[i].particleNetWithMass_QCD, -99.), sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNet_TvsQCD"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNetWithMass_H4qvsQCD", op.switch(op.rng_len(jets)>i,jets[i].particleNetWithMass_QCD, -99.), sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNet_H4qvsQCD"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNetWithMass_HbbvsQCD", op.switch(op.rng_len(jets)>i,jets[i].particleNetWithMass_QCD, -99.), sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNet_HbbvsQCD"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNetWithMass_WvsQCD", op.switch(op.rng_len(jets)>i,jets[i].particleNetWithMass_QCD, -99.), sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNet_WvsQCD"))

        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNetMD_QCD", op.switch(op.rng_len(jets)>i,jets[i].particleNet_QCD, -99.), sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNetMD_QCD"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNetMD_TvsQCD", op.switch(op.rng_len(jets)>i,jets[i].particleNet_QCD, -99.), sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNetMD_TvsQCD"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNetMD_H4qvsQCD", op.switch(op.rng_len(jets)>i,jets[i].particleNet_QCD, -99.), sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNetMD_H4qvsQCD"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNetMD_HbbvsQCD", op.switch(op.rng_len(jets)>i,jets[i].particleNet_QCD, -99.), sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNetMD_HbbvsQCD"))
        plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNetMD_WvsQCD", op.switch(op.rng_len(jets)>i,jets[i].particleNet_QCD, -99.), sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNetMD_WvsQCD"))

        # plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNetMD_QCD", jets[i].particleNetMD_QCD, sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNetMD_QCD"))
        # plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNetMD_Xbb", jets[i].particleNetMD_Xbb, sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNetMD_Xbb"))
        # plots.append(Plot.make1D(f"{sel_tag}_FatJet{i+1}_particleNet_WvsQCD", jets[i].particleNet_WvsQCD, sel, EqBin(20, 0, 1), xTitle=f"jet_{{{i+1}}} particleNet_WvsQCD"))

        return plots

def ZbosonPlots(Zboson, sel, sel_tag):
    plots = []

    plots.append(Plot.make1D(f"{sel_tag}_Zboson_mass",Zboson.M(),sel,EqBin(30,0.,200.),xTitle="m_{ll} [GeV]"))

    plots.append(Plot.make1D(f"{sel_tag}_Zboson_pt",Zboson.pt(),sel,EqBin(30,0.,200.),xTitle="Z boson p_{T} [GeV]"))

    return plots


def effPurityPlots(jets, gjets, recojets, sel, sel_tag, tree):
    plots = []
    eff_pteta_num = []
    eff_pteta_denum = []

    purity_pteta_num = []
    purity_pteta_denum = []

    eff_npveta_num = []
    eff_npveta_denum = []

    purity_npveta_num = []
    purity_npveta_denum = []


    # plot pT and eta at the beginning
    jetinputpt = op.map(jets,lambda j:j.pt)
    plots.append(Plot.make1D(f"{sel_tag}_effPurity_recoinput_pt", jetinputpt,sel,EqBin(200,0,1000),xTitle="p_{T}^{reco} [GeV]"))
    genjetinputpt = op.map(gjets,lambda j:j.pt)
    plots.append(Plot.make1D(f"{sel_tag}_effPurity_geninput_pt", genjetinputpt,sel,EqBin(200,0,1000),xTitle="p_{T}^{gen} [GeV]"))

    etaBinning = [etabin[0] for etatag, etabin in eta_binning.items() if "0p0to5p2" not in etatag]
    etaBinning+= [etabin[1] for etatag, etabin in eta_binning.items() if "0p0to5p2" not in etatag and etabin[1] not in etaBinning] 
    etaBinning = VarBin(etaBinning)


    ### eff: genjet pt > 30 GeV matched to reco jet over all genjets with pT > 30 GeV
    for ix in range(3):
        genjet = gjets[ix]
        recojet = op.rng_min_element_by(jets, lambda jet: op.deltaR(jet.p4,genjet.p4))

        denum_sel =op.AND(
            op.rng_len(gjets)>ix, 
            genjet.pt > 30
        ) 

        num_sel =op.AND(
            denum_sel,
            op.deltaR(recojet.p4,genjet.p4)<0.2, 
            op.rng_len(jets)>0
        ) 

        eff_pteta_num.append(Plot.make2D(f"{sel_tag}_eff_pteta_num_{ix}",( 
            op.switch(
                num_sel, 
                genjet.pt, -99.
            ), 
            op.abs(genjet.eta)
        ), sel, (EqBin(200,0,400), etaBinning), xTitle="p_{T}^{gen} [GeV]", yTitle="|#eta|" ))
        
        
        eff_pteta_denum.append(Plot.make2D(f"{sel_tag}_eff_pteta_denum_{ix}",( 
            op.switch(
                denum_sel, 
                genjet.pt, -99.
            ), 
            op.abs(genjet.eta)
        ), sel, (EqBin(200,0,400), etaBinning), xTitle="p_{T}^{gen} [GeV]", yTitle="|#eta|" ))

        #### vs npv, inclusive in pt
        eff_npveta_num.append(Plot.make2D(f"{sel_tag}_eff_npveta_num_{ix}",( 
            op.switch(
                num_sel, 
                tree.PV.npvsGood, -99.
            ), 
            op.abs(genjet.eta)
        ), sel, (EqBin(20,0,100), etaBinning), xTitle="N_{PV}", yTitle="|#eta|" ))
        
        
        eff_npveta_denum.append(Plot.make2D(f"{sel_tag}_eff_npveta_denum_{ix}",( 
            op.switch(
                denum_sel, 
                tree.PV.npvsGood, -99.
            ), 
            op.abs(genjet.eta)
        ), sel, (EqBin(20,0,100), etaBinning), xTitle="N_{PV}", yTitle="|#eta|" ))


        jets = op.select(jets, lambda j: j.idx!=recojet.idx)
    plots.append(SummedPlot(f"{sel_tag}_eff_pteta_num", eff_pteta_num, xTitle="p_{T}^{gen} [GeV]", yTitle="|#eta|"))
    plots.append(SummedPlot(f"{sel_tag}_eff_pteta_denum", eff_pteta_denum, xTitle="p_{T}^{gen} [GeV]", yTitle="|#eta|"))

    plots.append(SummedPlot(f"{sel_tag}_eff_npveta_num", eff_npveta_num, xTitle="N_{PV}", yTitle="|#eta|"))
    plots.append(SummedPlot(f"{sel_tag}_eff_npveta_denum", eff_npveta_denum, xTitle="N_{PV}", yTitle="|#eta|"))

    plots+=[num for num in eff_pteta_num]
    plots+=[denum for denum in eff_pteta_denum]

    plots+=[num for num in eff_npveta_num]
    plots+=[denum for denum in eff_npveta_denum]


    for ix in range(3):
        recojet = recojets[ix]
        genjet = op.rng_min_element_by(gjets, lambda jet: op.deltaR(recojet.p4,jet.p4))

        denum_sel =op.AND(
            op.rng_len(recojets)>ix, 
            recojet.pt > 30
        )
        num_sel = op.AND(
            denum_sel,
            op.deltaR(recojet.p4,genjet.p4)<0.2, 
            op.rng_len(gjets)>0
        )


        purity_pteta_num.append(Plot.make2D(f"{sel_tag}_purity_pteta_num_{ix}",( 
            op.switch(
                num_sel, 
                recojet.pt, -99.
            ), 
        op.abs(recojet.eta)
        ), sel, (EqBin(200,0,400), etaBinning), xTitle="p_{T}^{reco} [GeV]", yTitle="|#eta|" ))

        purity_pteta_denum.append(Plot.make2D(f"{sel_tag}_purity_pteta_denum_{ix}",(
            op.switch(
                denum_sel,
                recojet.pt, -99.
            ), 
           op.abs(recojet.eta)
        ), sel, (EqBin(200,0,400), etaBinning), xTitle="p_{T}^{reco} [GeV]", yTitle="|#eta|" ))

        #### vs npv, inclusive in pt
        purity_npveta_num.append(Plot.make2D(f"{sel_tag}_purity_npveta_num_{ix}",( 
            op.switch(
                num_sel, 
                tree.PV.npvsGood, -99.
            ), 
        op.abs(recojet.eta)
        ), sel, (EqBin(20,0,100), etaBinning), xTitle="N_{PV}", yTitle="|#eta|" ))

        purity_npveta_denum.append(Plot.make2D(f"{sel_tag}_purity_npveta_denum_{ix}",(
            op.switch(
                denum_sel, 
                tree.PV.npvsGood, -99.
            ), 
           op.abs(recojet.eta)
        ), sel, (EqBin(20,0,100), etaBinning), xTitle="N_{PV}", yTitle="|#eta|" ))
        
        gjets = op.select(gjets, lambda j: j.idx!=genjet.idx)
    plots.append(SummedPlot(f"{sel_tag}_purity_pteta_num", purity_pteta_num, xTitle="p_{T}^{reco} [GeV]", yTitle="|#eta|"))
    plots.append(SummedPlot(f"{sel_tag}_purity_pteta_denum", purity_pteta_denum, xTitle="p_{T}^{reco} [GeV]", yTitle="|#eta|"))

    plots.append(SummedPlot(f"{sel_tag}_purity_npveta_num", purity_npveta_num, xTitle="N_{PV}", yTitle="|#eta|"))
    plots.append(SummedPlot(f"{sel_tag}_purity_npveta_denum", purity_npveta_denum, xTitle="N_{PV}", yTitle="|#eta|"))


    plots+=[num for num in purity_pteta_num]
    plots+=[denum for denum in purity_pteta_denum]

    plots+=[num for num in purity_npveta_num]
    plots+=[denum for denum in purity_npveta_denum]

    return plots


def responsePlots(sel, sel_tag, genjets, jets, tree,ngenjets = 3, rawpt = False, debug_hists = False, deltaRcut = 0.2):
    plots = []
    ratios = []

    # sort genjets to choose the leading three
    genjets = op.sort(genjets, lambda j: -j.pt)

    # create binning
    etaBinning = [etabin[0] for etatag, etabin in eta_binning.items() if "0p0to5p2" not in etatag]
    etaBinning+= [etabin[1] for etatag, etabin in eta_binning.items() if "0p0to5p2" not in etatag and etabin[1] not in etaBinning] 
    etaBinning = VarBin(etaBinning)
    ptBinning = VarBin([ptbin[0] for pttag, ptbin in response_pt_binning.items()])

    # loop over n leading genjets
    for ix in range(ngenjets):
        genjet = genjets[ix]
        recojet = op.rng_min_element_by(jets, lambda jet: op.deltaR(jet.p4,genjet.p4))
        response = recojet.pt/genjet.pt if not rawpt else (tree._Jet.orig[recojet.idx].pt*(1-recojet.rawFactor))/genjet.pt

        # create n independent plots
        ratios.append( Plot.make3D(f"{sel_tag}_ratio_"+str(ix),
                                   (op.switch(
                                       op.AND(
                                           op.rng_len(genjets)>ix,
                                           op.deltaR(recojet.p4,genjet.p4)<deltaRcut, 
                                           op.rng_len(jets)>0
                                       ),
                                       response,
                                       -99.
                                   ),
                                   genjet.pt,
                                   op.abs(genjet.eta)),
                                   sel,
                                   (EqBin(100, 0., 2.), ptBinning, etaBinning),
                                   xTitle="p_{T}^{reco}/p_{T}^{gen}",
                                   yTitle="p_{T}^{gen}", zTitle="|#eta|"
                               )
                   )

        jets = op.select(jets, lambda j: j.idx!=recojet.idx)

    plots.append(SummedPlot(f"{sel_tag}_ratio", ratios, xTitle="p_{T}^{reco}/p_{T}^{gen}",yTitle="p_{T}^{gen}", zTitle="|#eta|"))

    plots+=[ratio for ratio in ratios]
    return plots



def eventPlots(tree, sel, sel_tag):
    plots = []
    
    plots.append(Plot.make1D(f"{sel_tag}_event_goodnpvs",tree.PV.npvsGood,sel,EqBin(100,0.,100.),xTitle = "Number of good PVs"))

    return plots


def efftauPlots(taus, jets, sel, sel_tag, ntaus = 3, deltaRcut = 0.4, bPNet = True):
    plots = []
    nums = []
    denums = []

    #sort tau by pT
    taus = op.sort(taus, lambda j: -j.pt)

    # create binning
    etaBinning = [etabin[0] for etatag, etabin in eta_binning.items() if "0p0to5p2" not in etatag]
    etaBinning+= [etabin[1] for etatag, etabin in eta_binning.items() if "0p0to5p2" not in etatag and etabin[1] not in etaBinning] 
    etaBinning = VarBin(etaBinning)
    # etaBinning = VarBin([0.0,2.0])
    ptBinning = VarBin([0.,10.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,100.])
    
    taupt = op.map(taus, lambda t:t.pt)
    plots.append(Plot.make1D(f"{sel_tag}_tau_pt",taupt,sel,EqBin(100,0.,200.),xTitle = "tau p_{T} [GeV]"))
    taueta = op.map(taus, lambda t:t.eta)
    plots.append(Plot.make1D(f"{sel_tag}_tau_eta",taueta,sel,EqBin(100,-5.,5.),xTitle = "tau #eta "))
    plots.append(Plot.make1D(f"{sel_tag}_tau_nTau",op.rng_len(taus),sel,EqBin(10,0.,10.),xTitle = "number of taus"))

    if bPNet:
        #### PNet Tau nodes
        tauPNet = op.map(jets, lambda j:j.btagPNetTauVJet)
        plots.append(Plot.make1D(f"{sel_tag}_jet_PNetTauvsJet",tauPNet,sel,EqBin(100,0.,1.),xTitle = "btagPNetTauVJet"))
        
        plots.append(Plot.make1D(f"{sel_tag}_jet1_PNetTauvsJet",jets[0].btagPNetTauVJet,sel,EqBin(100,0.,1.),xTitle = "btagPNetTauVJet"))



    # match each reco tau to closest jet
    for ix in range(ntaus):
        tau = taus[ix]
        recojet =  op.rng_min_element_by(jets, lambda jet: op.deltaR(jet.p4,tau.p4))

        denum_sel = op.rng_len(taus)>ix
        num_sel = op.AND(
            denum_sel,
            op.deltaR(recojet.p4,tau.p4)<deltaRcut, 
            op.rng_len(jets)>0
        )

        #PNet for matched jets
        if bPNet:
            plots.append(Plot.make1D(f"{sel_tag}_matchedjet_PNetTauvsJet_"+str(ix),op.switch(
                num_sel,
                recojet.btagPNetTauVJet,
                -99.
            ),sel,EqBin(100,0.,1.),xTitle = "btagPNetTauVJet"))
            
        nums.append( Plot.make2D(f"{sel_tag}_TAU_pteta_num_"+str(ix),
                                   (op.switch(
                                       num_sel,
                                       tau.pt,
                                       -99.
                                   ),
                                   op.abs(tau.eta)),
                                   sel,
                                   (ptBinning, etaBinning),
                                   xTitle="#tau p_{T}",
                                   yTitle="|#eta|"
                               )
                   )

        denums.append( Plot.make2D(f"{sel_tag}_TAU_pteta_denum_"+str(ix),
                                   (op.switch(
                                       denum_sel,
                                       tau.pt,
                                       -99.
                                   ),
                                   op.abs(tau.eta)),
                                   sel,
                                   (ptBinning, etaBinning),
                                   xTitle="#tau p_{T}",
                                   yTitle="|#eta|"
                               )
                   )



        jets = op.select(jets, lambda j: j.idx!=recojet.idx)
    
    plots.append(SummedPlot(f"{sel_tag}_TAU_pteta_num", nums, xTitle="#tau p_{T}",yTitle="|#eta|"))
    plots.append(SummedPlot(f"{sel_tag}_TAU_pteta_denum", denums, xTitle="#tau p_{T}",yTitle="|#eta|"))

    plots+=[num for num in nums]
    plots+=[denum for denum in denums]

    return plots
    
