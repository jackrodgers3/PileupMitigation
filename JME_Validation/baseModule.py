from bamboo.analysismodules import NanoAODModule, HistogramsModule
from bamboo.treedecorators import NanoAODDescription, CalcCollectionsGroups
from bamboo.analysisutils import makeMultiPrimaryDatasetTriggerSelection
from bamboo import treefunctions as op
from bamboo.analysisutils import forceDefine

from itertools import chain


class NanoBaseJME(NanoAODModule, HistogramsModule):
    def __init__(self, args):
        super(NanoBaseJME, self).__init__(args)

    def addArgs(self, parser):
        super(NanoBaseJME, self).addArgs(parser)
        parser.add_argument("--era",
                            action='store',
                            type=str,
                            default=None,
                            help='This has no use right now!')

    def prepareTree(self, tree, sample=None, sampleCfg=None, backend=None):
        def isMC():
            if sampleCfg['type'] == 'data':
                return False
            elif sampleCfg['type'] == 'mc':
                return True
            else:
                raise RuntimeError(
                    f"The type '{sampleCfg['type']}' of {sample} dataset not understood.")
        
        self.is_MC = isMC()
        isDriver = (self.args.distributed == "driver")
        era = sampleCfg['era']  # reserved for future use
        campaign = sampleCfg['campaign']
        jec = sampleCfg['jec']
        jet_algo_AK4 = 'AK4PF'+sampleCfg['jec_algo_AK4']
        jet_algo_AK8 = 'AK8PF'+sampleCfg['jec_algo_AK8']
        if jec=='':
            jec = 'Winter22Run3_V2_MC' if self.is_MC else 'Winter22Run3_RunD_V2_DATA'
        self.triggersPerPrimaryDataset = {}

        def addHLTPath(PD, HLT):
            if PD not in self.triggersPerPrimaryDataset.keys():
                self.triggersPerPrimaryDataset[PD] = []
            try:
                self.triggersPerPrimaryDataset[PD].append(
                    getattr(tree.HLT, HLT))
            except AttributeError:
                print("Couldn't find branch tree.HLT.%s, will omit it!" % HLT)

        def getNanoAODDescription():
            groups = ["HLT_", "MET_","PV_","Pileup_","Rho_"]
            collections = ["nElectron", "nJet", "nMuon", "nFatJet", "nSubJet","nGenJet","nGenVisTau","nJetCHS", "nTau","nGenJetAK8","nSubGenJetAK8"]
            varReaders = [CalcCollectionsGroups(Jet=("pt", "mass"),FatJet=("pt", "mass"))]
            return NanoAODDescription(groups=groups, collections=collections, systVariations=varReaders)

        tree, noSel, backend, lumiArgs = super(NanoBaseJME, self).prepareTree(tree=tree,
                                                                                 sample=sample,
                                                                                 sampleCfg=sampleCfg,
                                                                                 description=getNanoAODDescription(),
                                                                                 backend=backend)
        ### Triggers ###
        # Muon
        addHLTPath('Muon', 'IsoMu24')
        addHLTPath('Muon', 'IsoMu27')
        addHLTPath('Muon', 'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8')
        # EGamma
        addHLTPath('EGamma', 'Ele32_WPTight_Gsf')
        addHLTPath('EGamma', 'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL')
        # MuonEG
        addHLTPath('MuonEG', 'Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ')
        # SingleMuon
        addHLTPath('SingleMuon', 'IsoMu24')
        addHLTPath('SingleMuon', 'IsoMu27')
        # DoubleMuon
        addHLTPath('DoubleMuon', 'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8')

        # JetHT
        addHLTPath('JetHT', 'DiPFJetAve40')
        addHLTPath('JetHT', 'DiPFJetAve60')
        addHLTPath('JetHT', 'DiPFJetAve80')
        addHLTPath('JetHT', 'DiPFJetAve140')
        addHLTPath('JetHT', 'DiPFJetAve200')
        addHLTPath('JetHT', 'DiPFJetAve260')
        addHLTPath('JetHT', 'DiPFJetAve320')
        addHLTPath('JetHT', 'DiPFJetAve400')
        addHLTPath('JetHT', 'DiPFJetAve500')
        addHLTPath('JetHT', 'DiPFJetAve60_HFJEC')
        addHLTPath('JetHT', 'DiPFJetAve80_HFJEC')
        addHLTPath('JetHT', 'DiPFJetAve100_HFJEC')
        addHLTPath('JetHT', 'DiPFJetAve160_HFJEC')
        addHLTPath('JetHT', 'DiPFJetAve220_HFJEC')
        addHLTPath('JetHT', 'DiPFJetAve300_HFJEC')

        # Gen Weight and Triggers
        if self.is_MC:
            # noSel = noSel.refine('genWeight', weight=tree.genWeight, cut=(
            #     op.OR(*chain.from_iterable(self.triggersPerPrimaryDataset.values()))))
            pass
        else:
            noSel = noSel.refine('trigger', cut=[makeMultiPrimaryDatasetTriggerSelection(
                sample, self.triggersPerPrimaryDataset)])

        #### reapply JECs ###
        from bamboo.analysisutils import configureJets, configureType1MET
        isNotWorker = (self.args.distributed != "worker") 
        configureJets(tree._Jet, jet_algo_AK4,
                      jec=jec,
                      mayWriteCache= isNotWorker,
                      jecLevels = sampleCfg['jec_level'],
                      # cachedir='/afs/cern.ch/user/a/anmalara/workspace/WorkingArea/JME/jme-validation/JECs_2022/',
                      isMC=self.is_MC, backend = backend)
        configureJets(tree._FatJet, jet_algo_AK8,
                      jec=jec,
                      mayWriteCache= isNotWorker,
                      jecLevels = sampleCfg['jec_level'],
                      # cachedir='/afs/cern.ch/user/a/anmalara/workspace/WorkingArea/JME/jme-validation/JECs_2022/',
                      isMC=self.is_MC, backend = backend)

        # configureType1MET(tree._MET,
        #     jec="Summer16_07Aug2017_V20_MC",
        #     smear="Summer16_25nsV1_MC",
        #     jesUncertaintySources=["Total"],
        #     mayWriteCache= not isDriver,
        #     isMC=self.isMC(sample), backend=be)

        for calcProd in tree._Jet.calcProds:
            forceDefine(calcProd,noSel)

        return tree, noSel, backend, lumiArgs
