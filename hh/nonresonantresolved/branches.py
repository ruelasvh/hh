from .triggers import (
    trig_sets,
)

CAMPAIGNS = {"r13167": [2016], "r13144": [2017], "r13145": [2018]}

BASE_ALIASES = {
    "event_number": "eventNumber",
    "run_number": "runNumber",
}

JET_ALIASES = {
    "jet_pt": "recojet_antikt4PFlow_NOSYS_pt",
    "jet_eta": "recojet_antikt4PFlow_NOSYS_eta",
    "jet_phi": "recojet_antikt4PFlow_NOSYS_phi",
    "jet_mass": "recojet_antikt4PFlow_NOSYS_m",
    "jet_NNJvt": "recojet_antikt4PFlow_NOSYS_NNJvt",
    "jet_jvttag": "recojet_antikt4PFlow_NOSYS_NNJvtPass",
    "jet_btag_DL1dv01_70": "recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_FixedCutBEff_70",
    "jet_btag_DL1dv01_77": "recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_FixedCutBEff_77",
    "jet_btag_DL1dv01_85": "recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_FixedCutBEff_85",
    "jet_btag_DL1dv01_pb": "recojet_antikt4PFlow_NOSYS_DL1dv01_pb",
    "jet_btag_DL1dv01_pc": "recojet_antikt4PFlow_NOSYS_DL1dv01_pc",
    "jet_btag_DL1dv01_pu": "recojet_antikt4PFlow_NOSYS_DL1dv01_pu",
    "jet_btag_GN2v01_60": "recojet_antikt4PFlow_NOSYS_ftag_select_GN2v01_FixedCutBEff_60",
    "jet_btag_GN2v01_70": "recojet_antikt4PFlow_NOSYS_ftag_select_GN2v01_FixedCutBEff_70",
    "jet_btag_GN2v01_77": "recojet_antikt4PFlow_NOSYS_ftag_select_GN2v01_FixedCutBEff_77",
    "jet_btag_GN2v01_85": "recojet_antikt4PFlow_NOSYS_ftag_select_GN2v01_FixedCutBEff_85",
    "jet_btag_GN2v01_pb": "recojet_antikt4PFlow_NOSYS_GN2v01_pb",
    "jet_btag_GN2v01_pc": "recojet_antikt4PFlow_NOSYS_GN2v01_pc",
    "jet_btag_GN2v01_pu": "recojet_antikt4PFlow_NOSYS_GN2v01_pu",
}

MC_ALIASES = {
    "mc_event_weights": "mcEventWeights",
    "pileup_weight": "PileupWeight_NOSYS",
    "jet_truth_H_parents": "recojet_antikt4PFlow_NOSYS_parentHiggsParentsMask",
    "jet_truth_ID": "truthjet_antikt4_HadronConeExclTruthLabelID",
    # "jet_btag_sf_DL1dv01_70": "recojet_antikt4PFlow_NOSYS_ftag_effSF_DL1dv01_FixedCutBEff_70",
    # "jet_btag_sf_DL1dv01_77": "recojet_antikt4PFlow_NOSYS_ftag_effSF_DL1dv01_FixedCutBEff_77",
    # "jet_btag_sf_DL1dv01_85": "recojet_antikt4PFlow_NOSYS_ftag_effSF_DL1dv01_FixedCutBEff_85",
}


def get_branch_aliases(is_mc=False, trig_set=None, sample_metadata=None):
    aliases = {**BASE_ALIASES}
    aliases.update(
        {
            key: value if is_mc else value.replace("antikt4", "antikt4PFlow")
            for key, value in JET_ALIASES.items()
        }
    )
    if is_mc:
        aliases.update(MC_ALIASES)

    if trig_set:
        if sample_metadata:
            for c, y in CAMPAIGNS.items():
                if c in sample_metadata["logicalDatasetName"]:
                    year = y[0]
                    trig_set = f"{trig_set} {year}"
        aliases.update(
            {
                f"trig_{trig_short}": f"trigPassed_{trig_long}"
                for trig_long, trig_short, _ in trig_sets[trig_set]
            }
        )
    return aliases


# Example branches
# ['runNumber', 'eventNumber', 'lumiBlock', 'mcEventWeights', 'averageInteractionsPerCrossing', 'actualInteractionsPerCrossing', 'mcChannelNumber', 'generatorWeight_NOSYS', 'PileupWeight_NOSYS', 'trigPassed_HLT_j150_gsc175_bmv2c1060_split_j45_gsc60_bmv2c1060_split', 'trigPassed_HLT_2j15_gsc35_bmv2c1040_split_2j15_gsc35_boffperf_split_L14J15p0ETA25', 'trigPassed_HLT_2j35_gsc55_bmv2c1050_split_ht300_L1HT190_J15s5pETA21', 'trigPassed_HLT_j225_gsc275_bmv2c1060_split', 'trigPassed_HLT_j390_a10t_lcw_jes_30smcINF_L1J100', 'trigPassed_HLT_2j35_gsc55_bmv2c1060_split_ht300_L1HT190_J15s5pETA21', 'trigPassed_HLT_j110_gsc150_boffperf_split_2j35_gsc55_bmv2c1070_split_L1J85_3J30', 'trigPassed_HLT_j175_gsc225_bmv2c1040_split', 'trigPassed_HLT_j460_a10t_lcw_jes_L1J100', 'trigPassed_HLT_j420_a10t_lcw_jes_40smcINF_L1J100', 'trigPassed_HLT_j225_gsc300_bmv2c1070_split', 'truth_H1_pdgId', 'truth_H2_pdgId', 'truth_children_fromH1_pdgId', 'truth_children_fromH2_pdgId', 'truth_H1_pt', 'truth_H1_eta', 'truth_H1_phi', 'truth_H1_m', 'truth_H2_pt', 'truth_H2_eta', 'truth_H2_phi', 'truth_H2_m', 'truth_children_fromH1_pt', 'truth_children_fromH1_eta', 'truth_children_fromH1_phi', 'truth_children_fromH1_m', 'truth_children_fromH2_pt', 'truth_children_fromH2_eta', 'truth_children_fromH2_phi', 'truth_children_fromH2_m', 'recojet_antikt4PFlow_NOSYS_pt', 'recojet_antikt4PFlow_NOSYS_eta', 'recojet_antikt4PFlow_NOSYS_phi', 'recojet_antikt4PFlow_NOSYS_m', 'recojet_antikt4PFlow_NOSYS_NNJvtPass', 'recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_FixedCutBEff_77', 'recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_FixedCutBEff_70', 'recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_FixedCutBEff_85', 'recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_Continuous', 'recojet_antikt4PFlow_NOSYS_ftag_select_GN120220509_FixedCutBEff_70', 'recojet_antikt4PFlow_NOSYS_ftag_select_GN120220509_FixedCutBEff_77', 'recojet_antikt4PFlow_NOSYS_ftag_select_GN120220509_FixedCutBEff_85', 'recojet_antikt4PFlow_NOSYS_ftag_select_GN120220509_Continuous', 'recojet_antikt4PFlow_NOSYS_NoBJetCalibMomentum_pt', 'recojet_antikt4PFlow_NOSYS_NoBJetCalibMomentum_eta', 'recojet_antikt4PFlow_NOSYS_NoBJetCalibMomentum_phi', 'recojet_antikt4PFlow_NOSYS_NoBJetCalibMomentum_m', 'recojet_antikt4PFlow_NOSYS_Jvt', 'recojet_antikt4PFlow_NOSYS_JvtRpt', 'recojet_antikt4PFlow_NOSYS_JVFCorr', 'recojet_antikt4PFlow_NOSYS_jvt_selection', 'recojet_antikt4PFlow_NOSYS_NNJvt', 'recojet_antikt4PFlow_NOSYS_NNJvtRpt', 'recojet_antikt4PFlow_NOSYS_HadronConeExclTruthLabelID', 'recojet_antikt4PFlow_NOSYS_nTopToBChildren', 'recojet_antikt4PFlow_NOSYS_nTopToWChildren', 'recojet_antikt4PFlow_NOSYS_parentHiggsParentsMask', 'recojet_antikt4PFlow_NOSYS_parentTopParentsMask', 'recojet_antikt4PFlow_NOSYS_GN2v00_pb', 'recojet_antikt4PFlow_NOSYS_GN2v00_pc', 'recojet_antikt4PFlow_NOSYS_GN2v00_pu', 'recojet_antikt4PFlow_NOSYS_DL1dv01_pb', 'recojet_antikt4PFlow_NOSYS_DL1dv01_pc', 'recojet_antikt4PFlow_NOSYS_DL1dv01_pu', 'truthjet_antikt4_pt', 'truthjet_antikt4_eta', 'truthjet_antikt4_phi', 'truthjet_antikt4_m', 'truthjet_antikt4_PartonTruthLabelID', 'truthjet_antikt4_HadronConeExclTruthLabelID', 'generatorWeight_GEN_Var3cDown', 'generatorWeight_GEN_Var3cUp', 'generatorWeight_GEN_hardHi', 'generatorWeight_GEN_hardLo', 'generatorWeight_GEN_isrPDFminus', 'generatorWeight_GEN_isrPDFplus', 'generatorWeight_GEN_isrmuRfac05_fsrmuRfac05', 'generatorWeight_GEN_isrmuRfac05_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac05_fsrmuRfac20', 'generatorWeight_GEN_isrmuRfac0625_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac075_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac0875_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac05', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac0625', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac075', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac0875', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac125', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac15', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac175', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac20', 'generatorWeight_GEN_isrmuRfac125_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac15_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac175_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac20_fsrmuRfac05', 'generatorWeight_GEN_isrmuRfac20_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac20_fsrmuRfac20', 'generatorWeight_GEN_lhapdf13100', 'generatorWeight_GEN_lhapdf25200', 'generatorWeight_GEN_lhapdf260000', 'generatorWeight_GEN_lhapdf265000', 'generatorWeight_GEN_lhapdf266000', 'generatorWeight_GEN_lhapdf90400', 'generatorWeight_GEN_lhapdf90401', 'generatorWeight_GEN_lhapdf90402', 'generatorWeight_GEN_lhapdf90403', 'generatorWeight_GEN_lhapdf90404', 'generatorWeight_GEN_lhapdf90405', 'generatorWeight_GEN_lhapdf90406', 'generatorWeight_GEN_lhapdf90407', 'generatorWeight_GEN_lhapdf90408', 'generatorWeight_GEN_lhapdf90409', 'generatorWeight_GEN_lhapdf90410', 'generatorWeight_GEN_lhapdf90411', 'generatorWeight_GEN_lhapdf90412', 'generatorWeight_GEN_lhapdf90413', 'generatorWeight_GEN_lhapdf90414', 'generatorWeight_GEN_lhapdf90415', 'generatorWeight_GEN_lhapdf90416', 'generatorWeight_GEN_lhapdf90417', 'generatorWeight_GEN_lhapdf90418', 'generatorWeight_GEN_lhapdf90419', 'generatorWeight_GEN_lhapdf90420', 'generatorWeight_GEN_lhapdf90421', 'generatorWeight_GEN_lhapdf90422', 'generatorWeight_GEN_lhapdf90423', 'generatorWeight_GEN_lhapdf90424', 'generatorWeight_GEN_lhapdf90425', 'generatorWeight_GEN_lhapdf90426', 'generatorWeight_GEN_lhapdf90427', 'generatorWeight_GEN_lhapdf90428', 'generatorWeight_GEN_lhapdf90429', 'generatorWeight_GEN_lhapdf90430', 'generatorWeight_GEN_lhapdf90431', 'generatorWeight_GEN_lhapdf90432', 'generatorWeight_GEN_renscfact05_facscfact05', 'generatorWeight_GEN_renscfact05_facscfact10', 'generatorWeight_GEN_renscfact10_facscfact05', 'generatorWeight_GEN_renscfact10_facscfact20', 'generatorWeight_GEN_renscfact20_facscfact10', 'generatorWeight_GEN_renscfact20_facscfact20', 'PileupWeight_PRW_DATASF__1down', 'PileupWeight_PRW_DATASF__1up']
