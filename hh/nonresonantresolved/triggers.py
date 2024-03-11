trigs_long = [
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bdl1d77_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25",  # asymmetric
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_3j20c_020jvt_bdl1d82_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25",  # asymmetric
    "HLT_j150_2j55_0eta290_020jvt_bdl1d70_pf_ftf_preselj80XX2j45b90_L1J85_3J30",  # symmetric
    "HLT_2j35c_020jvt_bdl1d60_2j35c_020jvt_pf_ftf_presel2j25XX2j25b85_L14J15p0ETA25",  # symmetric
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bdl1d77_pf_ftf_presel2c20XX2c20b85_L1MU8F_2J15_J20",  # asymmetric
    "HLT_j225_gsc300_bmv2c1070_split",  # run2 2017 1b
    "HLT_j110_gsc150_boffperf_split_2j35_gsc55_bmv2c1070_split_L1J85_3J30",  # run2 2017 2b1j
    "HLT_2j35_gsc55_bmv2c1050_split_ht300_L1HT190_J15s5pETA21",  # run2 2017 2bHT
    "HLT_2j15_gsc35_bmv2c1040_split_2j15_gsc35_boffperf_split_L14J15p0ETA25",  # run2 2017 2b2j
]

trigs_short = [
    "assym_2b2j_delayed",
    "assym_3b1j",
    "2b1j",
    "symm_2b2j",
    "asymm_2b2j_L1mu",
    "1b",
    "2b1j",
    "2bHT",
    "2b2j",
]

trigs_labels = [
    "Asymm 2b2j DL1d@77% (Delayed)",  # run3, only used when running on delayed stream files
    "Asymm 3b1j DL1d@82% (Main)",  # run3
    "2b1j DL1d@70% (Main)",  # run2 reoptimized
    "Symm 2b2j DL1d@60% (Main)",  # run2 reoptimized
    "Asymm 2b2j+L1mu DL1d@77%",  # run3, not studied yet (not used in analysis)
    "1b",  # run2 2017
    "2b1j",  # run2 2017
    "2bHT",  # run2 2017
    "2b2j",  # run2 2017
]

run3_main_stream_idx = [1, 2, 3]
run3_delayed_stream_idx = [0, 1, 2, 3]
run3_asymm_L1_jet_idx = [0, 1]
run3_asymm_L1_all_idx = [0, 1, 4]
run2_reoptimized_idx = [2, 3]
run2_legacy_idx = [5, 6, 7, 8]


def _get_triggers(idx):
    return [(trigs_long[i], trigs_short[i], trigs_labels[i]) for i in idx]


run3_main_stream = _get_triggers(run3_main_stream_idx)
run3_delayed_stream = _get_triggers(run3_delayed_stream_idx)
run3_asymm_L1_jet = _get_triggers(run3_asymm_L1_jet_idx)
run3_asymm_L1_all = _get_triggers(run3_asymm_L1_all_idx)
run2_reoptimized = _get_triggers(run2_reoptimized_idx)
run2_legacy = _get_triggers(run2_legacy_idx)
all = dict(zip(trigs_labels, trigs_long))

trig_sets = {
    "Run 2 legacy": run2_legacy,
    "Run 2 reoptimized": run2_reoptimized,
    "Main physics stream": run3_main_stream,
    "Main + delayed streams": run3_delayed_stream,
    "Asymm L1 jet": run3_asymm_L1_jet,
    "Asymm L1 all": run3_asymm_L1_all,
}


# ['runNumber', 'eventNumber', 'lumiBlock', 'mcEventWeights', 'averageInteractionsPerCrossing', 'actualInteractionsPerCrossing', 'mcChannelNumber', 'generatorWeight_NOSYS', 'PileupWeight_NOSYS', 'trigPassed_HLT_j150_gsc175_bmv2c1060_split_j45_gsc60_bmv2c1060_split', 'trigPassed_HLT_2j15_gsc35_bmv2c1040_split_2j15_gsc35_boffperf_split_L14J15p0ETA25', 'trigPassed_HLT_2j35_gsc55_bmv2c1050_split_ht300_L1HT190_J15s5pETA21', 'trigPassed_HLT_j225_gsc275_bmv2c1060_split', 'trigPassed_HLT_j390_a10t_lcw_jes_30smcINF_L1J100', 'trigPassed_HLT_2j35_gsc55_bmv2c1060_split_ht300_L1HT190_J15s5pETA21', 'trigPassed_HLT_j110_gsc150_boffperf_split_2j35_gsc55_bmv2c1070_split_L1J85_3J30', 'trigPassed_HLT_j175_gsc225_bmv2c1040_split', 'trigPassed_HLT_j460_a10t_lcw_jes_L1J100', 'trigPassed_HLT_j420_a10t_lcw_jes_40smcINF_L1J100', 'trigPassed_HLT_j225_gsc300_bmv2c1070_split', 'truth_H1_pdgId', 'truth_H2_pdgId', 'truth_children_fromH1_pdgId', 'truth_children_fromH2_pdgId', 'truth_H1_pt', 'truth_H1_eta', 'truth_H1_phi', 'truth_H1_m', 'truth_H2_pt', 'truth_H2_eta', 'truth_H2_phi', 'truth_H2_m', 'truth_children_fromH1_pt', 'truth_children_fromH1_eta', 'truth_children_fromH1_phi', 'truth_children_fromH1_m', 'truth_children_fromH2_pt', 'truth_children_fromH2_eta', 'truth_children_fromH2_phi', 'truth_children_fromH2_m', 'recojet_antikt4PFlow_NOSYS_pt', 'recojet_antikt4PFlow_NOSYS_eta', 'recojet_antikt4PFlow_NOSYS_phi', 'recojet_antikt4PFlow_NOSYS_m', 'recojet_antikt4PFlow_NOSYS_NNJvtPass', 'recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_FixedCutBEff_77', 'recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_FixedCutBEff_70', 'recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_FixedCutBEff_85', 'recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_Continuous', 'recojet_antikt4PFlow_NOSYS_ftag_select_GN120220509_FixedCutBEff_70', 'recojet_antikt4PFlow_NOSYS_ftag_select_GN120220509_FixedCutBEff_77', 'recojet_antikt4PFlow_NOSYS_ftag_select_GN120220509_FixedCutBEff_85', 'recojet_antikt4PFlow_NOSYS_ftag_select_GN120220509_Continuous', 'recojet_antikt4PFlow_NOSYS_NoBJetCalibMomentum_pt', 'recojet_antikt4PFlow_NOSYS_NoBJetCalibMomentum_eta', 'recojet_antikt4PFlow_NOSYS_NoBJetCalibMomentum_phi', 'recojet_antikt4PFlow_NOSYS_NoBJetCalibMomentum_m', 'recojet_antikt4PFlow_NOSYS_Jvt', 'recojet_antikt4PFlow_NOSYS_JvtRpt', 'recojet_antikt4PFlow_NOSYS_JVFCorr', 'recojet_antikt4PFlow_NOSYS_jvt_selection', 'recojet_antikt4PFlow_NOSYS_NNJvt', 'recojet_antikt4PFlow_NOSYS_NNJvtRpt', 'recojet_antikt4PFlow_NOSYS_HadronConeExclTruthLabelID', 'recojet_antikt4PFlow_NOSYS_nTopToBChildren', 'recojet_antikt4PFlow_NOSYS_nTopToWChildren', 'recojet_antikt4PFlow_NOSYS_parentHiggsParentsMask', 'recojet_antikt4PFlow_NOSYS_parentTopParentsMask', 'recojet_antikt4PFlow_NOSYS_GN2v00_pb', 'recojet_antikt4PFlow_NOSYS_GN2v00_pc', 'recojet_antikt4PFlow_NOSYS_GN2v00_pu', 'recojet_antikt4PFlow_NOSYS_DL1dv01_pb', 'recojet_antikt4PFlow_NOSYS_DL1dv01_pc', 'recojet_antikt4PFlow_NOSYS_DL1dv01_pu', 'truthjet_antikt4_pt', 'truthjet_antikt4_eta', 'truthjet_antikt4_phi', 'truthjet_antikt4_m', 'truthjet_antikt4_PartonTruthLabelID', 'truthjet_antikt4_HadronConeExclTruthLabelID', 'generatorWeight_GEN_Var3cDown', 'generatorWeight_GEN_Var3cUp', 'generatorWeight_GEN_hardHi', 'generatorWeight_GEN_hardLo', 'generatorWeight_GEN_isrPDFminus', 'generatorWeight_GEN_isrPDFplus', 'generatorWeight_GEN_isrmuRfac05_fsrmuRfac05', 'generatorWeight_GEN_isrmuRfac05_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac05_fsrmuRfac20', 'generatorWeight_GEN_isrmuRfac0625_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac075_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac0875_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac05', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac0625', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac075', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac0875', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac125', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac15', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac175', 'generatorWeight_GEN_isrmuRfac10_fsrmuRfac20', 'generatorWeight_GEN_isrmuRfac125_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac15_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac175_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac20_fsrmuRfac05', 'generatorWeight_GEN_isrmuRfac20_fsrmuRfac10', 'generatorWeight_GEN_isrmuRfac20_fsrmuRfac20', 'generatorWeight_GEN_lhapdf13100', 'generatorWeight_GEN_lhapdf25200', 'generatorWeight_GEN_lhapdf260000', 'generatorWeight_GEN_lhapdf265000', 'generatorWeight_GEN_lhapdf266000', 'generatorWeight_GEN_lhapdf90400', 'generatorWeight_GEN_lhapdf90401', 'generatorWeight_GEN_lhapdf90402', 'generatorWeight_GEN_lhapdf90403', 'generatorWeight_GEN_lhapdf90404', 'generatorWeight_GEN_lhapdf90405', 'generatorWeight_GEN_lhapdf90406', 'generatorWeight_GEN_lhapdf90407', 'generatorWeight_GEN_lhapdf90408', 'generatorWeight_GEN_lhapdf90409', 'generatorWeight_GEN_lhapdf90410', 'generatorWeight_GEN_lhapdf90411', 'generatorWeight_GEN_lhapdf90412', 'generatorWeight_GEN_lhapdf90413', 'generatorWeight_GEN_lhapdf90414', 'generatorWeight_GEN_lhapdf90415', 'generatorWeight_GEN_lhapdf90416', 'generatorWeight_GEN_lhapdf90417', 'generatorWeight_GEN_lhapdf90418', 'generatorWeight_GEN_lhapdf90419', 'generatorWeight_GEN_lhapdf90420', 'generatorWeight_GEN_lhapdf90421', 'generatorWeight_GEN_lhapdf90422', 'generatorWeight_GEN_lhapdf90423', 'generatorWeight_GEN_lhapdf90424', 'generatorWeight_GEN_lhapdf90425', 'generatorWeight_GEN_lhapdf90426', 'generatorWeight_GEN_lhapdf90427', 'generatorWeight_GEN_lhapdf90428', 'generatorWeight_GEN_lhapdf90429', 'generatorWeight_GEN_lhapdf90430', 'generatorWeight_GEN_lhapdf90431', 'generatorWeight_GEN_lhapdf90432', 'generatorWeight_GEN_renscfact05_facscfact05', 'generatorWeight_GEN_renscfact05_facscfact10', 'generatorWeight_GEN_renscfact10_facscfact05', 'generatorWeight_GEN_renscfact10_facscfact20', 'generatorWeight_GEN_renscfact20_facscfact10', 'generatorWeight_GEN_renscfact20_facscfact20', 'PileupWeight_PRW_DATASF__1down', 'PileupWeight_PRW_DATASF__1up']
