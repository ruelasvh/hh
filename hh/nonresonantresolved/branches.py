from .triggers import (
    trig_sets,
)

CAMPAIGNS = {
    2016: ["r13167", "r14859"],
    2017: ["r13144", "r14860"],
    2018: ["r13145", "r14861"],
    2022: ["r14622", "r14932"],
    2023: ["r15224"],
}

BASE_ALIASES = {
    "event_number": "eventNumber",
    "run_number": "runNumber",
}

# ['truthjet_antikt10SoftDrop_eta', 'truthjet_antikt10SoftDrop_m', 'truthjet_antikt10SoftDrop_phi', 'truthjet_antikt10SoftDrop_pt', 'truthjet_antikt4_HadronConeExclTruthLabelID', 'truthjet_antikt4_PartonTruthLabelID', 'truthjet_antikt4_eta', 'truthjet_antikt4_m', 'truthjet_antikt4_phi', 'truthjet_antikt4_pt', 'RandomRunNumber', 'actualInteractionsPerCrossing', 'averageInteractionsPerCrossing', 'dataTakingYear', 'eventNumber', 'lumiBlock', 'mcChannelNumber', 'mcEventWeights', 'passRelativeDeltaRToVRJetCutUFO', 'bbbb_resolved_DL1dv01_FixedCutBEff_70_DeltaR12', 'bbbb_resolved_DL1dv01_FixedCutBEff_77_DeltaR12', 'bbbb_resolved_DL1dv01_FixedCutBEff_85_DeltaR12', 'bbbb_resolved_GN2v01_FixedCutBEff_65_DeltaR12', 'bbbb_resolved_GN2v01_FixedCutBEff_70_DeltaR12', 'bbbb_resolved_GN2v01_FixedCutBEff_77_DeltaR12', 'bbbb_resolved_GN2v01_FixedCutBEff_85_DeltaR12', 'bbbb_resolved_DL1dv01_FixedCutBEff_70_DeltaR13', 'bbbb_resolved_DL1dv01_FixedCutBEff_77_DeltaR13', 'bbbb_resolved_DL1dv01_FixedCutBEff_85_DeltaR13', 'bbbb_resolved_GN2v01_FixedCutBEff_65_DeltaR13', 'bbbb_resolved_GN2v01_FixedCutBEff_70_DeltaR13', 'bbbb_resolved_GN2v01_FixedCutBEff_77_DeltaR13', 'bbbb_resolved_GN2v01_FixedCutBEff_85_DeltaR13', 'bbbb_resolved_DL1dv01_FixedCutBEff_70_DeltaR14', 'bbbb_resolved_DL1dv01_FixedCutBEff_77_DeltaR14', 'bbbb_resolved_DL1dv01_FixedCutBEff_85_DeltaR14', 'bbbb_resolved_GN2v01_FixedCutBEff_65_DeltaR14', 'bbbb_resolved_GN2v01_FixedCutBEff_70_DeltaR14', 'bbbb_resolved_GN2v01_FixedCutBEff_77_DeltaR14', 'bbbb_resolved_GN2v01_FixedCutBEff_85_DeltaR14', 'bbbb_resolved_DL1dv01_FixedCutBEff_70_DeltaR23', 'bbbb_resolved_DL1dv01_FixedCutBEff_77_DeltaR23', 'bbbb_resolved_DL1dv01_FixedCutBEff_85_DeltaR23', 'bbbb_resolved_GN2v01_FixedCutBEff_65_DeltaR23', 'bbbb_resolved_GN2v01_FixedCutBEff_70_DeltaR23', 'bbbb_resolved_GN2v01_FixedCutBEff_77_DeltaR23', 'bbbb_resolved_GN2v01_FixedCutBEff_85_DeltaR23', 'bbbb_resolved_DL1dv01_FixedCutBEff_70_DeltaR24', 'bbbb_resolved_DL1dv01_FixedCutBEff_77_DeltaR24', 'bbbb_resolved_DL1dv01_FixedCutBEff_85_DeltaR24', 'bbbb_resolved_GN2v01_FixedCutBEff_65_DeltaR24', 'bbbb_resolved_GN2v01_FixedCutBEff_70_DeltaR24', 'bbbb_resolved_GN2v01_FixedCutBEff_77_DeltaR24', 'bbbb_resolved_GN2v01_FixedCutBEff_85_DeltaR24', 'bbbb_resolved_DL1dv01_FixedCutBEff_70_DeltaR34', 'bbbb_resolved_DL1dv01_FixedCutBEff_77_DeltaR34', 'bbbb_resolved_DL1dv01_FixedCutBEff_85_DeltaR34', 'bbbb_resolved_GN2v01_FixedCutBEff_65_DeltaR34', 'bbbb_resolved_GN2v01_FixedCutBEff_70_DeltaR34', 'bbbb_resolved_GN2v01_FixedCutBEff_77_DeltaR34', 'bbbb_resolved_GN2v01_FixedCutBEff_85_DeltaR34', 'bbbb_resolved_DL1dv01_FixedCutBEff_70_h1_m', 'bbbb_resolved_DL1dv01_FixedCutBEff_77_h1_m', 'bbbb_resolved_DL1dv01_FixedCutBEff_85_h1_m', 'bbbb_resolved_GN2v01_FixedCutBEff_65_h1_m', 'bbbb_resolved_GN2v01_FixedCutBEff_70_h1_m', 'bbbb_resolved_GN2v01_FixedCutBEff_77_h1_m', 'bbbb_resolved_GN2v01_FixedCutBEff_85_h1_m', 'bbbb_resolved_DL1dv01_FixedCutBEff_70_h2_m', 'bbbb_resolved_DL1dv01_FixedCutBEff_77_h2_m', 'bbbb_resolved_DL1dv01_FixedCutBEff_85_h2_m', 'bbbb_resolved_GN2v01_FixedCutBEff_65_h2_m', 'bbbb_resolved_GN2v01_FixedCutBEff_70_h2_m', 'bbbb_resolved_GN2v01_FixedCutBEff_77_h2_m', 'bbbb_resolved_GN2v01_FixedCutBEff_85_h2_m', 'bbbb_resolved_DL1dv01_FixedCutBEff_70_hh_m', 'bbbb_resolved_DL1dv01_FixedCutBEff_77_hh_m', 'bbbb_resolved_DL1dv01_FixedCutBEff_85_hh_m', 'bbbb_resolved_GN2v01_FixedCutBEff_65_hh_m', 'bbbb_resolved_GN2v01_FixedCutBEff_70_hh_m', 'bbbb_resolved_GN2v01_FixedCutBEff_77_hh_m', 'bbbb_resolved_GN2v01_FixedCutBEff_85_hh_m', 'runNumber', 'trigPassed_HLT_2j330_35smcINF_a10sd_cssk_pf_jes_ftf_presel2j225_L1J100', 'trigPassed_HLT_j420_35smcINF_a10sd_cssk_pf_jes_ftf_preselj225_L1J100', 'trigPassed_HLT_j75c_020jvt_j50c_020jvt_j25c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bgn177_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25', 'trigPassed_HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bgn177_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25', 'truth_H1_eta', 'truth_H1_m', 'truth_H1_pdgId', 'truth_H1_phi', 'truth_H1_pt', 'truth_H2_eta', 'truth_H2_m', 'truth_H2_pdgId', 'truth_H2_phi', 'truth_H2_pt', 'truth_HH_eta', 'truth_HH_m', 'truth_HH_phi', 'truth_HH_pt', 'truth_children_fromH1_eta', 'truth_children_fromH1_m', 'truth_children_fromH1_pdgId', 'truth_children_fromH1_phi', 'truth_children_fromH1_pt', 'truth_children_fromH2_eta', 'truth_children_fromH2_m', 'truth_children_fromH2_pdgId', 'truth_children_fromH2_phi', 'truth_children_fromH2_pt', 'truth_initial_children_fromH1_eta', 'truth_initial_children_fromH1_m', 'truth_initial_children_fromH1_pdgId', 'truth_initial_children_fromH1_phi', 'truth_initial_children_fromH1_pt', 'truth_initial_children_fromH2_eta', 'truth_initial_children_fromH2_m', 'truth_initial_children_fromH2_pdgId', 'truth_initial_children_fromH2_phi', 'truth_initial_children_fromH2_pt', 'recojet_antikt10UFO_ECF1_NOSYS', 'recojet_antikt10UFO_ECF2_NOSYS', 'recojet_antikt10UFO_ECF3_NOSYS', 'recojet_antikt10UFO_GN2Xv01_phbb_NOSYS', 'recojet_antikt10UFO_GN2Xv01_phcc_NOSYS', 'recojet_antikt10UFO_GN2Xv01_pqcd_NOSYS', 'recojet_antikt10UFO_GN2Xv01_ptop_NOSYS', 'recojet_antikt10UFO_GhostBHadronsFinalCount_NOSYS', 'recojet_antikt10UFO_GhostCHadronsFinalCount_NOSYS', 'recojet_antikt10UFO_R10TruthLabel_R21Precision_2022v1_NOSYS', 'recojet_antikt10UFO_R10TruthLabel_R22v1_NOSYS', 'recojet_antikt10UFO_Split12_NOSYS', 'recojet_antikt10UFO_Split23_NOSYS', 'recojet_antikt10UFO_Tau1_wta_NOSYS', 'recojet_antikt10UFO_Tau2_wta_NOSYS', 'recojet_antikt10UFO_Tau3_wta_NOSYS', 'recojet_antikt10UFO_VRTrackJetsTruthLabel_NOSYS', 'recojet_antikt10UFO_eta_NOSYS', 'recojet_antikt10UFO_goodVRTrackJets_NOSYS', 'recojet_antikt10UFO_leadingVRTrackJetsDeltaR12_NOSYS', 'recojet_antikt10UFO_leadingVRTrackJetsDeltaR13_NOSYS', 'recojet_antikt10UFO_leadingVRTrackJetsDeltaR32_NOSYS', 'recojet_antikt10UFO_leadingVRTrackJetsEta_NOSYS', 'recojet_antikt10UFO_leadingVRTrackJetsM_NOSYS', 'recojet_antikt10UFO_leadingVRTrackJetsPhi_NOSYS', 'recojet_antikt10UFO_leadingVRTrackJetsPt_NOSYS', 'recojet_antikt10UFO_m_NOSYS', 'recojet_antikt10UFO_minRelativeDeltaRToVRJet_NOSYS', 'recojet_antikt10UFO_nTopToBChildren_NOSYS', 'recojet_antikt10UFO_nTopToWChildren_NOSYS', 'recojet_antikt10UFO_parentHiggsNMatchedChildren_NOSYS', 'recojet_antikt10UFO_parentHiggsParentsMask_NOSYS', 'recojet_antikt10UFO_parentScalarNMatchedChildren_NOSYS', 'recojet_antikt10UFO_parentScalarParentsMask_NOSYS', 'recojet_antikt10UFO_parentTopNMatchedChildren_NOSYS', 'recojet_antikt10UFO_parentTopParentsMask_NOSYS', 'recojet_antikt10UFO_parentZParentsMask_NOSYS', 'recojet_antikt10UFO_passesOR_NOSYS', 'recojet_antikt10UFO_phi_NOSYS', 'recojet_antikt10UFO_pt_NOSYS', 'recojet_antikt4PFlow_DL1dv01_pb_NOSYS', 'recojet_antikt4PFlow_DL1dv01_pc_NOSYS', 'recojet_antikt4PFlow_DL1dv01_pu_NOSYS', 'recojet_antikt4PFlow_GN2v00_pb_NOSYS', 'recojet_antikt4PFlow_GN2v00_pc_NOSYS', 'recojet_antikt4PFlow_GN2v00_pu_NOSYS', 'recojet_antikt4PFlow_GN2v01_pb_NOSYS', 'recojet_antikt4PFlow_GN2v01_pc_NOSYS', 'recojet_antikt4PFlow_GN2v01_ptau_NOSYS', 'recojet_antikt4PFlow_GN2v01_pu_NOSYS', 'recojet_antikt4PFlow_HadronConeExclTruthLabelID_NOSYS', 'recojet_antikt4PFlow_JVFCorr_NOSYS', 'recojet_antikt4PFlow_Jvt_NOSYS', 'recojet_antikt4PFlow_JvtRpt_NOSYS', 'recojet_antikt4PFlow_NNJvt_NOSYS', 'recojet_antikt4PFlow_NNJvtRpt_NOSYS', 'recojet_antikt4PFlow_NoBJetCalibMomentum_eta_NOSYS', 'recojet_antikt4PFlow_NoBJetCalibMomentum_m_NOSYS', 'recojet_antikt4PFlow_NoBJetCalibMomentum_phi_NOSYS', 'recojet_antikt4PFlow_NoBJetCalibMomentum_pt_NOSYS', 'recojet_antikt4PFlow_PartonTruthLabelID_NOSYS', 'recojet_antikt4PFlow_bJetTruthDR_NOSYS', 'recojet_antikt4PFlow_bJetTruthPt_NOSYS', 'recojet_antikt4PFlow_eta_NOSYS', 'recojet_antikt4PFlow_ftag_effSF_DL1dv01_FixedCutBEff_70_NOSYS', 'recojet_antikt4PFlow_ftag_effSF_DL1dv01_FixedCutBEff_77_NOSYS', 'recojet_antikt4PFlow_ftag_effSF_DL1dv01_FixedCutBEff_85_NOSYS', 'recojet_antikt4PFlow_ftag_select_DL1dv01_FixedCutBEff_70_NOSYS', 'recojet_antikt4PFlow_ftag_select_DL1dv01_FixedCutBEff_77_NOSYS', 'recojet_antikt4PFlow_ftag_select_DL1dv01_FixedCutBEff_85_NOSYS', 'recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_65_NOSYS', 'recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_70_NOSYS', 'recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_77_NOSYS', 'recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_85_NOSYS', 'recojet_antikt4PFlow_jvt_effSF_NOSYS', 'recojet_antikt4PFlow_jvt_selection_NOSYS', 'recojet_antikt4PFlow_m_NOSYS', 'recojet_antikt4PFlow_muonCorrPt_NOSYS', 'recojet_antikt4PFlow_nTopToBChildren_NOSYS', 'recojet_antikt4PFlow_nTopToWChildren_NOSYS', 'recojet_antikt4PFlow_n_muons_NOSYS', 'recojet_antikt4PFlow_parentHiggsParentsMask_NOSYS', 'recojet_antikt4PFlow_parentScalarParentsMask_NOSYS', 'recojet_antikt4PFlow_parentTopParentsMask_NOSYS', 'recojet_antikt4PFlow_parentZParentsMask_NOSYS', 'recojet_antikt4PFlow_passesOR_NOSYS', 'recojet_antikt4PFlow_phi_NOSYS', 'recojet_antikt4PFlow_pt_NOSYS', 'recojet_antikt4PFlow_uncorrPt_NOSYS', 'PileupWeight_NOSYS', 'ftag_effSF_DL1dv01_FixedCutBEff_70_NOSYS', 'ftag_effSF_DL1dv01_FixedCutBEff_77_NOSYS', 'ftag_effSF_DL1dv01_FixedCutBEff_85_NOSYS', 'generatorWeight_NOSYS', 'jvt_effSF_NOSYS', 'PileupWeight_PRW_DATASF__1down', 'PileupWeight_PRW_DATASF__1up', 'met_met_NOSYS', 'met_phi_NOSYS', 'met_sumet_NOSYS']

JET_ALIASES = {
    "jet_pt": "recojet_antikt4PFlow_pt_NOSYS",
    "jet_eta": "recojet_antikt4PFlow_eta_NOSYS",
    "jet_phi": "recojet_antikt4PFlow_phi_NOSYS",
    "jet_mass": "recojet_antikt4PFlow_m_NOSYS",
    "jet_NNJvt": "recojet_antikt4PFlow_NNJvt_NOSYS",
    "jet_jvttag": "recojet_antikt4PFlow_jvt_selection_NOSYS",  # also recojet_antikt4PFlow_NOSYS_NNJvt, recojet_antikt4PFlow_NOSYS_Jvt, recojet_antikt4PFlow_NOSYS_JvtRpt, recojet_antikt4PFlow_NOSYS_NNJvtRpt, recojet_antikt4PFlow_NOSYS_jvt_selection
    "jet_truth_label_ID": "recojet_antikt4PFlow_HadronConeExclTruthLabelID_NOSYS",
    "jet_btag_DL1dv01_70": "recojet_antikt4PFlow_ftag_select_DL1dv01_FixedCutBEff_70_NOSYS",
    "jet_btag_DL1dv01_77": "recojet_antikt4PFlow_ftag_select_DL1dv01_FixedCutBEff_77_NOSYS",
    "jet_btag_DL1dv01_85": "recojet_antikt4PFlow_ftag_select_DL1dv01_FixedCutBEff_85_NOSYS",
    "jet_btag_DL1dv01_pb": "recojet_antikt4PFlow_DL1dv01_pb_NOSYS",
    "jet_btag_DL1dv01_pc": "recojet_antikt4PFlow_DL1dv01_pc_NOSYS",
    "jet_btag_DL1dv01_pu": "recojet_antikt4PFlow_DL1dv01_pu_NOSYS",
    "jet_btag_GN2v01_65": "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_65_NOSYS",
    "jet_btag_GN2v01_70": "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_70_NOSYS",
    "jet_btag_GN2v01_77": "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_77_NOSYS",
    "jet_btag_GN2v01_85": "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_85_NOSYS",
    "jet_btag_GN2v01_pb": "recojet_antikt4PFlow_GN2v01_pb_NOSYS",
    "jet_btag_GN2v01_pc": "recojet_antikt4PFlow_GN2v01_pc_NOSYS",
    "jet_btag_GN2v01_pu": "recojet_antikt4PFlow_GN2v01_pu_NOSYS",
}

MC_ALIASES = {
    "mc_event_weights": "mcEventWeights",
    "pileup_weight": "PileupWeight_NOSYS",
    "truth_jet_H_parent_mask": "recojet_antikt4PFlow_parentHiggsParentsMask_NOSYS",
    "truth_jet_label_ID": "truthjet_antikt4_HadronConeExclTruthLabelID",
    "h1_truth_ID": "truth_H1_pdgId",
    "h1_truth_pt": "truth_H1_pt",
    "h1_truth_eta": "truth_H1_eta",
    "h1_truth_phi": "truth_H1_phi",
    "h1_truth_mass": "truth_H1_m",
    "h2_truth_ID": "truth_H2_pdgId",
    "h2_truth_pt": "truth_H2_pt",
    "h2_truth_eta": "truth_H2_eta",
    "h2_truth_phi": "truth_H2_phi",
    "h2_truth_mass": "truth_H2_m",
    "hh_truth_eta": "truth_HH_eta",
    "hh_truth_phi": "truth_HH_phi",
    "hh_truth_mass": "truth_HH_m",
    "hh_truth_pt": "truth_HH_pt",
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
            for year, rtags in CAMPAIGNS.items():
                for rtag in rtags:
                    if rtag in sample_metadata["logicalDatasetName"]:
                        trig_set = f"{trig_set} {year}"
        aliases.update(
            {
                f"trig_{trig_short}": f"trigPassed_{trig_long}"
                for trig_long, trig_short, _ in trig_sets[trig_set]
            }
        )
    return aliases
