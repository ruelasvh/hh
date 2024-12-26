from .triggers import trig_sets as trigger_sets

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

# MC23 aliases
# JET_ALIASES = {
#     "jet_pt": "recojet_antikt4PFlow_pt_NOSYS",
#     "jet_eta": "recojet_antikt4PFlow_eta_NOSYS",
#     "jet_phi": "recojet_antikt4PFlow_phi_NOSYS",
#     "jet_mass": "recojet_antikt4PFlow_m_NOSYS",
#     "jet_NNJvt": "recojet_antikt4PFlow_NNJvt_NOSYS",
#     "jet_jvttag": "recojet_antikt4PFlow_jvt_selection_NOSYS",  # also recojet_antikt4PFlow_NOSYS_NNJvt, recojet_antikt4PFlow_NOSYS_Jvt, recojet_antikt4PFlow_NOSYS_JvtRpt, recojet_antikt4PFlow_NOSYS_NNJvtRpt, recojet_antikt4PFlow_NOSYS_jvt_selection
#     "jet_truth_H_parent_mask": "recojet_antikt4PFlow_parentHiggsParentsMask_NOSYS",
#     "jet_truth_label_ID": "recojet_antikt4PFlow_HadronConeExclTruthLabelID_NOSYS",
#     "jet_btag_DL1dv01_70": "recojet_antikt4PFlow_ftag_select_DL1dv01_FixedCutBEff_70_NOSYS",
#     "jet_btag_DL1dv01_77": "recojet_antikt4PFlow_ftag_select_DL1dv01_FixedCutBEff_77_NOSYS",
#     "jet_btag_DL1dv01_85": "recojet_antikt4PFlow_ftag_select_DL1dv01_FixedCutBEff_85_NOSYS",
#     "jet_btag_DL1dv01_pb": "recojet_antikt4PFlow_DL1dv01_pb_NOSYS",
#     "jet_btag_DL1dv01_pc": "recojet_antikt4PFlow_DL1dv01_pc_NOSYS",
#     "jet_btag_DL1dv01_pu": "recojet_antikt4PFlow_DL1dv01_pu_NOSYS",
#     "jet_btag_DL1dv01_ptau": "recojet_antikt4PFlow_DL1dv01_pu_NOSYS",
#     "jet_btag_GN2v01_65": "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_65_NOSYS",
#     "jet_btag_GN2v01_70": "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_70_NOSYS",
#     "jet_btag_GN2v01_77": "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_77_NOSYS",
#     "jet_btag_GN2v01_85": "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_85_NOSYS",
#     "jet_btag_GN2v01_pb": "recojet_antikt4PFlow_GN2v01_pb_NOSYS",
#     "jet_btag_GN2v01_pc": "recojet_antikt4PFlow_GN2v01_pc_NOSYS",
#     "jet_btag_GN2v01_pu": "recojet_antikt4PFlow_GN2v01_pu_NOSYS",
#     "jet_btag_GN2v01_ptau": "recojet_antikt4PFlow_GN2v01_ptau_NOSYS",
# }

# MC20 aliases
JET_ALIASES = {
    "jet_pt": "recojet_antikt4PFlow_NOSYS_pt",
    "jet_eta": "recojet_antikt4PFlow_NOSYS_eta",
    "jet_phi": "recojet_antikt4PFlow_NOSYS_phi",
    "jet_mass": "recojet_antikt4PFlow_NOSYS_m",
    "jet_NNJvt": "recojet_antikt4PFlow_NOSYS_NNJvt",
    "jet_jvttag": "recojet_antikt4PFlow_NOSYS_jvt_selection",  # also recojet_antikt4PFlow_NOSYS_NNJvt, recojet_antikt4PFlow_NOSYS_Jvt, recojet_antikt4PFlow_NOSYS_JvtRpt, recojet_antikt4PFlow_NOSYS_NNJvtRpt, recojet_antikt4PFlow_NOSYS_jvt_selection
    "jet_truth_H_parent_mask": "recojet_antikt4PFlow_NOSYS_parentHiggsParentsMask",
    "jet_truth_label_ID": "recojet_antikt4PFlow_NOSYS_HadronConeExclTruthLabelID",
    "jet_btag_DL1dv01_70": "recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_FixedCutBEff_70",
    "jet_btag_DL1dv01_77": "recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_FixedCutBEff_77",
    "jet_btag_DL1dv01_85": "recojet_antikt4PFlow_NOSYS_ftag_select_DL1dv01_FixedCutBEff_85",
    "jet_btag_DL1dv01_pb": "recojet_antikt4PFlow_NOSYS_DL1dv01_pb",
    "jet_btag_DL1dv01_pc": "recojet_antikt4PFlow_NOSYS_DL1dv01_pc",
    "jet_btag_DL1dv01_pu": "recojet_antikt4PFlow_NOSYS_DL1dv01_pu",
    "jet_btag_DL1dv01_ptau": "recojet_antikt4PFlow_NOSYS_DL1dv01_pu",
    # "jet_btag_GN2v01_65": "recojet_antikt4PFlow_NOSYS_ftag_select_GN2v01_FixedCutBEff_65",
    "jet_btag_GN2v01_70": "recojet_antikt4PFlow_NOSYS_ftag_select_GN2v01_FixedCutBEff_70",
    "jet_btag_GN2v01_77": "recojet_antikt4PFlow_NOSYS_ftag_select_GN2v01_FixedCutBEff_77",
    "jet_btag_GN2v01_85": "recojet_antikt4PFlow_NOSYS_ftag_select_GN2v01_FixedCutBEff_85",
    "jet_btag_GN2v01_pb": "recojet_antikt4PFlow_NOSYS_GN2v01_pb",
    "jet_btag_GN2v01_pc": "recojet_antikt4PFlow_NOSYS_GN2v01_pc",
    "jet_btag_GN2v01_pu": "recojet_antikt4PFlow_NOSYS_GN2v01_pu",
    "jet_btag_GN2v01_ptau": "recojet_antikt4PFlow_NOSYS_GN2v01_ptau",
}

MC_ALIASES = {
    "mc_event_weights": "mcEventWeights",
    "pileup_weight": "PileupWeight_NOSYS",
    "truth_jet_label_ID": "truthjet_antikt4_HadronConeExclTruthLabelID",
    "truth_jet_eta": "truthjet_antikt4_eta",
    "truth_jet_mass": "truthjet_antikt4_m",
    "truth_jet_phi": "truthjet_antikt4_phi",
    "truth_jet_pt": "truthjet_antikt4_pt",
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


def get_trigger_branch_aliases(trig_set, year=None):
    if year is not None:
        trig_set = f"{trig_set} {year}"
    return {
        f"trig_{trig_short}": f"trigPassed_{trig_long}"
        for trig_long, trig_short, _ in trigger_sets[trig_set]
    }


def get_branch_aliases(is_mc=False, trig_sets=None, sample_metadata=None):
    aliases = {**BASE_ALIASES}
    aliases.update(
        {
            key: value if is_mc else value.replace("antikt4", "antikt4PFlow")
            for key, value in JET_ALIASES.items()
        }
    )
    if is_mc:
        aliases.update(MC_ALIASES)

    if trig_sets:
        year = sample_metadata.get("dataTakingYear", None)
        for trig_set in trig_sets:
            aliases.update(get_trigger_branch_aliases(trig_set, year))

    return aliases
