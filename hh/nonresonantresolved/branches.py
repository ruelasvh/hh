from .triggers import (
    trig_sets,
)

CAMPAIGNS = {
    2016: ["r13167"],
    2017: ["r13144"],
    2018: ["r13145"],
    2022: ["r14622", "r14932"],
    2023: ["r15224"],
}

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
    "jet_btag_GN2v01_65": "recojet_antikt4PFlow_NOSYS_ftag_select_GN2v01_FixedCutBEff_65",
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
    "h1_truth_ID": "truth_H1_pdgId",
    "h1_truth_pt": "truth_H1_pt",
    "h1_truth_eta": "truth_H1_eta",
    "h1_truth_phi": "truth_H1_phi",
    "h1_truth_m": "truth_H1_m",
    "h2_truth_ID": "truth_H2_pdgId",
    "h2_truth_pt": "truth_H2_pt",
    "h2_truth_eta": "truth_H2_eta",
    "h2_truth_phi": "truth_H2_phi",
    "h2_truth_m": "truth_H2_m",
    "hh_truth_eta": "truth_HH_eta",
    "hh_truth_phi": "truth_HH_phi",
    "hh_truth_m": "truth_HH_m",
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
