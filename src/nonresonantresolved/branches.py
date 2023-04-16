from .triggers import run3_all as triggers_run3_all

BASE_ALIASES = {
    "mc_event_weight": "mcEventWeights",
    "pileup_weight": "pileupWeight_NOSYS",
    "jet_pt": "recojet_antikt4_NOSYS_pt",
    "jet_eta": "recojet_antikt4_NOSYS_eta",
    "jet_phi": "recojet_antikt4_NOSYS_phi",
    "jet_m": "recojet_antikt4_NOSYS_m",
    "jet_btag_DL1dv00_70": "recojet_antikt4_NOSYS_DL1dv00_FixedCutBEff_70",
    "jet_btag_DL1dv00_77": "recojet_antikt4_NOSYS_DL1dv00_FixedCutBEff_77",
    # "reco_H1_m_DL1dv00_70": "resolved_DL1dv00_FixedCutBEff_70_h1_m",
    # "reco_H2_m_DL1dv00_70": "resolved_DL1dv00_FixedCutBEff_70_h2_m",
    # **{trig: f"trigPassed_{trig}" for trig in triggers_run3_all},
}

SIGNAL_ALIASES = {
    "H1_pt": "truth_H1_pt",
    "H1_eta": "truth_H1_eta",
    "H1_phi": "truth_H1_phi",
    "H1_m": "truth_H1_m",
    "H2_pt": "truth_H2_pt",
    "H2_eta": "truth_H2_eta",
    "H2_phi": "truth_H2_phi",
    "H2_m": "truth_H2_m",
    "reco_H1_truth_paired": "resolved_DL1dv01_FixedCutBEff_70_h1_closestTruthBsHaveSameInitialParticle",
    "reco_H2_truth_paired": "resolved_DL1dv01_FixedCutBEff_70_h2_closestTruthBsHaveSameInitialParticle",
    "resolved_truth_mached_jet_pt": "resolved_truthMatched_DL1dv01_FixedCutBEff_70_pt",
    "resolved_truth_mached_jet_eta": "resolved_truthMatched_DL1dv01_FixedCutBEff_70_eta",
    "resolved_truth_mached_jet_phi": "resolved_truthMatched_DL1dv01_FixedCutBEff_70_phi",
    "resolved_truth_mached_jet_m": "resolved_truthMatched_DL1dv01_FixedCutBEff_70_m",
}


def get_branch_aliases(signal=False):
    aliases = {**BASE_ALIASES}
    if signal:
        aliases |= SIGNAL_ALIASES
    return aliases


def get_jet_branch_alias_names():
    alias_names = get_branch_aliases().keys()
    jet_alias_names = list(filter(lambda alias: "jet_" in alias, alias_names))
    return jet_alias_names
