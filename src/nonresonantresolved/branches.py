from .triggers import run3_all as triggers_run3_all

base_aliases = {
    "mc_event_weight": "mcEventWeights",
    "pileup_weight": "pileupWeight_NOSYS",
    "jet_pt": "recojet_antikt4_NOSYS_pt",
    "jet_eta": "recojet_antikt4_NOSYS_eta",
    "jet_phi": "recojet_antikt4_NOSYS_phi",
    "jet_m": "recojet_antikt4_NOSYS_m",
    "btag_DL1dv01_70": "recojet_antikt4_NOSYS_DL1dv01_FixedCutBEff_70",
    "btag_GN1_70": "recojet_antikt4_NOSYS_GN120220509_FixedCutBEff_70",
    "reco_H1_m_DL1dv01_70": "resolved_DL1dv01_FixedCutBEff_70_h1_m",
    "reco_H2_m_DL1dv01_70": "resolved_DL1dv01_FixedCutBEff_70_h2_m",
    **{trig: f"trigPassed_{trig}" for trig in triggers_run3_all},
}

signal_aliases = {
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
    aliases = {**base_aliases}
    if signal:
        aliases |= signal_aliases
    return aliases
