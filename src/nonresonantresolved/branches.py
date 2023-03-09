from .triggers import run3_all as triggers_run3_all

aliases = {
    "jet_pt": "recojet_antikt4_NOSYS_pt",
    "jet_eta": "recojet_antikt4_NOSYS_eta",
    "jet_phi": "recojet_antikt4_NOSYS_phi",
    "jet_m": "recojet_antikt4_NOSYS_m",
    "btag_DL1dv01_70": "recojet_antikt4_NOSYS_DL1dv01_FixedCutBEff_70",
    "btag_GN1_70": "recojet_antikt4_NOSYS_GN120220509_FixedCutBEff_70",
    "H1_pt": "truth_H1_pt",
    "H1_eta": "truth_H1_eta",
    "H1_phi": "truth_H1_phi",
    "H1_m": "truth_H1_m",
    "H2_pt": "truth_H2_pt",
    "H2_eta": "truth_H2_eta",
    "H2_phi": "truth_H2_phi",
    "H2_m": "truth_H2_m",
    "reco_H1_m_DL1dv01_70": "resolved_DL1dv01_FixedCutBEff_70_h1_m",
    "reco_H2_m_DL1dv01_70": "resolved_DL1dv01_FixedCutBEff_70_h2_m",
    # "reco_H1_truth_paired": "resolved_DL1dv01_FixedCutBEff_70_h1_closestTruthBsHaveSameInitialParticle",
    # "reco_H2_truth_paired": "resolved_DL1dv01_FixedCutBEff_70_h2_closestTruthBsHaveSameInitialParticle",
    **{trig: f"trigPassed_{trig}" for trig in triggers_run3_all},
}

names = aliases.keys()

filter_name = [
    "/recojet_antikt4_NOSYS_(pt|eta|phi|m)/",
    "/recojet_antikt4_NOSYS_(DL1dv01|GN120220509)_FixedCutBEff_70/",
    "/truth_H1_(pt|eta|phi|m)/",
    "/truth_H2_(pt|eta|phi|m)/",
    "/resolved_DL1dv01_FixedCutBEff_70_h(1|2)_m/",
    "/resolved_DL1dv01_FixedCutBEff_70_h(1|2)_closestTruthBsHaveSameInitialParticle/",
    *[f"trigPassed_{trig}" for trig in triggers_run3_all],
]
