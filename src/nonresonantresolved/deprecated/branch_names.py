from triggers import run3_all as triggers_run3_all

filter_name = [
    "/recojet_antikt4_NOSYS_(pt|eta|phi|m)/",
    "/recojet_antikt4_NOSYS_(DL1dv01|GN120220509)_FixedCutBEff_70/",
    "/truth_H1_(pt|eta|phi|m)/",
    "/truth_H2_(pt|eta|phi|m)/",
    "/resolved_DL1dv01_FixedCutBEff_70_h(1|2)_m/",
    "/resolved_DL1dv01_FixedCutBEff_70_h(1|2)_closestTruthBsHaveSameInitialParticle/",
    *[f"trigPassed_{trig}" for trig in triggers_run3_all],
]
