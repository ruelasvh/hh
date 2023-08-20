from .triggers import (
    run3_main_stream as trigs_run3_main,
    run2_reoptimized as trigs_run2_reoptimized,
)

BASE_ALIASES = {
    "event_number": "eventNumber",
    "run_number": "runNumber",
}

JET_ALIASES = {
    "jet_pt": "recojet_antikt4_NOSYS_pt",
    "jet_eta": "recojet_antikt4_NOSYS_eta",
    "jet_phi": "recojet_antikt4_NOSYS_phi",
    "jet_mass": "recojet_antikt4_NOSYS_m",
    "jet_NNJvt": "recojet_antikt4_NOSYS_NNJvt",
    "jet_jvttag": "recojet_antikt4_NOSYS_NNJvtPass",
    "jet_btag_DL1dv01_70": "recojet_antikt4_NOSYS_ftag_select_DL1dv01_FixedCutBEff_70",
    "jet_btag_DL1dv01_77": "recojet_antikt4_NOSYS_ftag_select_DL1dv01_FixedCutBEff_77",
    "jet_btag_DL1dv01_85": "recojet_antikt4_NOSYS_ftag_select_DL1dv01_FixedCutBEff_85",
}

MC_ALIASES = {
    "mc_event_weights": "mcEventWeights",
    "pileup_weight": "PileupWeight_NOSYS",
    "jet_truth_H_parents": "recojet_antikt4_NOSYS_parentHiggsParentsMask",
    # "jet_btag_sf_DL1dv01_70": "recojet_antikt4_NOSYS_ftag_effSF_DL1dv01_FixedCutBEff_70",
    # "jet_btag_sf_DL1dv01_77": "recojet_antikt4_NOSYS_ftag_effSF_DL1dv01_FixedCutBEff_77",
    # "jet_btag_sf_DL1dv01_85": "recojet_antikt4_NOSYS_ftag_effSF_DL1dv01_FixedCutBEff_85",
}


def get_branch_aliases(is_mc=False, run=2):
    aliases = {**BASE_ALIASES}
    aliases.update({
        key: value if is_mc else value.replace("antikt4", "antikt4PFlow")
        for key, value in JET_ALIASES.items()
    })
    if is_mc:
        aliases.update(MC_ALIASES)
    if run == 2:
        aliases.update({
            f"trig_passed_{trig_short}": f"trigPassed_{trig_long}"
            for trig_long, trig_short, _ in trigs_run2_reoptimized
        })
    elif run == 3:
        aliases.update({
            f"trig_passed_{trig_short}": f"trigPassed_{trig_long}"
            for trig_long, trig_short, _ in trigs_run3_main
        })
    return aliases


def get_jet_branch_alias_names(aliases):
    jet_alias_names = list(filter(lambda alias: "jet_" in alias, aliases))
    return jet_alias_names
