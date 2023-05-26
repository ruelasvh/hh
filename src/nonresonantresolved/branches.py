from src.nonresonantresolved.triggers import run3_all as triggers_run3_all
from src.nonresonantresolved.utils import format_btagger_model_name

BASE_ALIASES = {
    "mc_event_weight": "mcEventWeights",
    "pileup_weight": "PileupWeight_NOSYS",
    **{trig: f"trigPassed_{trig}" for trig in triggers_run3_all},
}

JET_ALIASES = {
    "jet_pt": "recojet_antikt4_NOSYS_pt",
    "jet_eta": "recojet_antikt4_NOSYS_eta",
    "jet_phi": "recojet_antikt4_NOSYS_phi",
    "jet_m": "recojet_antikt4_NOSYS_m",
    "jet_btag_DL1dv01_70": "recojet_antikt4_NOSYS_ftag_select_DL1dv01_FixedCutBEff_70",
    "jet_btag_DL1dv01_77": "recojet_antikt4_NOSYS_ftag_select_DL1dv01_FixedCutBEff_77",
}

SIGNAL_ALIASES = {"jet_truth_H_parents": "recojet_antikt4_NOSYS_parentHiggsParentsMask"}


def get_branch_aliases(signal=False):
    aliases = {**BASE_ALIASES}
    aliases |= JET_ALIASES
    if signal:
        aliases |= SIGNAL_ALIASES
    return aliases


def get_jet_branch_alias_names(aliases):
    jet_alias_names = list(filter(lambda alias: "jet_" in alias, aliases))
    return jet_alias_names


def add_default_branches_from_config(branch_aliases, config):
    btagging = config["btagging"]
    btagger = format_btagger_model_name(btagging["model"], btagging["efficiency"])
    branch_names = list(branch_aliases.keys())
    btag_branch = [branch for branch in branch_names if btagger in branch][0]
    branch_aliases["jet_btag_default"] = branch_aliases[btag_branch]
    return branch_aliases
