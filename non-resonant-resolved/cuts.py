import numpy as np
import awkward as ak
import triggers


def select_ge4_central_jets_events(events, pt_cut=None, eta_cut=None):
    jets_pt = events["recojet_antikt4_NOSYS_pt"]
    jets_eta = events["recojet_antikt4_NOSYS_eta"]
    if pt_cut and eta_cut:
        passed_kin_cut_mask = (jets_pt > pt_cut) & (np.abs(jets_eta) < eta_cut)
    elif pt_cut and not eta_cut:
        passed_kin_cut_mask = jets_pt > pt_cut
    elif eta_cut and not pt_cut:
        passed_kin_cut_mask = np.abs(jets_eta) < eta_cut
    else:
        passed_kin_cut_mask = None
    if passed_kin_cut_mask is not None:
        passed_4_jets = ak.num(jets_pt[passed_kin_cut_mask]) > 3
    else:
        passed_4_jets = ak.num(jets_pt) > 3
    valid_events = events[passed_4_jets]
    return valid_events


def select_jets_sorted_by_pt(events):
    indices = ak.argsort(events["recojet_antikt4_NOSYS_pt"], ascending=False)
    sorted_events_jets = events[
        [
            "recojet_antikt4_NOSYS_pt",
            "recojet_antikt4_NOSYS_eta",
            "recojet_antikt4_NOSYS_phi",
            "recojet_antikt4_NOSYS_m",
            "recojet_antikt4_NOSYS_DL1dv01_FixedCutBEff_70",
            "recojet_antikt4_NOSYS_GN120220509_FixedCutBEff_70",
        ],
        indices,
    ]
    return sorted_events_jets


def select_trigger_decisions(events):
    trigs = events[[f"trigPassed_{trig}" for trig in triggers.run3_all]]
    return trigs
