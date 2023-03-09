import awkward as ak
import triggers


def select_jets_sorted_by_pt(events):
    indices = ak.argsort(events["jet_pt"], ascending=False)
    sorted_events_jets = events[
        [
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_m",
            "jet_DL1dv01_FixedCutBEff_70",
            "jet_GN120220509_FixedCutBEff_70",
        ],
        indices,
    ]
    return sorted_events_jets


def select_trigger_decisions(events):
    trigs = events[[trig for trig in triggers.run3_all]]
    return trigs
