import pandas as pd
import numpy as np
import awkward as ak
from shared.utils import find_hist


def fillhists(events: pd.DataFrame, hists):
    """Fill histograms"""
    # fill_jet_kin_histograms(
    #     events=events,
    #     hists=hists,
    # )
    fill_top_veto_histograms(
        events=events,
        hists=hists,
    )


# def fill_jet_kin_histograms(events, hists: list, lumi_weight: float) -> None:
#     """Fill jet kinematics histograms"""

#     for jet_var in kin_labels.keys():
#         hist = find_hist(hists, lambda h: f"jet_{jet_var}" in h.name)
#         logger.debug(hist.name)
#         jets = events[f"jet_{jet_var}"]
#         mc_evt_weight_nom = events.mc_event_weight[:, np.newaxis]
#         mc_evt_weight_nom, _ = ak.broadcast_arrays(mc_evt_weight_nom, jets)
#         pileup_weight = events.pileup_weight[:, np.newaxis]
#         pileup_weight, _ = ak.broadcast_arrays(pileup_weight, jets)
#         weights = mc_evt_weight_nom * pileup_weight * lumi_weight
#         hist.fill(np.array(ak.flatten(jets)), weights=np.array(ak.flatten(weights)))


def fill_top_veto_histograms(events, hists: list) -> None:
    """Fill top veto histograms"""

    top_veto_discrim_hist = find_hist(hists, lambda h: "top_veto_baseline" in h.name)
    top_veto_discrim_hist.fill(events.X_wt_tag.values)

    # top_veto_nbtags_hist = find_hist(hists, lambda h: "top_veto_n_btags" in h.name)
    # top_veto_nbtags_hist.fill(np.array(events.btag_num))
