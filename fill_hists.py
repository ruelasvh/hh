import numpy as np
import awkward as ak
import vector as p4
from collections import defaultdict
from histograms import EffHistogram, EffHistogramdd
import triggers
import logging


def init_leading_jets_passed_trig_hists():
    leading_jets_passed_trig_hists = defaultdict(lambda: defaultdict(int))
    for ith_jet in np.arange(0, 4):
        # bin_range = [20_000, 1_300_000]
        bin_range = [0, 1_300_000]
        if ith_jet == 2:
            # bin_range = [20_000, 900_000]
            bin_range = [0, 900_000]
        if ith_jet == 3:
            # bin_range = [20_000, 700_000]
            bin_range = [0, 700_000]
        for trig in triggers.run3_all:
            leading_jets_passed_trig_hists[ith_jet][trig] = EffHistogram(
                f"jet{ith_jet}_passed_{trig}",
                bin_range,
                bins=100,
            )
    return leading_jets_passed_trig_hists


def fill_leading_jet_pt_passed_trig_hists(jet_pt, trigs_decisions, output):
    for ith_jet, passed_trig_hists in output.items():
        for trig, hist in passed_trig_hists.items():
            trig_decisions = trigs_decisions[f"trigPassed_{trig}"]
            hist.fill(
                jet_pt[trig_decisions][:, ith_jet],
                jet_pt[:, ith_jet],
            )


def init_mH_passed_trig_hists():
    mH_passed_trig_hists = defaultdict(lambda: defaultdict(int))
    bin_range = [0, 200_000]
    for ith_var in np.arange(0, 2):
        for trig in triggers.run3_all:
            mH_passed_trig_hists[ith_var][trig] = EffHistogram(
                f"mH{ith_var}_passed_{trig}",
                bin_range,
                bins=100,
            )
    return mH_passed_trig_hists


def fill_mH_passed_trig_hists(events, trigs_decisions, output):
    h1_m = events["resolved_DL1dv01_FixedCutBEff_70_h1_m"]
    h2_m = events["resolved_DL1dv01_FixedCutBEff_70_h2_m"]

    logging.debug(
        f"total number events from jets, H1 and H2 in events (all should be equal): {len(h1_m)}, {len(h2_m)}"
    )

    for ith_var, passed_trig_hists in output.items():
        for trig, hist in passed_trig_hists.items():
            trig_decisions = trigs_decisions[f"trigPassed_{trig}"]
            if ith_var == 0:
                hist.fill(
                    h1_m[trig_decisions],
                    h1_m,
                )
            if ith_var == 1:
                hist.fill(
                    h2_m[trig_decisions],
                    h2_m,
                )


def init_mH_plane_passed_trig_hists():
    mH_plane_passed_trig_hists = defaultdict(lambda: defaultdict(int))
    bin_range = (0, 200_000)
    for ith_var in np.arange(0, 1):
        for trig in triggers.run3_all:
            mH_plane_passed_trig_hists[ith_var][trig] = EffHistogramdd(
                f"mH_plane_passed_{trig}",
                bin_range,
                bins=100,
            )
    return mH_plane_passed_trig_hists


def fill_mH_plane_passed_trig_hists(events, trigs_decisions, output):

    h1_m = events["resolved_DL1dv01_FixedCutBEff_70_h1_m"]
    h2_m = events["resolved_DL1dv01_FixedCutBEff_70_h2_m"]

    for ith_var, passed_trig_hists in output.items():
        for trig, hist in passed_trig_hists.items():
            trig_decisions = trigs_decisions[f"trigPassed_{trig}"]
            passed_trig = np.column_stack((h1_m[trig_decisions], h2_m[trig_decisions]))
            total = np.column_stack((h1_m, h2_m))
            hist.fill(total)
