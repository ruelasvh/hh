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


def init_mH_plane_passed_trig_hists(triggersOR=False):
    mH_plane_passed_trig_hists = defaultdict(lambda: defaultdict(int))
    bin_range = (20_000, 200_000)
    for ith_var in np.arange(0, 1):
        for trig in triggers.run3_all:
            mH_plane_passed_trig_hists[ith_var][trig] = EffHistogramdd(
                f"mH_plane_passed_{trig}",
                bin_range,
                bins=30,
            )
        if triggersOR:
            mH_plane_passed_trig_hists[ith_var]["triggersOR"] = EffHistogramdd(
                "mH_plane_passed_triggersOR",
                bin_range,
                bins=30,
            )
    return mH_plane_passed_trig_hists


def fill_mH_plane_passed_trig_hists(events, trigs_decisions, output):
    h1_m = events["resolved_DL1dv01_FixedCutBEff_70_h1_m"]
    h2_m = events["resolved_DL1dv01_FixedCutBEff_70_h2_m"]
    for ith_var, passed_trig_hists in output.items():
        for trig, hist in passed_trig_hists.items():
            trig_decisions = trigs_decisions[f"trigPassed_{trig}"]
            passed = np.column_stack((h1_m[trig_decisions], h2_m[trig_decisions]))
            total = np.column_stack((h1_m, h2_m))
            hist.fill(passed, total)


def fill_mH_plane_passed_exclusive_trig_hists(events, trigs_decisions, output):
    from functools import reduce

    def get_all_trigs_or(skip_trig=None):
        trigs = list(filter(lambda trig: trig != skip_trig, triggers.run3_all))
        # return the or of all trigs expect skip_trig
        return reduce(
            lambda acc, it: acc | trigs_decisions[f"trigPassed_{it}"],
            trigs,
            trigs_decisions[f"trigPassed_{trigs[0]}"],
        )

    def get_exclusive_trig(trig):
        trig_decicions_this = trigs_decisions[f"trigPassed_{trig}"]
        trig_decicions_rest = get_all_trigs_or(trig)
        # return true if this trig passes but not passing the rest
        return trig_decicions_this & ~trig_decicions_rest

    h1_m = events["resolved_DL1dv01_FixedCutBEff_70_h1_m"]
    h2_m = events["resolved_DL1dv01_FixedCutBEff_70_h2_m"]
    for ith_var, passed_trig_hists in output.items():
        for trig, hist in passed_trig_hists.items():
            if "triggersOR" in trig:
                trig_decisions = get_all_trigs_or()
                passed = np.column_stack((h1_m[trig_decisions], h2_m[trig_decisions]))
                total = np.column_stack((h1_m, h2_m))
                hist.fill(passed, total)
            else:
                trig_decisions = get_exclusive_trig(trig)
                passed = np.column_stack((h1_m[trig_decisions], h2_m[trig_decisions]))
                total = np.column_stack((h1_m, h2_m))
                hist.fill(passed, total)


def init_mH_passed_pairing_hists():
    mH_passed_pairing_hists = defaultdict(lambda: defaultdict(int))
    bin_range = [0, 200_000]
    for ith_var in np.arange(0, 2):
        for ith_pair in np.arange(0, 1):
            mH_passed_pairing_hists[ith_var][ith_pair] = EffHistogram(
                f"mH{ith_var}_passed_h{ith_var}_pairing",
                bin_range,
                bins=100,
            )
    return mH_passed_pairing_hists


def fill_mH_passed_pairing_hists(events, output):
    h1_m = events["resolved_DL1dv01_FixedCutBEff_70_h1_m"]
    h2_m = events["resolved_DL1dv01_FixedCutBEff_70_h2_m"]
    h1_pairing_decisions = (
        events[
            "resolved_DL1dv01_FixedCutBEff_70_h1_closestTruthBsHaveSameInitialParticle"
        ]
        == 1
    )
    h2_pairing_decisions = (
        events[
            "resolved_DL1dv01_FixedCutBEff_70_h2_closestTruthBsHaveSameInitialParticle"
        ]
        == 1
    )
    for ith_var, passed_trig_hists in output.items():
        for ith_pair, hist in passed_trig_hists.items():
            if ith_var == 0:
                hist.fill(
                    h1_m[h1_pairing_decisions],
                    h1_m,
                )
            if ith_var == 1:
                hist.fill(
                    h2_m[h2_pairing_decisions],  # mass of the matched reconstructed
                    h2_m,  # truth reconstructured mass
                    # would take out resolution effects
                )
