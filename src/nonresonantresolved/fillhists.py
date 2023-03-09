import re
import logging
import numpy as np
from .triggers import run3_all as triggers_run3_all
from .utils import find_hist, find_all_hists, get_all_trigs_or


def fill_hists(events, hists, args, uncut_events=None):
    """Fill histograms with data"""

    logging.info("Filling hitograms with batch")
    fill_leading_jets_histograms(events, hists)
    fill_reco_mH_histograms(events, hists)
    fill_reco_mH_2d_histograms(events, hists)
    if args.signal:
        fill_reco_mH_truth_pairing_histograms(events, hists)


def fill_leading_jets_histograms(events: dict, hists: list) -> None:
    """Fill leading jets histograms"""

    leading_jets_hists = find_all_hists(hists, "leading_jet_[1234]_pt")
    jet_pt = events["jet_pt"]
    for ith_jet in [1, 2, 3, 4]:
        hist = find_hist(
            leading_jets_hists, lambda h: f"leading_jet_{ith_jet}_pt" in h.name
        )
        logging.debug(hist.name)
        hist.fill(jet_pt[:, ith_jet - 1])
        for trig in triggers_run3_all:
            hist = find_hist(
                leading_jets_hists,
                lambda h: f"leading_jet_{ith_jet}_pt_trigPassed_{trig}" in h.name,
            )
            logging.debug(hist.name)
            trig_decisions = events[trig]
            hist.fill(jet_pt[trig_decisions][:, ith_jet - 1])


def fill_reco_mH_histograms(events: dict, hists: list) -> None:
    """Fill reconstructed H invariant mass 1D histograms"""

    mH_hists = find_all_hists(hists, "mH[12]")

    for ith_H in [1, 2]:
        hist = find_hist(mH_hists, lambda h: f"mH{ith_H}" in h.name)
        logging.debug(hist.name)
        branch_name = f"reco_H{ith_H}_m_DL1dv01_70"
        hist.fill(events[branch_name])
        for trig in triggers_run3_all:
            hist = find_hist(
                mH_hists, lambda h: f"mH{ith_H}_trigPassed_{trig}" in h.name
            )
            logging.debug(hist.name)
            trig_decisions = events[trig]
            hist.fill(events[branch_name][trig_decisions])


def fill_reco_mH_2d_histograms(events: dict, hists: list) -> None:
    """Fill reconstructed H invariant mass 2D histograms"""

    mH_plane_hists = find_all_hists(hists, "mH_plane")

    mH1 = events["reco_H1_m_DL1dv01_70"]
    mH2 = events["reco_H2_m_DL1dv01_70"]

    hist = find_hist(mH_plane_hists, lambda h: "mH_plane" in h.name)
    logging.debug(hist.name)
    hist.fill(np.column_stack((mH1, mH2)))
    hist = find_hist(
        mH_plane_hists, lambda h: "mH_plane_trigPassed_allTriggersOR" in h.name
    )
    trigs_OR_decisions = get_all_trigs_or(events, triggers_run3_all)
    hist.fill(np.column_stack((mH1[trigs_OR_decisions], mH2[trigs_OR_decisions])))
    for trig in triggers_run3_all:
        hist = find_hist(
            mH_plane_hists, lambda h: f"mH_plane_trigPassed_{trig}" in h.name
        )
        logging.debug(hist.name)
        trig_decisions = events[trig]
        hist.fill(np.column_stack((mH1[trig_decisions], mH2[trig_decisions])))


def fill_reco_mH_truth_pairing_histograms(events: dict, hists: list) -> None:
    """Fill reconstructed H have same bb children that originated from the truth H histograms"""

    H_pairing_hists = find_all_hists(hists, "mH[12]_pairingPassedTruth")
    for ith_H in [1, 2]:
        hist = find_hist(
            H_pairing_hists, lambda h: f"mH{ith_H}_pairingPassedTruth" in h.name
        )
        logging.debug(hist.name)
        mH = events[f"reco_H{ith_H}_m_DL1dv01_70"]
        pairing_decisions = events[f"reco_H{ith_H}_truth_paired"] == 1
        hist.fill(mH[pairing_decisions])
