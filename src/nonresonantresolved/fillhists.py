import numpy as np
import vector as p4
import awkward as ak
from .triggers import run3_all as triggers_run3_all
from .utils import find_hist, find_all_hists, get_all_trigs_or, kin_vars
from .selection import select_n_jets_events
from shared.utils import logger


def fill_hists(events, hists, luminosity_weight, args):
    """Fill histograms with data"""

    fill_jet_kin_histograms(events, hists, luminosity_weight)
    fill_leading_jets_histograms(events, hists)
    fill_reco_mH_histograms(events, hists)
    fill_reco_mH_2d_histograms(events, hists)
    if args.signal:
        fill_reco_mH_truth_pairing_histograms(events, hists)
        fill_truth_matched_mjj_histograms(events, hists)
        fill_truth_matched_mjj_passed_pairing_histograms(events, hists)


def fill_jet_kin_histograms(events: dict, hists: list, lumi_weight: float) -> None:
    """Fill jet kinematics histograms"""

    for jet_var in kin_vars.keys():
        hist = find_hist(hists, lambda h: f"jet_{jet_var}" in h.name)
        logger.debug(hist.name)
        jets = events[f"jet_{jet_var}"]
        mc_evt_weight_nom = events["mc_event_weight"][:, 0]
        mc_evt_weight_nom, _ = ak.broadcast_arrays(mc_evt_weight_nom, jets)
        pileup_weight = events["pileup_weight"]
        pileup_weight, _ = ak.broadcast_arrays(pileup_weight, jets)
        weights = mc_evt_weight_nom * pileup_weight * lumi_weight
        hist.fill(np.array(ak.flatten(jets)), np.array(ak.flatten(weights)))


def fill_leading_jets_histograms(events: dict, hists: list) -> None:
    """Fill leading jets histograms"""

    jet_vars = ["jet_pt", "jet_eta", "jet_phi", "jet_m"]
    leading_jets_events = select_n_jets_events(
        events,
        pt_cut=20_000,
        eta_cut=2.5,
        njets=4,
        jet_vars=jet_vars,
    )
    pt_var, *_ = jet_vars
    jet_pt = leading_jets_events[pt_var]

    leading_jets_hists = find_all_hists(hists, "leading_jet_[1234]_pt")
    for ith_jet in [1, 2, 3, 4]:
        hist = find_hist(
            leading_jets_hists, lambda h: f"leading_jet_{ith_jet}_pt" in h.name
        )
        logger.debug(hist.name)
        hist.fill(np.array(jet_pt[:, ith_jet - 1]))
        for trig in triggers_run3_all:
            hist = find_hist(
                leading_jets_hists,
                lambda h: f"leading_jet_{ith_jet}_pt_trigPassed_{trig}" in h.name,
            )
            logger.debug(hist.name)
            trig_decisions = leading_jets_events[trig]
            hist.fill(np.array(jet_pt[trig_decisions][:, ith_jet - 1]))


def fill_truth_matched_mjj_histograms(events: dict, hists: list) -> None:
    """Fill reconstructed mjj invariant mass 1D histograms"""

    jet_vars = [
        "resolved_truth_mached_jet_pt",
        "resolved_truth_mached_jet_eta",
        "resolved_truth_mached_jet_phi",
        "resolved_truth_mached_jet_m",
    ]
    pt_var, eta_var, phi_var, mass_var = jet_vars
    leading_jets_events = select_n_jets_events(
        events,
        pt_cut=20_000,
        eta_cut=2.5,
        njets=4,
        jet_vars=jet_vars,
    )
    leading_jets_p4 = p4.zip(
        {
            "pt": leading_jets_events[pt_var],
            "eta": leading_jets_events[eta_var],
            "phi": leading_jets_events[phi_var],
            "mass": leading_jets_events[mass_var],
        }
    )

    mjj_hists = find_all_hists(hists, "mjj[12]")
    for ith_jj in [1, 2]:
        hist = find_hist(mjj_hists, lambda h: f"mjj{ith_jj}" in h.name)
        logger.debug(hist.name)
        mjj_p4 = (
            leading_jets_p4[:, 0 if ith_jj == 1 else 2]
            + leading_jets_p4[:, 1 if ith_jj == 1 else 3]
        )
        hist.fill(mjj_p4.mass)


def fill_truth_matched_mjj_passed_pairing_histograms(events: dict, hists: list) -> None:
    """Fill reconstructed mjj invariant mass 1D histograms"""

    jet_vars = [
        "resolved_truth_mached_jet_pt",
        "resolved_truth_mached_jet_eta",
        "resolved_truth_mached_jet_phi",
        "resolved_truth_mached_jet_m",
    ]
    pt_var, eta_var, phi_var, mass_var = jet_vars
    jets_p4 = p4.zip(
        {
            "pt": events[pt_var],
            "eta": events[eta_var],
            "phi": events[phi_var],
            "mass": events[mass_var],
        }
    )

    mjj_hists = find_all_hists(hists, "mjj[12]_pairingPassedTruth")
    for ith_jj in [1, 2]:
        hist = find_hist(mjj_hists, lambda h: f"mjj{ith_jj}" in h.name)
        logger.debug(hist.name)
        pairing_decisions = events[f"reco_H{ith_jj}_truth_paired"] == 1
        logger.debug("pairing decisions", pairing_decisions)
        truth_paired_jets_p4 = jets_p4[pairing_decisions]
        mjj_p4 = (
            truth_paired_jets_p4[:, 0 if ith_jj == 1 else 2]
            + truth_paired_jets_p4[:, 1 if ith_jj == 1 else 3]
        )
        hist.fill(mjj_p4.mass)


def fill_reco_mH_histograms(events: dict, hists: list) -> None:
    """Fill reconstructed H invariant mass 1D histograms"""

    mH_hists = find_all_hists(hists, "mH[12]")

    for ith_H in [1, 2]:
        hist = find_hist(mH_hists, lambda h: f"mH{ith_H}" in h.name)
        logger.debug(hist.name)
        branch_name = f"reco_H{ith_H}_m_DL1dv01_70"
        hist.fill(np.array(events[branch_name]))
        for trig in triggers_run3_all:
            hist = find_hist(
                mH_hists, lambda h: f"mH{ith_H}_trigPassed_{trig}" in h.name
            )
            logger.debug(hist.name)
            trig_decisions = events[trig]
            hist.fill(np.array(events[branch_name][trig_decisions]))


def fill_reco_mH_2d_histograms(events: dict, hists: list) -> None:
    """Fill reconstructed H invariant mass 2D histograms"""

    mH_plane_hists = find_all_hists(hists, "mH_plane")

    mH1 = events["reco_H1_m_DL1dv01_70"]
    mH2 = events["reco_H2_m_DL1dv01_70"]

    hist = find_hist(mH_plane_hists, lambda h: "mH_plane" in h.name)
    logger.debug(hist.name)
    hist.fill(np.column_stack((mH1, mH2)))
    hist = find_hist(
        mH_plane_hists, lambda h: "mH_plane_trigPassed_allTriggersOR" in h.name
    )
    trigs_OR_decisions = get_all_trigs_or(events, triggers_run3_all)
    hist.fill(
        np.column_stack(
            (np.array(mH1[trigs_OR_decisions]), np.array(mH2[trigs_OR_decisions]))
        )
    )
    for trig in triggers_run3_all:
        hist = find_hist(
            mH_plane_hists, lambda h: f"mH_plane_trigPassed_{trig}" in h.name
        )
        logger.debug(hist.name)
        trig_decisions = events[trig]
        hist.fill(
            np.column_stack(
                (np.array(mH1[trig_decisions]), np.array(mH2[trig_decisions]))
            )
        )


def fill_reco_mH_truth_pairing_histograms(events: dict, hists: list) -> None:
    """Fill reconstructed H have same bb children that originated from the truth H histograms"""

    H_pairing_hists = find_all_hists(hists, "mH[12]_pairingPassedTruth")
    for ith_H in [1, 2]:
        hist = find_hist(
            H_pairing_hists, lambda h: f"mH{ith_H}_pairingPassedTruth" in h.name
        )
        logger.debug(hist.name)
        mH = events[f"reco_H{ith_H}_m_DL1dv01_70"]
        pairing_decisions = events[f"reco_H{ith_H}_truth_paired"] == 1
        hist.fill(mH[pairing_decisions])
