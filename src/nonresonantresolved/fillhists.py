import numpy as np
import vector as p4
import awkward as ak
from .triggers import run3_main_stream as triggers_run3_all
from shared.utils import (
    logger,
    find_hist,
    find_hists,
    find_hists_by_name,
    kin_labels,
)
from .selection import (
    select_n_jets_events,
)


def fill_hists(
    events: ak.Record,
    hists: list,
    is_mc: bool = True,
) -> list:  # hists:
    """Fill histograms for analysis regions"""
    fill_jet_kin_histograms(events, hists)
    fill_top_veto_histograms(
        events,
        hists=find_hists(hists, lambda h: "top_veto" in h.name),
    )
    fill_hh_deltaeta_histograms(
        events,
        hists=find_hists(hists, lambda h: "hh_deltaeta" in h.name),
    )
    fill_hh_mass_discrim_histograms(
        events,
        hists=find_hists(hists, lambda h: "hh_mass_discrim" in h.name),
    )
    leading_h_jet_idx = events.leading_h_jet_idx
    subleading_h_jet_idx = events.subleading_h_jet_idx
    jet_p4 = p4.zip(
        {var: events[f"jet_{var}"] for var in kin_labels.keys()},
    )
    h1 = (
        jet_p4[leading_h_jet_idx[:, 0, np.newaxis]]
        + jet_p4[leading_h_jet_idx[:, 1, np.newaxis]]
    )
    h2 = (
        jet_p4[subleading_h_jet_idx[:, 0, np.newaxis]]
        + jet_p4[subleading_h_jet_idx[:, 1, np.newaxis]]
    )
    fill_reco_mH_histograms(
        mh1=np.squeeze(h1.mass),
        mh2=np.squeeze(h2.mass),
        weights=events.event_weight,
        hists=find_hists_by_name(hists, "mH[12]_baseline"),
    )
    fill_reco_mH_2d_histograms(
        mh1=np.squeeze(h1.mass),
        mh2=np.squeeze(h2.mass),
        weights=events.event_weight,
        hist=find_hist(hists, lambda h: "mH_plane_baseline" in h.name),
    )
    signal_event = events.signal_event
    signal_mh1 = np.squeeze(h1.mass)[signal_event]
    signal_mh2 = np.squeeze(h2.mass)[signal_event]
    signal_event_weight = events.event_weight[signal_event]
    fill_reco_mH_histograms(
        mh1=signal_mh1,
        mh2=signal_mh2,
        weights=signal_event_weight,
        hists=find_hists_by_name(hists, "mH[12]_baseline_signal_region"),
    )
    fill_reco_mH_2d_histograms(
        mh1=signal_mh1,
        mh2=signal_mh2,
        weights=signal_event_weight,
        hist=find_hist(hists, lambda h: "mH_plane_baseline_signal_region" in h.name),
    )
    control_event = events.control_event
    control_mh1 = np.squeeze(h1.mass)[control_event]
    control_mh2 = np.squeeze(h2.mass)[control_event]
    control_event_weight = events.event_weight[control_event]
    fill_reco_mH_histograms(
        mh1=control_mh1,
        mh2=control_mh2,
        weights=control_event_weight,
        hists=find_hists_by_name(hists, "mH[12]_baseline_control_region"),
    )
    fill_reco_mH_2d_histograms(
        mh1=control_mh1,
        mh2=control_mh2,
        weights=control_event_weight,
        hist=find_hist(hists, lambda h: "mH_plane_baseline_control_region" in h.name),
    )

    return hists


def fill_jet_kin_histograms(events, hists: list) -> None:
    """Fill jet kinematics histograms"""

    for jet_var in kin_labels.keys():
        hist = find_hist(hists, lambda h: f"jet_{jet_var}" in h.name)
        logger.debug(hist.name)
        jets = events[f"jet_{jet_var}"]
        event_weight = events.event_weight[:, np.newaxis]
        event_weight, _ = ak.broadcast_arrays(event_weight, jets)
        hist.fill(
            np.array(ak.flatten(jets)), weights=np.array(ak.flatten(event_weight))
        )


def fill_leading_jets_histograms(events, hists: list) -> None:
    """Fill leading jets histograms"""

    jet_pt = events["jet_pt"]

    leading_jets_hists = find_hists_by_name(hists, "leading_jet_[1234]_pt")
    for ith_jet in [1, 2, 3, 4]:
        hist = find_hist(
            leading_jets_hists, lambda h: f"leading_jet_{ith_jet}_pt" in h.name
        )
        logger.debug(hist.name)
        hist.fill(np.array(jet_pt[:, ith_jet - 1]))
        # for trig in triggers_run3_all:
        #     hist = find_hist(
        #         leading_jets_hists,
        #         lambda h: f"leading_jet_{ith_jet}_pt_trigPassed_{trig}" in h.name,
        #     )
        #     logger.debug(hist.name)
        #     trig_decisions = events[trig]
        #     hist.fill(np.array(jet_pt[trig_decisions][:, ith_jet - 1]))


def fill_truth_matched_mjj_histograms(events, hists: list) -> None:
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
        njets_cut=4,
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

    mjj_hists = find_hists_by_name(hists, "mjj[12]")
    for ith_jj in [1, 2]:
        hist = find_hist(mjj_hists, lambda h: f"mjj{ith_jj}" in h.name)
        logger.debug(hist.name)
        mjj_p4 = (
            leading_jets_p4[:, 0 if ith_jj == 1 else 2]
            + leading_jets_p4[:, 1 if ith_jj == 1 else 3]
        )
        hist.fill(mjj_p4.mass)


def fill_truth_matched_mjj_passed_pairing_histograms(events, hists: list) -> None:
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

    mjj_hists = find_hists_by_name(hists, "mjj[12]_pairingPassedTruth")
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


def fill_reco_mH_histograms(mh1, mh2, weights, hists: list) -> None:
    """Fill reconstructed H invariant mass 1D histograms"""

    if len(hists) != 2:
        raise ValueError("Expected 2 histograms, got", len(hists))

    if ak.count(mh1) != 0 and ak.count(mh2) != 0:
        for hist, mH in zip(hists, [mh1, mh2]):
            logger.debug(hist.name)
            hist.fill(ak.to_numpy(mH), weights=ak.to_numpy(weights))


def fill_reco_mH_2d_histograms(mh1, mh2, weights, hist) -> None:
    """Fill reconstructed H invariant mass 2D histograms"""

    if ak.count(mh1) != 0 and ak.count(mh2) != 0:
        mHH = np.column_stack((mh1, mh2))
        hist.fill(ak.to_numpy(mHH), weights=ak.to_numpy(weights))


def fill_reco_mH_truth_pairing_histograms(events, hists: list) -> None:
    """Fill reconstructed H have same bb children that originated from the truth H histograms"""

    H_pairing_hists = find_hists_by_name(hists, "mH[12]_pairingPassedTruth")
    for ith_H in [1, 2]:
        hist = find_hist(
            H_pairing_hists, lambda h: f"mH{ith_H}_pairingPassedTruth" in h.name
        )
        logger.debug(hist.name)
        mH = events[f"reco_H{ith_H}_m_DL1dv01_70"]
        pairing_decisions = events[f"reco_H{ith_H}_truth_paired"] == 1
        hist.fill(mH[pairing_decisions])


def fill_hh_deltaeta_histograms(events, hists: list) -> None:
    """Fill HH deltaeta histograms"""
    hh_deltar = events.hh_deltaeta_discriminant
    if ak.count(hh_deltar) != 0:
        valid_events = ~ak.is_none(hh_deltar)
        hist = find_hist(hists, lambda h: "hh_deltaeta_baseline" in h.name)
        logger.debug(hist.name)
        hist.fill(
            np.array(hh_deltar[valid_events]),
            weights=np.array(events.event_weight[valid_events]),
        )


def fill_hh_mass_discrim_histograms(events, hists: list) -> None:
    """Fill HH mass discriminant histograms"""
    hh_mass_discrim = events.hh_mass_discriminant_signal
    if ak.count(hh_mass_discrim) != 0:
        valid_events = ~ak.is_none(hh_mass_discrim)
        hist = find_hist(hists, lambda h: "hh_mass_discrim_baseline" in h.name)
        logger.debug(hist.name)
        hist.fill(
            np.array(hh_mass_discrim[valid_events]),
            weights=np.array(events.event_weight[valid_events]),
        )


def fill_top_veto_histograms(events, hists: list) -> None:
    """Fill top veto histograms"""

    X_Wt_discrim = events.X_Wt_discriminant_min
    if ak.count(X_Wt_discrim) != 0:
        valid_events = ~ak.is_none(X_Wt_discrim)
        top_veto_discrim_hist = find_hist(
            hists, lambda h: "top_veto_baseline" in h.name
        )
        top_veto_discrim_hist.fill(
            np.array(X_Wt_discrim[valid_events]),
            weights=np.array(events.event_weight[valid_events]),
        )

        top_veto_nbtags_hist = find_hist(hists, lambda h: "top_veto_n_btags" in h.name)
        top_veto_nbtags_hist.fill(np.array(events.btag_num[valid_events]))
