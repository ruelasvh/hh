import numpy as np
import vector as p4
import awkward as ak
from .triggers import run3_all as triggers_run3_all
from .utils import find_hist, find_all_hists, get_all_trigs_or, kin_labels
from .selection import (
    select_n_jets_events,
    select_n_bjets,
    hh_reconstruct_mindeltar,
    select_X_Wt_events,
    select_hh_events,
)
from shared.utils import logger
from src.nonresonantresolved.branches import (
    get_jet_branch_alias_names,
)


def fill_hists(events, hists, luminosity_weight, args):
    """Fill histograms with data"""

    logger.info("Filling histograms")
    logger.info("Events: %s", len(events))
    baseline_events = select_n_jets_events(
        events, jet_vars=get_jet_branch_alias_names(), pt_cut=20_000
    )
    logger.info(
        "Events with >= 4 central jets and kinematic requirements: %s",
        len(baseline_events),
    )
    fill_jet_kin_histograms(baseline_events, hists, luminosity_weight)
    fill_leading_jets_histograms(baseline_events, hists)
    leading_bjets, remaining_jets = select_n_bjets(
        baseline_events,
        jet_vars=get_jet_branch_alias_names(),
        btag_cut="jet_btag_DL1dv00_70",
    )
    logger.info("Events with >= 4 b-tagged central jets: %s", len(leading_bjets))
    # logger.info("Events with >= 6 central or forward jets", len(events_with_central_or_forward_jets))
    h1_events, h2_events = hh_reconstruct_mindeltar(leading_bjets)
    h1_events_with_deltaeta_cut, h2_events_with_deltaeta_cut = select_hh_events(
        h1_events, h2_events, deltaeta_cut=1.5
    )
    logger.info(
        "Events with |deltaEta_HH| < threshold: %s", len(h1_events_with_deltaeta_cut)
    )
    fill_hh_deltaeta_histograms(
        h1_events, h2_events, hists=find_all_hists(hists, "hh_deltaeta_baseline")
    )
    top_vetod_events, _, top_veto_discrim = select_X_Wt_events(
        (leading_bjets, remaining_jets)
    )
    fill_top_veto_histograms(
        top_veto_discrim, hists=find_all_hists(hists, "top_veto_baseline")
    )
    logger.info(
        "Events with top-veto discriminant > threshold: %s", len(top_vetod_events)
    )
    # hh_events_with_mass_discrim_cut = select_hh_events(h1_events_with_deltaeta_cut, h2_events_with_deltaeta_cut, mass_discriminant_cut=1.6)
    # print("Events with mass discriminant < threshold: %s", len(hh_events_with_mass_discrim_cut))

    fill_reco_mH_histograms(
        mh1=h1_events_with_deltaeta_cut.m,
        mh2=h2_events_with_deltaeta_cut.m,
        hists=find_all_hists(hists, "mH[12]_baseline"),
    )
    fill_reco_mH_2d_histograms(
        mh1=h1_events.m,
        mh2=h2_events.m,
        hist=find_hist(hists, lambda h: "mH_plane_baseline" in h.name),
    )
    # if args.signal:
    #     fill_reco_mH_truth_pairing_histograms(events, hists)
    #     fill_truth_matched_mjj_histograms(events, hists)
    #     fill_truth_matched_mjj_passed_pairing_histograms(events, hists)


def fill_jet_kin_histograms(events: dict, hists: list, lumi_weight: float) -> None:
    """Fill jet kinematics histograms"""

    for jet_var in kin_labels.keys():
        hist = find_hist(hists, lambda h: f"jet_{jet_var}" in h.name)
        logger.debug(hist.name)
        jets = events[f"jet_{jet_var}"]
        mc_evt_weight_nom = ak.ravel(events["mc_event_weight"][:, 0])
        mc_evt_weight_nom, _ = ak.broadcast_arrays(mc_evt_weight_nom, jets)
        pileup_weight = ak.ravel(events["pileup_weight"])
        pileup_weight, _ = ak.broadcast_arrays(pileup_weight, jets)
        weights = mc_evt_weight_nom * pileup_weight * lumi_weight
        hist.fill(np.array(ak.flatten(jets)), np.array(ak.flatten(weights)))


def fill_leading_jets_histograms(events: dict, hists: list) -> None:
    """Fill leading jets histograms"""

    jet_pt = events["jet_pt"]

    leading_jets_hists = find_all_hists(hists, "leading_jet_[1234]_pt")
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


def fill_reco_mH_histograms(mh1, mh2, hists: list) -> None:
    """Fill reconstructed H invariant mass 1D histograms"""

    if len(hists) != 2:
        raise ValueError("Expected 2 histograms, got", len(hists))

    for hist, mH in zip(hists, [mh1, mh2]):
        logger.debug(hist.name)
        hist.fill(np.array(mH))


# def fill_reco_mH_histograms(events: dict, hists: list) -> None:
#     """Fill reconstructed H invariant mass 1D histograms"""

#     mH_hists = find_all_hists(hists, "mH[12]")
#     print([mH_hists[i].name for i in range(len(mH_hists))])
#     for ith_H in [1, 2]:
#         hist = find_hist(mH_hists, lambda h: f"mH{ith_H}" in h.name)
#         logger.debug(hist.name)
#         branch_name = f"reco_H{ith_H}_m_DL1dv01_70"
#         hist.fill(np.array(events[branch_name]))
#         for trig in triggers_run3_all:
#             hist = find_hist(
#                 mH_hists, lambda h: f"mH{ith_H}_trigPassed_{trig}" in h.name
#             )
#             logger.debug(hist.name)
#             trig_decisions = events[trig]
#             hist.fill(np.array(events[branch_name][trig_decisions]))


def fill_reco_mH_2d_histograms(mh1, mh2, hist: list) -> None:
    """Fill reconstructed H invariant mass 2D histograms"""

    logger.debug(hist.name)
    if len(mh1) != 0 and len(mh2) != 0:
        hist.fill(np.column_stack((mh1, mh2)))


# def fill_reco_mH_2d_histograms(events: dict, hists: list) -> None:
#     """Fill reconstructed H invariant mass 2D histograms"""

#     mH_plane_hists = find_all_hists(hists, "mH_plane")

#     mH1 = events["reco_H1_m_DL1dv01_70"]
#     mH2 = events["reco_H2_m_DL1dv01_70"]

#     hist = find_hist(mH_plane_hists, lambda h: "mH_plane" in h.name)
#     logger.debug(hist.name)
#     hist.fill(np.column_stack((mH1, mH2)))
#     hist = find_hist(
#         mH_plane_hists, lambda h: "mH_plane_trigPassed_allTriggersOR" in h.name
#     )
#     trigs_OR_decisions = get_all_trigs_or(events, triggers_run3_all)
#     hist.fill(
#         np.column_stack(
#             (np.array(mH1[trigs_OR_decisions]), np.array(mH2[trigs_OR_decisions]))
#         )
#     )
#     for trig in triggers_run3_all:
#         hist = find_hist(
#             mH_plane_hists, lambda h: f"mH_plane_trigPassed_{trig}" in h.name
#         )
#         logger.debug(hist.name)
#         trig_decisions = events[trig]
#         hist.fill(
#             np.column_stack(
#                 (np.array(mH1[trig_decisions]), np.array(mH2[trig_decisions]))
#             )
#         )


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


def fill_hh_deltaeta_histograms(h1_events, h2_events, hists=list) -> None:
    """Fill HH deltaeta histograms"""

    hh_deltar = abs(h1_events.eta - h2_events.eta)
    for hist in hists:
        logger.debug(hist.name)
        hist.fill(np.array(hh_deltar))


def fill_top_veto_histograms(discriminant, hists: list) -> None:
    """Fill top veto histograms"""

    print("top veto", np.array(ak.flatten(discriminant)))
    for hist in hists:
        logger.debug(hist.name)
        hist.fill(np.array(ak.flatten(discriminant)))
