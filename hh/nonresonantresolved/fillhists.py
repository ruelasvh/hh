import numpy as np
import vector as p4
import awkward as ak
from hh.shared.utils import (
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
    selection: dict,
    is_mc: bool = True,
) -> list:  # hists:
    """Fill histograms for analysis regions"""
    fill_event_no_histograms(events, hists)
    fill_jet_kin_histograms(events, hists)
    fill_leading_jets_histograms(events, hists)

    if "top_veto" in selection:
        fill_top_veto_histograms(
            events,
            hists=find_hists(hists, lambda h: "top_veto" in h.name),
        )

    if "hh_deltaeta_veto" in selection:
        fill_hh_deltaeta_histograms(
            events,
            hists=find_hists(hists, lambda h: "hh_deltaeta" in h.name),
        )

    if "hh_mass_veto" in selection:
        fill_hh_mass_discrim_histograms(
            events,
            hists=find_hists(hists, lambda h: "hh_mass_discrim" in h.name),
        )

    if all(
        sel in selection
        for sel in [
            "central_jets",
            "btagging",
            "top_veto",
            "hh_deltaeta_veto",
            "hh_mass_veto",
        ]
    ):
        fill_HH_histograms(events, hists)

    if is_mc:
        fill_event_weight_histograms(events, hists)

    return hists


def fill_event_no_histograms(events, hists: list) -> None:
    """Fill event number histograms"""

    hist = find_hist(hists, lambda h: "event_number" in h.name)
    logger.debug(hist.name)
    hist.fill(events.event_number.to_numpy())


def fill_event_weight_histograms(events, hists: list) -> None:
    """Fill mc event weight histograms"""

    for h_name in [
        "mc_event_weight",
        "mc_event_weight_baseline_signal_region",
        "mc_event_weight_baseline_control_region",
    ]:
        hist = find_hist(hists, lambda h: h_name in h.name)
        if hist:
            logger.debug(hist.name)
            if "signal" in h_name:
                valid = events.signal_event
            elif "control" in h_name:
                valid = events.control_event
            else:
                valid = np.ones(len(events), dtype=bool)
            hist.fill(events[valid].mc_event_weights[:, 0].to_numpy())
    for h_name in [
        "total_event_weight",
        "total_event_weight_baseline_signal_region",
        "totla_event_weight_baseline_control_region",
    ]:
        hist = find_hist(hists, lambda h: h_name in h.name)
        if hist:
            logger.debug(hist.name)
            if "signal" in h_name:
                valid = events.signal_event
            elif "control" in h_name:
                valid = events.control_event
            else:
                valid = np.ones(len(events), dtype=bool)
            hist.fill(events[valid].event_weight.to_numpy())


def fill_jet_kin_histograms(events, hists: list) -> None:
    """Fill jet kinematics histograms"""

    regions = ["", "_baseline_signal_region", "_baseline_control_region"]
    for region in regions:
        for kin_var in kin_labels.keys():
            hist = find_hist(hists, lambda h: f"jet_{kin_var}{region}" in h.name)
            if hist:
                logger.debug(hist.name)
                if "signal" in region:
                    valid = events.signal_event
                elif "control" in region:
                    valid = events.control_event
                else:
                    valid = np.ones(len(events), dtype=bool)
                jets = events[valid][f"jet_{kin_var}"]
                event_weight = events[valid].event_weight[:, np.newaxis]
                event_weight, _ = ak.broadcast_arrays(event_weight, jets)
                hist.fill(
                    np.array(ak.flatten(jets)),
                    weights=np.array(ak.flatten(event_weight)),
                )


def fill_leading_jets_histograms(events, hists: list) -> None:
    """Fill leading jets histograms"""

    jet_pt = ak.sort(events["jet_pt"], ascending=False)

    regions = ["", "_baseline_signal_region", "_baseline_control_region"]
    for region in regions:
        for ith_jet in [1, 2, 3, 4]:
            hist = find_hist(
                hists, lambda h: f"leading_jet_{ith_jet}_pt{region}" in h.name
            )
            if hist:
                logger.debug(hist.name)
                if "signal" in region:
                    valid = events.signal_event
                elif "control" in region:
                    valid = events.control_event
                else:
                    valid = np.ones(len(events), dtype=bool)
                hist.fill(np.array(jet_pt[valid, ith_jet - 1]))


def fill_truth_matched_mjj_histograms(events, hists: list) -> None:
    """Fill reconstructed mjj invariant mass 1D histograms"""

    jet_vars = {
        "pt": "resolved_truth_mached_jet_pt",
        "eta": "resolved_truth_mached_jet_eta",
        "phi": "resolved_truth_mached_jet_phi",
        "mass": "resolved_truth_mached_jet_m",
    }
    leading_jets = select_n_jets_events(
        events,
        pt_cut=20_000,
        eta_cut=2.5,
        njets_cut=4,
        jet_vars=jet_vars.values(),
    )
    leading_jets_p4 = p4.zip(
        {key: leading_jets[value] for key, value in jet_vars.items()}
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

    jet_vars = {
        "pt": "resolved_truth_mached_jet_pt",
        "eta": "resolved_truth_mached_jet_eta",
        "phi": "resolved_truth_mached_jet_phi",
        "mass": "resolved_truth_mached_jet_m",
    }
    jets_p4 = p4.zip({key: events[value] for key, value in jet_vars.items()})

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


def fill_HH_histograms(events, hists: list) -> None:
    if all(
        field in events.fields
        for field in ["leading_h_jet_idx", "subleading_h_jet_idx"]
    ):
        leading_h_jet_idx = events.leading_h_jet_idx
        subleading_h_jet_idx = events.subleading_h_jet_idx
        jet_p4 = p4.zip(
            {var: events[f"jet_{var}"] for var in kin_labels.keys()},
        )
        h1 = (
            jet_p4[leading_h_jet_idx[:, 0, np.newaxis]]
            + jet_p4[leading_h_jet_idx[:, 1, np.newaxis]]
        )
        h1 = h1[:, 0]  # remove the extra dimension
        h2 = (
            jet_p4[subleading_h_jet_idx[:, 0, np.newaxis]]
            + jet_p4[subleading_h_jet_idx[:, 1, np.newaxis]]
        )
        h2 = h2[:, 0]  # remove the extra dimension
        hh = h1 + h2
        fill_reco_H_histograms(
            h1=h1,
            h2=h2,
            weights=events.event_weight,
            hists=find_hists_by_name(hists, "h[12]_(pt|eta|phi|mass)_baseline"),
        )
        fill_reco_mH_2d_histograms(
            mh1=h1.mass,
            mh2=h2.mass,
            weights=events.event_weight,
            hist=find_hist(hists, lambda h: "mH_plane_baseline" in h.name),
        )
        fill_reco_HH_histograms(
            hh=hh,
            weights=events.event_weight,
            hists=find_hists_by_name(hists, "hh_(pt|eta|phi|mass)_baseline"),
        )

        if "signal_event" in events.fields:
            signal_event = events.signal_event
            signal_h1 = h1[signal_event]
            signal_h2 = h2[signal_event]
            signal_event_weights = events.event_weight[signal_event]
            fill_reco_H_histograms(
                h1=signal_h1,
                h2=signal_h2,
                weights=signal_event_weights,
                hists=find_hists_by_name(
                    hists, "h[12]_(pt|eta|phi|mass)_baseline_signal_region"
                ),
            )
            fill_reco_mH_2d_histograms(
                mh1=signal_h1.mass,
                mh2=signal_h2.mass,
                weights=signal_event_weights,
                hist=find_hist(
                    hists, lambda h: "mH_plane_baseline_signal_region" in h.name
                ),
            )
            fill_reco_HH_histograms(
                hh=hh[signal_event],
                weights=events.event_weight[signal_event],
                hists=find_hists_by_name(
                    hists, "hh_(pt|eta|phi|mass)_baseline_signal_region"
                ),
            )
            fill_reco_H_truth_jet_histograms(
                events[signal_event],
                weights=events.event_weight[signal_event],
                hists=find_hists_by_name(
                    hists, "h[12]_truth_jet_baseline_signal_region"
                ),
            )

        if "control_event" in events.fields:
            control_event = events.control_event
            control_h1 = h1[control_event]
            control_h2 = h2[control_event]
            control_event_weight = events.event_weight[control_event]
            fill_reco_H_histograms(
                h1=control_h1,
                h2=control_h2,
                weights=control_event_weight,
                hists=find_hists_by_name(
                    hists, "h[12]_(pt|eta|phi|mass)_baseline_control_region"
                ),
            )
            fill_reco_mH_2d_histograms(
                mh1=control_h1.mass,
                mh2=control_h2.mass,
                weights=control_event_weight,
                hist=find_hist(
                    hists, lambda h: "mH_plane_baseline_control_region" in h.name
                ),
            )
            fill_reco_HH_histograms(
                hh=hh[control_event],
                weights=events.event_weight[control_event],
                hists=find_hists_by_name(
                    hists, "hh_(pt|eta|phi|mass)_baseline_control_region"
                ),
            )


def fill_reco_H_truth_jet_histograms(events, hists: list, weights=None) -> None:
    """Fill jet truth ID histograms"""

    if "jet_truth_ID" in events.fields:
        h1_truth_jet_idx = events.jet_truth_ID[events.leading_h_jet_idx]
        h2_truth_jet_idx = events.jet_truth_ID[events.subleading_h_jet_idx]
        for ith_H, h_truth_jet_idx in zip([1, 2], [h1_truth_jet_idx, h2_truth_jet_idx]):
            hist = find_hist(
                hists,
                lambda h: f"h{ith_H}_truth_jet_baseline_signal_region" in h.name,
            )
            logger.debug(hist.name)
            hist.fill(
                ak.flatten(h_truth_jet_idx),
                weights=np.repeat(weights, 2) if weights is not None else None,
            )


def fill_reco_H_histograms(h1, h2, weights, hists: list) -> None:
    """Fill reconstructed H1 and H2 kinematics 1D histograms"""

    if ak.count(h1) != 0 and ak.count(h2) != 0:
        for h_i, h in zip(["1", "2"], [h1, h2]):
            for kin_var in kin_labels.keys():
                hist = find_hist(hists, lambda h: f"h{h_i}_{kin_var}" in h.name)
                logger.debug(hist.name)
                if kin_var == "pt":
                    hist.fill(ak.to_numpy(h.pt), weights=ak.to_numpy(weights))
                elif kin_var == "eta":
                    hist.fill(ak.to_numpy(h.eta), weights=ak.to_numpy(weights))
                elif kin_var == "phi":
                    hist.fill(ak.to_numpy(h.phi), weights=ak.to_numpy(weights))
                elif kin_var == "mass":
                    hist.fill(ak.to_numpy(h.mass), weights=ak.to_numpy(weights))


def fill_reco_mH_2d_histograms(mh1, mh2, weights, hist) -> None:
    """Fill reconstructed H invariant mass 2D histograms"""

    logger.debug(hist.name)
    if ak.count(mh1) != 0 and ak.count(mh2) != 0:
        mHH = np.column_stack((mh1, mh2))
        hist.fill(ak.to_numpy(mHH), weights=ak.to_numpy(weights))


def fill_reco_HH_histograms(hh, weights, hists: list) -> None:
    """Fill diHiggs (HH) kinematics histograms"""

    if ak.count(hh, axis=0) != 0:
        for kin_var in kin_labels.keys():
            hist = find_hist(hists, lambda h: f"hh_{kin_var}" in h.name)
            logger.debug(hist.name)
            if kin_var == "pt":
                hist.fill(ak.to_numpy(hh.pt), weights=ak.to_numpy(weights))
            elif kin_var == "eta":
                hist.fill(ak.to_numpy(hh.eta), weights=ak.to_numpy(weights))
            elif kin_var == "phi":
                hist.fill(ak.to_numpy(hh.phi), weights=ak.to_numpy(weights))
            elif kin_var == "mass":
                hist.fill(ak.to_numpy(hh.mass), weights=ak.to_numpy(weights))


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

    if "hh_deltaeta_discriminant" in events.fields:
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

    if "hh_mass_discriminant_signal" in events.fields:
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

    if "X_Wt_discriminant_min" in events.fields:
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

            top_veto_nbtags_hist = find_hist(
                hists, lambda h: "top_veto_n_btags" in h.name
            )
            top_veto_nbtags_hist.fill(np.array(events.btag_num[valid_events]))
