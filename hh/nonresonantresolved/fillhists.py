import numpy as np
import vector as p4
import awkward as ak
from hh.shared.utils import (
    logger,
    find_hist,
    kin_labels,
    find_hists_by_name,
    format_btagger_model_name,
)
from hh.nonresonantresolved.pairing import pairing_methods
from .selection import (
    select_n_jets_events,
)


def fill_hists(
    events: ak.Record,
    hists: list,
    selections: dict,
    is_mc: bool = True,
) -> list:
    """Fill histograms for analysis regions"""
    if is_mc:
        fill_H_histograms(
            h1=p4.zip({v: events[f"h1_truth_{v}"] for v in kin_labels}),
            h2=p4.zip({v: events[f"h2_truth_{v}"] for v in kin_labels}),
            weights=events.event_weight,
            hists=find_hists_by_name(hists, "h[12]_(pt|eta|phi|mass)_truth"),
        )
        fill_HH_histograms(
            hh=p4.zip({v: events[f"hh_truth_{v}"] for v in kin_labels}),
            weights=events.event_weight,
            hists=find_hists_by_name(hists, "hh_(pt|eta|phi|mass)_truth"),
        )
        fill_reco_truth_matched_jets_histograms(events, hists)
        fill_reco_vs_truth_variable_response_histograms(events, hists, kin_labels)
        ## fill_hh_jets_vs_trigs_histograms(events, hists)
        if "btagging" in selections:
            bjets_sel = selections["btagging"]
            if isinstance(bjets_sel, dict):
                bjets_sel = [bjets_sel]
            for i_bjets_sel in bjets_sel:
                btag_model = i_bjets_sel["model"]
                btag_eff = i_bjets_sel["efficiency"]
                btag_count = i_bjets_sel["count"]["value"]
                btag_op = i_bjets_sel["count"]["operator"]
                btagger = format_btagger_model_name(
                    btag_model,
                    btag_eff,
                )
                fill_reco_hh_histograms(events, hists, btag_count, btagger)
                fill_hh_jets_pairings_histograms(events, hists, btag_count, btagger)
                fill_mHH_plane_vs_pairing_histograms(events, hists, btag_count, btagger)
                fill_X_HH_histograms(events, hists, btag_count, btagger)
                fill_mHH_regions_histograms(events, hists, btag_count, btagger)
    return hists


def fill_H_histograms(h1, h2, weights=None, hists: list = None) -> None:
    """Fill reconstructed H1 and H2 kinematics 1D histograms"""

    weights = ak.to_numpy(weights) if weights is not None else None
    if ak.count(h1) != 0 and ak.count(h2) != 0:
        for h_i, h in zip(["1", "2"], [h1, h2]):
            for kin_var in kin_labels:
                hist = find_hist(hists, lambda h: f"h{h_i}_{kin_var}" in h.name)
                logger.debug(hist.name)
                if kin_var == "pt":
                    hist.fill(ak.to_numpy(h.pt), weights=weights)
                elif kin_var == "eta":
                    hist.fill(ak.to_numpy(h.eta), weights=weights)
                elif kin_var == "phi":
                    hist.fill(ak.to_numpy(h.phi), weights=weights)
                elif kin_var == "mass":
                    hist.fill(ak.to_numpy(h.mass), weights=weights)


def fill_HH_histograms(hh, weights=None, hists: list = None) -> None:
    """Fill diHiggs (HH) kinematics histograms"""

    weights = ak.to_numpy(weights) if weights is not None else None
    if ak.count(hh, axis=0) != 0:
        for kin_var in kin_labels:
            hist = find_hist(hists, lambda h: f"hh_{kin_var}" in h.name)
            logger.debug(hist.name)
            if kin_var == "pt":
                hist.fill(ak.to_numpy(hh.pt), weights=weights)
            elif kin_var == "eta":
                hist.fill(ak.to_numpy(hh.eta), weights=weights)
            elif kin_var == "phi":
                hist.fill(ak.to_numpy(hh.phi), weights=weights)
            elif kin_var == "mass":
                hist.fill(ak.to_numpy(hh.mass), weights=weights)


def fill_reco_hh_histograms(events, hists, n_btag, btagger) -> None:
    """Fill reco HH histograms"""

    h1_truth_p4 = p4.zip({v: events[f"h1_truth_{v}"] for v in kin_labels})
    h2_truth_p4 = p4.zip({v: events[f"h2_truth_{v}"] for v in kin_labels})
    hh_truth_p4 = h1_truth_p4 + h2_truth_p4
    weights = events.event_weight
    valid_event = ~ak.is_none(events[f"valid_central_{n_btag}_btag_{btagger}_jets"])
    fill_HH_histograms(
        hh=hh_truth_p4[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(
            hists,
            f"hh_(pt|eta|phi|mass)_truth_reco_central_{n_btag}b_{btagger}_jets_selection",
        ),
    )
    valid_event = ~ak.is_none(
        events[f"reco_truth_matched_central_{n_btag}_btag_{btagger}_jets"]
    )
    fill_HH_histograms(
        hh=hh_truth_p4[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(
            hists,
            f"hh_(pt|eta|phi|mass)_truth_reco_central_{n_btag}b_{btagger}_4_plus_truth_matched_jets_selection",
        ),
    )
    valid_event = ~ak.is_none(
        events[f"reco_truth_matched_central_{n_btag}_btag_{btagger}_jets_v2"]
    )
    fill_HH_histograms(
        hh=hh_truth_p4[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(
            hists,
            f"hh_(pt|eta|phi|mass)_truth_reco_central_{n_btag}b_{btagger}_4_plus_truth_matched_jets_selection_v2",
        ),
    )
    for pairing in pairing_methods:
        valid_event = ~ak.is_none(
            events[f"correct_hh_{n_btag}b_{btagger}_{pairing}_mask"], axis=0
        )
        fill_HH_histograms(
            hh=hh_truth_p4[valid_event],
            weights=weights[valid_event],
            hists=find_hists_by_name(
                hists,
                f"hh_(pt|eta|phi|mass)_truth_reco_central_{n_btag}b_{btagger}_4_plus_truth_matched_jets_correct_{pairing}_selection",
            ),
        )


def fill_reco_truth_matched_jets_histograms(events, hists: list) -> None:
    """Fill reco truth matched jets histograms"""

    jets_p4 = p4.zip({v: events[f"jet_{v}"] for v in kin_labels})
    weights = events.event_weight
    # reconstruct HH using central jets selecting only events that have 4 central truth matched jets
    truth_matched_jets_mask = events.reco_truth_matched_central_jets
    jets_truth_matched_p4 = jets_p4[truth_matched_jets_mask]
    reco_hh_p4 = ak.sum(
        jets_truth_matched_p4[:, 0:4], axis=1
    )  # check that it's summing only first 4 jets
    valid_event = ~ak.is_none(reco_hh_p4, axis=0)
    fill_HH_histograms(
        hh=reco_hh_p4[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(hists, "hh_(pt|eta|phi|mass)_reco_truth_matched"),
    )
    truth_hh = p4.zip({v: events[f"hh_truth_{v}"] for v in kin_labels})
    fill_HH_histograms(
        hh=truth_hh[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(hists, "hh_(pt|eta|phi|mass)_truth_reco_matched"),
    )
    ### jets truth matched with HadronConeExclTruthLabelID ###
    truth_matched_jets_mask = events.reco_truth_matched_central_jets_v2
    jets_truth_matched_p4 = jets_p4[truth_matched_jets_mask]
    reco_hh_p4 = ak.sum(jets_truth_matched_p4[:, 0:4], axis=1)
    valid_event = ~ak.is_none(reco_hh_p4, axis=0)
    fill_HH_histograms(
        hh=reco_hh_p4[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(hists, "hh_(pt|eta|phi|mass)_reco_truth_matched_v2"),
    )

    h1_truth_p4 = p4.zip({v: events[f"h1_truth_{v}"] for v in kin_labels})
    h2_truth_p4 = p4.zip({v: events[f"h2_truth_{v}"] for v in kin_labels})
    hh_truth_p4 = h1_truth_p4 + h2_truth_p4
    valid_event = ~ak.is_none(events.valid_central_jets, axis=0)
    fill_HH_histograms(
        hh=hh_truth_p4[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(
            hists, "hh_(pt|eta|phi|mass)_truth_reco_central_jets_selection"
        ),
    )
    valid_event = ~ak.is_none(events.reco_truth_matched_central_jets, axis=0)
    fill_HH_histograms(
        hh=hh_truth_p4[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(
            hists,
            "hh_(pt|eta|phi|mass)_truth_reco_central_truth_matched_jets_selection",
        ),
    )


def fill_reco_vs_truth_variable_response_histograms(
    events, hists: list, vars: list
) -> None:
    """Fill variable response histograms.

    Parameters
    ----------
    events : ak.Array: ak.Record
        Events record.
    hists : list: Hist
        List of histograms to fill.
    vars : list: str
        1D kinematic variables.
    """

    jets_p4 = p4.zip({v: events[f"jet_{v}"] for v in kin_labels})
    # reconstruct HH using central jets selecting only events that have 4 central truth matched jets
    truth_matched_jets_mask = events.reco_truth_matched_central_jets
    jets_truth_matched_p4 = jets_p4[truth_matched_jets_mask]
    reco_hh_p4 = ak.sum(jets_truth_matched_p4[:, 0:4], axis=1)
    valid_event = ~ak.is_none(reco_hh_p4, axis=0)
    reco_hh_p4 = reco_hh_p4[valid_event]
    reco_hh = ak.zip(
        {
            "pt": reco_hh_p4.pt,
            "eta": reco_hh_p4.eta,
            "phi": reco_hh_p4.phi,
            "mass": reco_hh_p4.mass,
        }
    )
    truth_hh = ak.zip({v: events[f"hh_truth_{v}"] for v in kin_labels})
    truth_matched_hh = truth_hh[valid_event]
    weights = events.event_weight[valid_event]
    for var in vars:
        hist = find_hist(hists, lambda h: f"hh_{var}_reco_vs_truth_response" in h.name)
        var_res = (reco_hh[var] - truth_matched_hh[var]) / truth_matched_hh[var]
        if hist:
            logger.debug(hist.name)
            hist.fill(ak.to_numpy(var_res) * 100, weights=ak.to_numpy(weights))


def fill_hh_jets_vs_trigs_histograms(events, hists: list) -> None:
    """Fill leading jets histograms"""

    jets = ak.zip({v: events[f"jet_{v}"] for v in kin_labels})
    trigs = ak.zip(
        {
            "2b2j_asym": events["trig_assym_2b2j_delayed"],
        }
    )
    weights = events.event_weight
    categories = [
        "_truth_matched",
        "_truth_matched_4_btags",
        "_truth_matched_2b2j_asym",
        "_truth_matched_2b2j_asym_n_btags",
        "_truth_matched_2b2j_asym_4_btags",
    ]
    for cat in categories:
        for i in range(4):
            for kin_var in kin_labels:
                hist = find_hist(
                    hists, lambda h: f"hh_jet_{i+1}_{kin_var}{cat}" in h.name
                )
                if hist:
                    logger.debug(hist.name)
                    if (trig := "2b2j_asym") in cat:
                        passed_trig = trigs[trig]
                    else:
                        trig = None
                        passed_trig = np.ones(len(events), dtype=bool)
                    if "n_btags" in cat and trig is not None:
                        valid_jets = ak.mask(
                            events.reco_truth_matched_btagged_jets, passed_trig
                        )
                    elif "4_btags" in cat and trig is not None:
                        valid_jets = ak.mask(
                            events.reco_truth_matched_btagged_jets, passed_trig
                        )
                    elif "4_btags" in cat and trig is None:
                        valid_jets = events.reco_truth_matched_btagged_jets
                    else:
                        valid_jets = ak.mask(
                            events.reco_truth_matched_central_jets, passed_trig
                        )

                    # finally fill the histogram
                    valid_jets = jets[valid_jets][kin_var][:, i : i + 1]
                    keep_event_mask = ~ak.is_none(valid_jets)
                    hist.fill(
                        ak.to_numpy(ak.flatten(valid_jets)),
                        weights=ak.to_numpy(weights[keep_event_mask]),
                    )


def fill_hh_jets_pairings_histograms(
    events, hists: list, n_btag: int, btagger: str
) -> None:
    """Fill reco truth matched jets histograms"""

    jets_p4 = p4.zip({v: events[f"jet_{v}"] for v in kin_labels})
    weights = events.event_weight
    hh_truth_p4 = p4.zip({v: events[f"hh_truth_{v}"] for v in kin_labels})
    extra_hh_vars = ["sum_jet_pt", "delta_eta"]

    def get_hh_sum_jet_pt(h1_jet_idx, h2_jet_idx):
        return ak.sum(
            jets_p4[ak.concatenate([h1_jet_idx, h2_jet_idx], axis=1)].pt, axis=1
        )

    for pairing in pairing_methods:
        h1_jet_idx = events[f"H1_{n_btag}b_{btagger}_{pairing}_jet_idx"]
        h2_jet_idx = events[f"H2_{n_btag}b_{btagger}_{pairing}_jet_idx"]
        h1_p4 = ak.sum(jets_p4[h1_jet_idx], axis=1)
        h2_p4 = ak.sum(jets_p4[h2_jet_idx], axis=1)
        hh_reco_p4 = h1_p4 + h2_p4
        hh_delta_eta = h1_p4.eta - h2_p4.eta
        hh_sum_jet_pt = get_hh_sum_jet_pt(h1_jet_idx, h2_jet_idx)
        valid_event_paired = ~ak.is_none(
            events[f"reco_truth_matched_central_{n_btag}_btag_{btagger}_jets"], axis=0
        )
        valid_event_paired_correct = ~ak.is_none(
            events[f"correct_hh_{n_btag}b_{btagger}_{pairing}_mask"],
            axis=0,
        )
        valid_events = [valid_event_paired, valid_event_paired_correct]
        for cat, valid_event in zip(
            [
                f"reco_{n_btag}_btag_{btagger}_{pairing}",
                f"reco_{n_btag}_btag_{btagger}_{pairing}_correct",
            ],
            valid_events,
        ):
            fill_HH_histograms(
                hh=hh_reco_p4[valid_event],
                weights=weights[valid_event],
                hists=find_hists_by_name(hists, f"hh_(pt|eta|phi|mass)_{cat}"),
            )
            hist = find_hist(hists, lambda h: f"hh_{extra_hh_vars[0]}_{cat}" in h.name)
            hist.fill(
                ak.to_numpy(hh_sum_jet_pt[valid_event]),
                weights=ak.to_numpy(weights[valid_event]),
            )
            hist = find_hist(hists, lambda h: f"hh_{extra_hh_vars[1]}_{cat}" in h.name)
            hist.fill(
                ak.to_numpy(hh_delta_eta[valid_event]),
                weights=ak.to_numpy(weights[valid_event]),
            )
        for cat, valid_event in zip(
            [
                f"reco_truth_matched_{n_btag}_btag_{btagger}_{pairing}",
                f"reco_truth_matched_{n_btag}_btag_{btagger}_{pairing}_correct",
            ],
            valid_events,
        ):
            fill_HH_histograms(
                hh=hh_truth_p4[valid_event],
                weights=weights[valid_event],
                hists=find_hists_by_name(hists, f"hh_(pt|eta|phi|mass)_{cat}"),
            )
            hist = find_hist(hists, lambda h: f"hh_{extra_hh_vars[0]}_{cat}" in h.name)
            hist.fill(
                ak.to_numpy(hh_sum_jet_pt[valid_event]),
                weights=ak.to_numpy(weights[valid_event]),
            )
            hist = find_hist(hists, lambda h: f"hh_{extra_hh_vars[1]}_{cat}" in h.name)
            hist.fill(
                ak.to_numpy(hh_delta_eta[valid_event]),
                weights=ak.to_numpy(weights[valid_event]),
            )


def fill_mHH_regions_histograms(events, hists: list, n_btag: int, btagger: str) -> None:
    jet_p4 = p4.zip(
        {var: events[f"jet_{var}"] for var in kin_labels},
    )
    weights = events.event_weight
    for pairing in pairing_methods:
        h1_jets_idx = events[f"H1_{n_btag}b_{btagger}_{pairing}_jet_idx"]
        h2_jets_idx = events[f"H2_{n_btag}b_{btagger}_{pairing}_jet_idx"]
        h1_p4 = ak.sum(jet_p4[h1_jets_idx], axis=1)
        h2_p4 = ak.sum(jet_p4[h2_jets_idx], axis=1)
        hh_reco_p4 = h1_p4 + h2_p4
        for region in ["signal", "control"]:
            region_mask = events[f"{region}_{n_btag}b_{btagger}_{pairing}_mask"]
            fill_HH_histograms(
                hh=hh_reco_p4[region_mask],
                weights=weights[region_mask],
                hists=find_hists_by_name(
                    hists,
                    f"hh_(pt|eta|phi|mass)_reco_{region}_{n_btag}b_{btagger}_{pairing}",
                ),
            )
            lt_300_GeV_mask = ~ak.is_none(
                ak.mask(region_mask, (hh_reco_p4).mass < 300_000), axis=0
            )
            fill_HH_histograms(
                hh=hh_reco_p4[lt_300_GeV_mask],
                weights=weights[lt_300_GeV_mask],
                hists=find_hists_by_name(
                    hists,
                    f"hh_(pt|eta|phi|mass)_reco_{region}_{n_btag}b_{btagger}_{pairing}_lt_300_GeV",
                ),
            )
            geq_300_GeV_mask = ~ak.is_none(
                ak.mask(region_mask, (hh_reco_p4).mass >= 300_000), axis=0
            )
            fill_HH_histograms(
                hh=hh_reco_p4[geq_300_GeV_mask],
                weights=weights[geq_300_GeV_mask],
                hists=find_hists_by_name(
                    hists,
                    f"hh_(pt|eta|phi|mass)_reco_{region}_{n_btag}b_{btagger}_{pairing}_geq_300_GeV",
                ),
            )


def fill_mHH_plane_vs_pairing_histograms(
    events, hists: list, n_btag: int, btagger: str
) -> None:
    jet_p4 = p4.zip(
        {var: events[f"jet_{var}"] for var in kin_labels},
    )
    weights = events.event_weight
    for pairing in pairing_methods:
        h1_jets_idx = events[f"H1_{n_btag}b_{btagger}_{pairing}_jet_idx"]
        h2_jets_idx = events[f"H2_{n_btag}b_{btagger}_{pairing}_jet_idx"]
        h1_p4 = ak.sum(jet_p4[h1_jets_idx], axis=1)
        h2_p4 = ak.sum(jet_p4[h2_jets_idx], axis=1)
        valid_event_mask = ~ak.is_none(h1_p4)
        fill_mH_2d_histograms(
            mh1=h1_p4.mass[valid_event_mask],
            mh2=h2_p4.mass[valid_event_mask],
            weights=weights[valid_event_mask],
            hist=find_hist(
                hists,
                lambda h: f"mHH_plane_reco_{n_btag}b_{btagger}_{pairing}" in h.name,
            ),
        )
        lt_300_GeV_mask = ~ak.is_none(
            ak.mask(valid_event_mask, (h1_p4 + h2_p4).mass < 300_000), axis=0
        )
        fill_mH_2d_histograms(
            mh1=h1_p4.mass[lt_300_GeV_mask],
            mh2=h2_p4.mass[lt_300_GeV_mask],
            weights=weights[lt_300_GeV_mask],
            hist=find_hist(
                hists,
                lambda h: f"mHH_plane_reco_{n_btag}b_{btagger}_{pairing}_lt_300_GeV"
                in h.name,
            ),
        )
        geq_300_GeV_mask = ~ak.is_none(
            ak.mask(valid_event_mask, (h1_p4 + h2_p4).mass >= 300_000), axis=0
        )
        fill_mH_2d_histograms(
            mh1=h1_p4.mass[geq_300_GeV_mask],
            mh2=h2_p4.mass[geq_300_GeV_mask],
            weights=weights[geq_300_GeV_mask],
            hist=find_hist(
                hists,
                lambda h: f"mHH_plane_reco_{n_btag}b_{btagger}_{pairing}_geq_300_GeV"
                in h.name,
            ),
        )


def fill_mH_2d_histograms(mh1, mh2, weights, hist) -> None:
    """Fill reconstructed H invariant mass 2D histograms"""

    logger.debug(hist.name)
    if ak.count(mh1) != 0 and ak.count(mh2) != 0:
        mHH = np.column_stack((mh1, mh2))
        hist.fill(ak.to_numpy(mHH), weights=ak.to_numpy(weights))


def fill_X_HH_histograms(events, hists, n_btag: int, btagger: str) -> None:
    """Fill X_HH histograms"""
    for pairing in pairing_methods:
        for region in ["signal", "control"]:
            hist = find_hist(
                hists,
                lambda h: f"hh_mass_discrim_reco_{region}_{n_btag}b_{btagger}_{pairing}"
                in h.name,
            )
            if hist:
                logger.debug(hist.name)
                hist_values = events[f"X_HH_{n_btag}b_{btagger}_{pairing}_discrim"]
                valid_event = ~ak.is_none(hist_values, axis=0)
                hist.fill(
                    ak.to_numpy(hist_values[valid_event]),
                    weights=ak.to_numpy(events.event_weight[valid_event]),
                )


#### Old plotting functions ####
def fill_event_no_histograms(events, hists: list) -> None:
    """Fill event number histograms"""

    hist = find_hist(hists, lambda h: "event_number" in h.name)
    logger.debug(hist.name)
    hist.fill(ak.to_numpy(events.event_number))


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
            if "signal" in h_name and "signal_event" in events.fields:
                valid = events.signal_event
            elif "control" in h_name and "control_event" in events.fields:
                valid = events.control_event
            else:
                valid = np.ones(len(events), dtype=bool)
            hist.fill(ak.to_numpy(events[valid].mc_event_weights[:, 0]))
    for h_name in [
        "total_event_weight",
        "total_event_weight_baseline_signal_region",
        "totla_event_weight_baseline_control_region",
    ]:
        hist = find_hist(hists, lambda h: h_name in h.name)
        if hist:
            logger.debug(hist.name)
            if "signal" in h_name and "signal_event" in events.fields:
                valid = events.signal_event
            elif "control" in h_name and "control_event" in events.fields:
                valid = events.control_event
            else:
                valid = np.ones(len(events), dtype=bool)
            hist.fill(ak.to_numpy(events[valid].event_weight))


def fill_jet_kin_histograms(events, hists: list) -> None:
    """Fill jet kinematics histograms"""

    regions = ["", "_baseline_signal_region", "_baseline_control_region"]
    for region in regions:
        for kin_var in kin_labels:
            hist = find_hist(hists, lambda h: f"jet_{kin_var}{region}" in h.name)
            if hist:
                logger.debug(hist.name)
                if "signal" in hist.name and "signal_event" in events.fields:
                    valid = events.signal_event
                elif "control" in hist.name and "control_event" in events.fields:
                    valid = events.control_event
                else:
                    valid = np.ones(len(events), dtype=bool)
                jets = events[valid][f"jet_{kin_var}"]
                event_weight = events[valid].event_weight[:, np.newaxis]
                event_weight, _ = ak.broadcast_arrays(event_weight, jets)
                hist.fill(
                    ak.to_numpy(ak.flatten(jets)),
                    weights=ak.to_numpy(ak.flatten(event_weight)),
                )


def fill_leading_jets_histograms(events, hists: list) -> None:
    """Fill leading jets histograms"""

    jet_pt = ak.sort(events["jet_pt"], ascending=False)

    regions = ["", "_baseline_signal_region", "_baseline_control_region"]
    for region in regions:
        for i in range(4):
            hist = find_hist(
                hists, lambda h: f"leading_jet_{i + 1}_pt{region}" in h.name
            )
            if hist:
                logger.debug(hist.name)
                if "signal" in hist.name and "signal_event" in events.fields:
                    valid = events.signal_event
                elif "control" in hist.name and "control_event" in events.fields:
                    valid = events.control_event
                else:
                    valid = np.ones(len(events), dtype=bool)
                hist.fill(ak.to_numpy(ak.flatten(jet_pt[valid, i : i + 1])))


def fill_truth_matched_mjj_histograms(events, hists: list) -> None:
    """Fill reconstructed mjj invariant mass 1D histograms"""

    leading_jets = select_n_jets_events(
        events,
        pt_cut=20_000,
        eta_cut=2.5,
        njets_cut=4,
        jet_vars=kin_labels.keys(),
    )
    leading_jets_p4 = p4.zip(
        {v: leading_jets[f"resolved_truth_matched_jet_{v}"]} for v in kin_labels
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

    jets_p4 = p4.zip({v: events[f"resolved_truth_matched_jet_{v}"]} for v in kin_labels)

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


def fill_H1_H2_HH_histograms(events, hists: list) -> None:
    if all(
        field in events.fields
        for field in ["leading_h_jet_idx", "subleading_h_jet_idx"]
    ):
        leading_h_jet_idx = events.leading_h_jet_idx
        subleading_h_jet_idx = events.subleading_h_jet_idx
        jet_p4 = p4.zip(
            {var: events[f"jet_{var}"] for var in kin_labels},
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
        fill_H_histograms(
            h1=h1,
            h2=h2,
            weights=events.event_weight,
            hists=find_hists_by_name(hists, "h[12]_(pt|eta|phi|mass)_baseline"),
        )
        fill_mH_2d_histograms(
            mh1=h1.mass,
            mh2=h2.mass,
            weights=events.event_weight,
            hist=find_hist(hists, lambda h: "mHH_plane_baseline" in h.name),
        )
        fill_HH_histograms(
            hh=hh,
            weights=events.event_weight,
            hists=find_hists_by_name(hists, "hh_(pt|eta|phi|mass)_baseline"),
        )

        if "signal_event" in events.fields and sum(events.signal_event) > 0:
            signal_event = events.signal_event
            signal_h1 = h1[signal_event]
            signal_h2 = h2[signal_event]
            signal_event_weights = events.event_weight[signal_event]
            fill_H_histograms(
                h1=signal_h1,
                h2=signal_h2,
                weights=signal_event_weights,
                hists=find_hists_by_name(
                    hists, "h[12]_(pt|eta|phi|mass)_baseline_signal_region"
                ),
            )
            fill_mH_2d_histograms(
                mh1=signal_h1.mass,
                mh2=signal_h2.mass,
                weights=signal_event_weights,
                hist=find_hist(
                    hists, lambda h: "mHH_plane_baseline_signal_region" in h.name
                ),
            )
            fill_HH_histograms(
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

        if "control_event" in events.fields and sum(events.control_event) > 0:
            control_event = events.control_event
            control_h1 = h1[control_event]
            control_h2 = h2[control_event]
            control_event_weight = events.event_weight[control_event]
            fill_H_histograms(
                h1=control_h1,
                h2=control_h2,
                weights=control_event_weight,
                hists=find_hists_by_name(
                    hists, "h[12]_(pt|eta|phi|mass)_baseline_control_region"
                ),
            )
            fill_mH_2d_histograms(
                mh1=control_h1.mass,
                mh2=control_h2.mass,
                weights=control_event_weight,
                hist=find_hist(
                    hists, lambda h: "mHH_plane_baseline_control_region" in h.name
                ),
            )
            fill_HH_histograms(
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
                ak.to_numpy(ak.flatten(h_truth_jet_idx, axis=None)),
                weights=(
                    ak.to_numpy(np.repeat(weights, 2)) if weights is not None else None
                ),
            )


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
