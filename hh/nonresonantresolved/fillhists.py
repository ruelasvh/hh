import numpy as np
import vector as p4
import awkward as ak
from hh.shared.utils import (
    logger,
    find_hist,
    find_hists_by_name,
    format_btagger_model_name,
)
from hh.shared.labels import kin_labels
from hh.nonresonantresolved.pairing import pairing_methods
from hh.nonresonantresolved.selection import (
    select_n_jets_events,
)


def fill_analysis_hists(
    events: ak.Record,
    hists: dict,
    selections: dict,
    is_mc: bool = True,
) -> list:
    """Fill histograms for analysis regions"""

    if is_mc:
        fill_H_histograms(
            h1=p4.zip({v: events[f"h1_truth_{v}"] for v in kin_labels}),
            h2=p4.zip({v: events[f"h2_truth_{v}"] for v in kin_labels}),
            weights=events.event_weight,
            hists=find_hists_by_name(hists, "h[12]_(pt|eta|phi|mass)_truth$"),
        )
        fill_HH_histograms(
            hh=p4.zip({v: events[f"hh_truth_{v}"] for v in kin_labels}),
            weights=events.event_weight,
            hists=find_hists_by_name(hists, "hh_(pt|eta|phi|mass)_truth$"),
        )
        fill_reco_vs_truth_variable_response_histograms(events, hists, kin_labels)
        # need to refactor this
        # fill_hh_jets_vs_trigs_histograms(events, hists)
        #
        if "jets" in selections and "btagging" in selections["jets"]:
            bjets_sel = selections["jets"]["btagging"]
            if isinstance(bjets_sel, dict):
                bjets_sel = [bjets_sel]
            for i_bjets_sel in bjets_sel:
                btag_model = i_bjets_sel["model"]
                btag_eff = i_bjets_sel["efficiency"]
                n_btags = i_bjets_sel["count"]["value"]
                btagger = format_btagger_model_name(
                    btag_model,
                    btag_eff,
                )
                fill_leading_jets_histograms(
                    events,
                    hists,
                    jet_type="jet",
                    hist_prefix=f"leading_resolved_{n_btags}btags_{btagger}_reco_jet",
                    selection_mask=events[f"valid_{n_btags}btags_{btagger}_events"],
                )
                fill_reco_truth_matched_jets_histograms(events, hists, 2, btagger)
                fill_h1_h2_histograms(events, hists, n_btags, btagger)
                fill_hh_jets_histograms(events, hists, n_btags, btagger)
                fill_mHH_plane_histograms(events, hists, n_btags, btagger)
                fill_X_HH_histograms(events, hists, n_btags, btagger)
                fill_HH_abs_deltaeta_discrim_histograms(events, hists, n_btags, btagger)
                fill_top_veto_discrim_histograms(events, hists, n_btags, btagger)
                fill_hh_regions_histograms(events, hists, n_btags, btagger)
                fill_jet_flavor_composition_histograms(
                    events, hists, n_btags, btag_model, btag_eff
                )
                fill_HH_combined_pairing_histograms(events, hists, n_btags, btagger)

    return hists


def fill_H_histograms(h1, h2, weights=None, hists: dict = None) -> None:
    """Fill reconstructed H1 and H2 kinematics 1D histograms"""

    weights = ak.to_numpy(weights) if weights is not None else None
    if ak.count(h1) != 0 and ak.count(h2) != 0:
        for h_i, h in zip(["1", "2"], [h1, h2]):
            for kin_var in kin_labels:
                hist = find_hist(hists, lambda h_name: f"h{h_i}_{kin_var}" in h_name)
                if hist:
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
            hist = find_hist(hists, lambda h_name: f"hh_{kin_var}" in h_name)
            if hist:
                logger.debug(hist.name)
                if kin_var == "pt":
                    hist.fill(ak.to_numpy(hh.pt), weights=weights)
                elif kin_var == "eta":
                    hist.fill(ak.to_numpy(hh.eta), weights=weights)
                elif kin_var == "phi":
                    hist.fill(ak.to_numpy(hh.phi), weights=weights)
                elif kin_var == "mass":
                    hist.fill(ak.to_numpy(hh.mass), weights=weights)


def fill_h1_h2_histograms(events, hists, n_btags, btagger) -> None:
    """Fill reco HH histograms"""

    hh_truth_p4 = p4.zip({v: events[f"hh_truth_{v}"] for v in kin_labels})
    weights = events.event_weight

    for pairing in pairing_methods:
        h1_jets_idx = events[f"H1_{n_btags}btags_{btagger}_{pairing}_jet_idx"]
        h2_jets_idx = events[f"H2_{n_btags}btags_{btagger}_{pairing}_jet_idx"]
        jet_p4 = p4.zip({v: events[f"jet_{v}"] for v in kin_labels})
        h1_p4 = ak.sum(jet_p4[h1_jets_idx], axis=1)
        h2_p4 = ak.sum(jet_p4[h2_jets_idx], axis=1)
        hh_reco_p4 = h1_p4 + h2_p4
        valid_event_mask = events[f"valid_{n_btags}btags_{btagger}_events"]
        fill_H_histograms(
            h1=h1_p4[valid_event_mask],
            h2=h2_p4[valid_event_mask],
            weights=weights[valid_event_mask],
            hists=find_hists_by_name(
                hists,
                f"h[12]_(pt|eta|phi|mass)_reco_{n_btags}btags_{btagger}_{pairing}$",
            ),
        )
        valid_event_correct_pairs_mask = (
            events[f"correct_hh_{n_btags}btags_{btagger}_{pairing}_mask"] == True
        ) & valid_event_mask
        fill_H_histograms(
            h1=h1_p4[valid_event_correct_pairs_mask],
            h2=h2_p4[valid_event_correct_pairs_mask],
            weights=weights[valid_event_correct_pairs_mask],
            hists=find_hists_by_name(
                hists,
                f"h[12]_(pt|eta|phi|mass)_reco_{n_btags}btags_{btagger}_{pairing}_correct_pairs$",
            ),
        )
        valid_event_wrong_pairs_mask = (
            events[f"correct_hh_{n_btags}btags_{btagger}_{pairing}_mask"] == False
        ) & valid_event_mask
        fill_H_histograms(
            h1=h1_p4[valid_event_wrong_pairs_mask],
            h2=h2_p4[valid_event_wrong_pairs_mask],
            weights=weights[valid_event_wrong_pairs_mask],
            hists=find_hists_by_name(
                hists,
                f"h[12]_(pt|eta|phi|mass)_reco_{n_btags}btags_{btagger}_{pairing}_wrong_pairs$",
            ),
        )
        lt_370_GeV_mask = (hh_reco_p4.mass < 370_000) & valid_event_mask
        fill_H_histograms(
            h1=h1_p4[lt_370_GeV_mask],
            h2=h2_p4[lt_370_GeV_mask],
            weights=weights[lt_370_GeV_mask],
            hists=find_hists_by_name(
                hists,
                f"h[12]_(pt|eta|phi|mass)_reco_{n_btags}btags_{btagger}_{pairing}_lt_370_GeV$",
            ),
        )
        geq_370_GeV_mask = (hh_reco_p4.mass >= 370_000) & valid_event_mask
        fill_H_histograms(
            h1=h1_p4[geq_370_GeV_mask],
            h2=h2_p4[geq_370_GeV_mask],
            weights=weights[geq_370_GeV_mask],
            hists=find_hists_by_name(
                hists,
                f"h[12]_(pt|eta|phi|mass)_reco_{n_btags}btags_{btagger}_{pairing}_geq_370_GeV$",
            ),
        )


def fill_reco_truth_matched_jets_histograms(
    events, hists: list, n_btags: int, btagger: str
) -> None:
    """Fill reco truth matched jets histograms"""

    jets_p4 = p4.zip({v: events[f"jet_{v}"] for v in kin_labels})
    weights = events.event_weight
    # reconstruct HH using central jets selecting only events that have 4 central truth matched jets
    truth_matched_jets_mask = events.reco_truth_matched_jets
    jets_truth_matched_p4 = jets_p4[truth_matched_jets_mask]
    reco_hh_p4 = ak.sum(
        jets_truth_matched_p4[:, 0:4], axis=1
    )  # check that it's summing only first 4 jets
    valid_event = ~ak.is_none(reco_hh_p4, axis=0)
    fill_HH_histograms(
        hh=reco_hh_p4[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(hists, "hh_(pt|eta|phi|mass)_reco_truth_matched$"),
    )
    truth_hh = p4.zip({v: events[f"hh_truth_{v}"] for v in kin_labels})
    fill_HH_histograms(
        hh=truth_hh[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(hists, "hh_(pt|eta|phi|mass)_truth_reco_matched$"),
    )
    ### jets truth matched with HadronConeExclTruthLabelID ###
    truth_matched_jets_mask = events.reco_truth_matched_jets_v2
    jets_truth_matched_p4 = jets_p4[truth_matched_jets_mask]
    reco_hh_p4 = ak.sum(jets_truth_matched_p4[:, 0:4], axis=1)
    valid_event = ~ak.is_none(reco_hh_p4, axis=0)
    fill_HH_histograms(
        hh=reco_hh_p4[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(hists, "hh_(pt|eta|phi|mass)_reco_truth_matched_v2$"),
    )

    h1_truth_p4 = p4.zip({v: events[f"h1_truth_{v}"] for v in kin_labels})
    h2_truth_p4 = p4.zip({v: events[f"h2_truth_{v}"] for v in kin_labels})
    hh_truth_p4 = h1_truth_p4 + h2_truth_p4
    valid_event = ~ak.is_none(events[f"valid_{n_btags}btags_{btagger}_events"], axis=0)
    fill_HH_histograms(
        hh=hh_truth_p4[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(
            hists, "hh_(pt|eta|phi|mass)_truth_reco_central_jets_selection$"
        ),
    )
    valid_event = ~ak.is_none(events.reco_truth_matched_jets, axis=0)
    fill_HH_histograms(
        hh=hh_truth_p4[valid_event],
        weights=weights[valid_event],
        hists=find_hists_by_name(
            hists,
            "hh_(pt|eta|phi|mass)_truth_reco_central_truth_matched_jets_selection$",
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
    truth_matched_jets_mask = events.reco_truth_matched_jets
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
        hist = find_hist(
            hists, lambda h_name: f"hh_{var}_reco_vs_truth_response" == h_name
        )
        if hist:
            logger.debug(hist.name)
            var_res = (reco_hh[var] - truth_matched_hh[var]) / truth_matched_hh[var]
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
                            events.reco_truth_matched_jets, passed_trig
                        )

                    # finally fill the histogram
                    valid_jets = jets[valid_jets][kin_var][:, i : i + 1]
                    keep_event_mask = ~ak.is_none(valid_jets)
                    hist.fill(
                        ak.to_numpy(ak.flatten(valid_jets)),
                        weights=ak.to_numpy(weights[keep_event_mask]),
                    )


def fill_hh_jets_histograms(events, hists: list, n_btags: int, btagger: str) -> None:
    """Fill reco truth matched and truth reco matched histograms"""

    hh_truth_p4 = p4.zip({v: events[f"hh_truth_{v}"] for v in kin_labels})
    weights = events.event_weight

    jets_p4 = p4.zip({v: events[f"jet_{v}"] for v in kin_labels})

    def get_hh_sum_jet_pt(h1_jet_idx, h2_jet_idx):
        return ak.sum(
            jets_p4[ak.concatenate([h1_jet_idx, h2_jet_idx], axis=1)].pt, axis=1
        )

    for pairing in pairing_methods:
        # fill truth reco-matched HH histograms
        valid_event_mask = ~ak.is_none(events[f"valid_{n_btags}btags_{btagger}_jets"])
        fill_HH_histograms(
            hh=hh_truth_p4[valid_event_mask],
            weights=weights[valid_event_mask],
            hists=find_hists_by_name(
                hists,
                f"hh_(pt|eta|phi|mass)_truth_reco_central_{n_btags}btags_{btagger}_jets_selection$",
            ),
        )
        valid_event_mask = ~ak.is_none(
            events[f"reco_truth_matched_{n_btags}btags_{btagger}_jets"]
        )
        fill_HH_histograms(
            hh=hh_truth_p4[valid_event_mask],
            weights=weights[valid_event_mask],
            hists=find_hists_by_name(
                hists,
                f"hh_(pt|eta|phi|mass)_truth_reco_central_{n_btags}btags_{btagger}_4_plus_truth_matched_jets_selection$",
            ),
        )
        valid_event_mask = ~ak.is_none(
            events[f"reco_truth_matched_central_{n_btags}btags_{btagger}_jets_v2"]
        )
        fill_HH_histograms(
            hh=hh_truth_p4[valid_event_mask],
            weights=weights[valid_event_mask],
            hists=find_hists_by_name(
                hists,
                f"hh_(pt|eta|phi|mass)_truth_reco_central_{n_btags}btags_{btagger}_4_plus_truth_matched_jets_selection_v2$",
            ),
        )
        valid_event_paired_correct = (
            events[f"correct_hh_{n_btags}btags_{btagger}_{pairing}_mask"] == True
        ) & valid_event_mask
        fill_HH_histograms(
            hh=hh_truth_p4[valid_event_paired_correct],
            weights=weights[valid_event_paired_correct],
            hists=find_hists_by_name(
                hists,
                f"hh_(pt|eta|phi|mass)_truth_reco_central_{n_btags}btags_{btagger}_4_plus_truth_matched_jets_correct_{pairing}_selection$",
            ),
        )

        # fill reco truth-matched HH histograms
        extra_hh_vars = ["sum_jet_pt", "delta_eta"]
        h1_jet_idx = events[f"H1_{n_btags}btags_{btagger}_{pairing}_jet_idx"]
        h2_jet_idx = events[f"H2_{n_btags}btags_{btagger}_{pairing}_jet_idx"]
        h1_p4 = ak.sum(jets_p4[h1_jet_idx], axis=1)
        h2_p4 = ak.sum(jets_p4[h2_jet_idx], axis=1)
        hh_reco_p4 = h1_p4 + h2_p4
        hh_delta_eta = h1_p4.eta - h2_p4.eta
        hh_sum_jet_pt = get_hh_sum_jet_pt(h1_jet_idx, h2_jet_idx)
        valid_event_paired = ~ak.is_none(
            events[f"reco_truth_matched_{n_btags}btags_{btagger}_jets"], axis=0
        )
        valid_event_paired_correct = (
            events[f"correct_hh_{n_btags}btags_{btagger}_{pairing}_mask"] == True
        )
        valid_events = [valid_event_paired, valid_event_paired_correct]
        for cat, valid_event in zip(
            [
                f"reco_{n_btags}btags_{btagger}_{pairing}",
                f"reco_{n_btags}btags_{btagger}_{pairing}_correct_pairs",
            ],
            valid_events,
        ):
            fill_HH_histograms(
                hh=hh_reco_p4[valid_event],
                weights=weights[valid_event],
                hists=find_hists_by_name(hists, f"hh_(pt|eta|phi|mass)_{cat}$"),
            )
            hist = find_hist(
                hists, lambda h_name: f"hh_{extra_hh_vars[0]}_{cat}" == h_name
            )
            hist.fill(
                ak.to_numpy(hh_sum_jet_pt[valid_event]),
                weights=ak.to_numpy(weights[valid_event]),
            )
            hist = find_hist(
                hists, lambda h_name: f"hh_{extra_hh_vars[1]}_{cat}" == h_name
            )
            hist.fill(
                ak.to_numpy(hh_delta_eta[valid_event]),
                weights=ak.to_numpy(weights[valid_event]),
            )
        for cat, valid_event in zip(
            [
                f"reco_truth_matched_{n_btags}btags_{btagger}_{pairing}",
                f"reco_truth_matched_{n_btags}btags_{btagger}_{pairing}_correct_pairs",
            ],
            valid_events,
        ):
            fill_HH_histograms(
                hh=hh_truth_p4[valid_event],
                weights=weights[valid_event],
                hists=find_hists_by_name(hists, f"hh_(pt|eta|phi|mass)_{cat}$"),
            )
            hist = find_hist(
                hists, lambda h_name: f"hh_{extra_hh_vars[0]}_{cat}" == h_name
            )
            hist.fill(
                ak.to_numpy(hh_sum_jet_pt[valid_event]),
                weights=ak.to_numpy(weights[valid_event]),
            )
            hist = find_hist(
                hists, lambda h_name: f"hh_{extra_hh_vars[1]}_{cat}" == h_name
            )
            hist.fill(
                ak.to_numpy(hh_delta_eta[valid_event]),
                weights=ak.to_numpy(weights[valid_event]),
            )


def fill_hh_regions_histograms(events, hists: list, n_btags: int, btagger: str) -> None:
    jet_p4 = p4.zip(
        {var: events[f"jet_{var}"] for var in kin_labels},
    )
    weights = events.event_weight
    for pairing in pairing_methods:
        h1_jets_idx = events[f"H1_{n_btags}btags_{btagger}_{pairing}_jet_idx"]
        h2_jets_idx = events[f"H2_{n_btags}btags_{btagger}_{pairing}_jet_idx"]
        h1_p4 = ak.sum(jet_p4[h1_jets_idx], axis=1)
        h2_p4 = ak.sum(jet_p4[h2_jets_idx], axis=1)
        hh_reco_p4 = h1_p4 + h2_p4
        valid_event_mask = events[f"valid_{n_btags}btags_{btagger}_events"]
        for region in ["signal", "control"]:
            region_mask = (
                events[f"{region}_{n_btags}btags_{btagger}_{pairing}_mask"]
                & valid_event_mask
            )
            fill_HH_histograms(
                hh=hh_reco_p4[region_mask],
                weights=weights[region_mask],
                hists=find_hists_by_name(
                    hists,
                    f"hh_(pt|eta|phi|mass)_reco_{region}_{n_btags}btags_{btagger}_{pairing}$",
                ),
            )
            region_wrong_pairs_mask = (
                events[
                    f"{region}_correct_pairs_{n_btags}btags_{btagger}_{pairing}_mask"
                ]
                == False
            )
            fill_HH_histograms(
                hh=hh_reco_p4[region_wrong_pairs_mask],
                weights=weights[region_wrong_pairs_mask],
                hists=find_hists_by_name(
                    hists,
                    f"hh_(pt|eta|phi|mass)_reco_{region}_{n_btags}btags_{btagger}_{pairing}_wrong_pairs$",
                ),
            )
            lt_370_GeV_mask = (hh_reco_p4.mass < 370_000) & region_mask
            fill_HH_histograms(
                hh=hh_reco_p4[lt_370_GeV_mask],
                weights=weights[lt_370_GeV_mask],
                hists=find_hists_by_name(
                    hists,
                    f"hh_(pt|eta|phi|mass)_reco_{region}_{n_btags}btags_{btagger}_{pairing}_lt_370_GeV$",
                ),
            )
            geq_370_GeV_mask = (hh_reco_p4.mass >= 370_000) & region_mask
            fill_HH_histograms(
                hh=hh_reco_p4[geq_370_GeV_mask],
                weights=weights[geq_370_GeV_mask],
                hists=find_hists_by_name(
                    hists,
                    f"hh_(pt|eta|phi|mass)_reco_{region}_{n_btags}btags_{btagger}_{pairing}_geq_370_GeV$",
                ),
            )


def fill_mHH_plane_histograms(events, hists: list, n_btags: int, btagger: str) -> None:
    jet_p4 = p4.zip(
        {var: events[f"jet_{var}"] for var in kin_labels},
    )
    weights = events.event_weight
    for pairing in pairing_methods:
        h1_jets_idx = events[f"H1_{n_btags}btags_{btagger}_{pairing}_jet_idx"]
        h2_jets_idx = events[f"H2_{n_btags}btags_{btagger}_{pairing}_jet_idx"]
        h1_p4 = ak.sum(jet_p4[h1_jets_idx], axis=1)
        h2_p4 = ak.sum(jet_p4[h2_jets_idx], axis=1)
        hh_reco_p4 = h1_p4 + h2_p4
        valid_event_mask = events[f"valid_{n_btags}btags_{btagger}_events"]
        fill_mHH_2d_histograms(
            mh1=h1_p4.mass[valid_event_mask],
            mh2=h2_p4.mass[valid_event_mask],
            weights=weights[valid_event_mask],
            hist=find_hist(
                hists,
                lambda h_name: f"mHH_plane_reco_{n_btags}btags_{btagger}_{pairing}"
                == h_name,
            ),
        )
        wrong_pairs_mask = (
            events[f"correct_hh_{n_btags}btags_{btagger}_{pairing}_mask"] == False
        ) & valid_event_mask
        fill_mHH_2d_histograms(
            mh1=h1_p4.mass[wrong_pairs_mask],
            mh2=h2_p4.mass[wrong_pairs_mask],
            weights=weights[wrong_pairs_mask],
            hist=find_hist(
                hists,
                lambda h_name: f"mHH_plane_reco_{n_btags}btags_{btagger}_{pairing}_wrong_pairs"
                == h_name,
            ),
        )
        lt_370_GeV_mask = (hh_reco_p4.mass < 370_000) & valid_event_mask
        fill_mHH_2d_histograms(
            mh1=h1_p4.mass[lt_370_GeV_mask],
            mh2=h2_p4.mass[lt_370_GeV_mask],
            weights=weights[lt_370_GeV_mask],
            hist=find_hist(
                hists,
                lambda h_name: f"mHH_plane_reco_{n_btags}btags_{btagger}_{pairing}_lt_370_GeV"
                == h_name,
            ),
        )
        geq_370_GeV_mask = (hh_reco_p4.mass >= 370_000) & valid_event_mask
        fill_mHH_2d_histograms(
            mh1=h1_p4.mass[geq_370_GeV_mask],
            mh2=h2_p4.mass[geq_370_GeV_mask],
            weights=weights[geq_370_GeV_mask],
            hist=find_hist(
                hists,
                lambda h_name: f"mHH_plane_reco_{n_btags}btags_{btagger}_{pairing}_geq_370_GeV"
                == h_name,
            ),
        )
        for region in ["signal", "control"]:
            region_mask = (
                events[f"{region}_{n_btags}btags_{btagger}_{pairing}_mask"]
                & valid_event_mask
            )
            fill_mHH_2d_histograms(
                mh1=h1_p4.mass[region_mask],
                mh2=h2_p4.mass[region_mask],
                weights=weights[region_mask],
                hist=find_hist(
                    hists,
                    lambda h_name: f"mHH_plane_reco_{region}_{n_btags}btags_{btagger}_{pairing}"
                    == h_name,
                ),
            )
            region_wrong_pairs_mask = (
                events[
                    f"{region}_correct_pairs_{n_btags}btags_{btagger}_{pairing}_mask"
                ]
                == False
            ) & valid_event_mask
            fill_mHH_2d_histograms(
                mh1=h1_p4.mass[region_wrong_pairs_mask],
                mh2=h2_p4.mass[region_wrong_pairs_mask],
                weights=weights[region_wrong_pairs_mask],
                hist=find_hist(
                    hists,
                    lambda h_name: f"mHH_plane_reco_{region}_{n_btags}btags_{btagger}_{pairing}_wrong_pairs"
                    == h_name,
                ),
            )
            lt_370_GeV_mask = (hh_reco_p4.mass < 370_000) & region_mask
            fill_mHH_2d_histograms(
                mh1=h1_p4.mass[lt_370_GeV_mask],
                mh2=h2_p4.mass[lt_370_GeV_mask],
                weights=weights[lt_370_GeV_mask],
                hist=find_hist(
                    hists,
                    lambda h_name: f"mHH_plane_reco_{region}_{n_btags}btags_{btagger}_{pairing}_lt_370_GeV"
                    == h_name,
                ),
            )
            geq_370_GeV_mask = (hh_reco_p4.mass >= 370_000) & region_mask
            fill_mHH_2d_histograms(
                mh1=h1_p4.mass[geq_370_GeV_mask],
                mh2=h2_p4.mass[geq_370_GeV_mask],
                weights=weights[geq_370_GeV_mask],
                hist=find_hist(
                    hists,
                    lambda h_name: f"mHH_plane_reco_{region}_{n_btags}btags_{btagger}_{pairing}_geq_370_GeV"
                    == h_name,
                ),
            )


def fill_mHH_2d_histograms(mh1, mh2, weights, hist) -> None:
    """Fill reconstructed H invariant mass 2D histograms"""

    logger.debug(hist.name)
    if ak.count(mh1) != 0 and ak.count(mh2) != 0:
        mHH = np.column_stack((mh1, mh2))
        hist.fill(ak.to_numpy(mHH), weights=ak.to_numpy(weights))


def fill_HH_abs_deltaeta_discrim_histograms(
    events, hists: list, n_btags: int, btagger: str
) -> None:
    """Fill HH deltaeta histograms"""

    for pairing in pairing_methods:
        discrim = abs(events[f"deltaeta_HH_{n_btags}btags_{btagger}_{pairing}_discrim"])
        valid_event = ~ak.is_none(discrim, axis=0)
        hist = find_hist(
            hists,
            lambda h_name: f"hh_abs_deltaeta_discrim_reco_{n_btags}btags_{btagger}_{pairing}"
            == h_name,
        )
        if hist:
            logger.debug(hist.name)
            hist.fill(
                ak.to_numpy(discrim[valid_event]),
                weights=ak.to_numpy(events.event_weight[valid_event]),
            )


def fill_top_veto_discrim_histograms(
    events, hists: list, n_btags: int, btagger: str
) -> None:
    """Fill top veto discriminant histograms"""

    for pairing in pairing_methods:
        discrim = events[f"X_Wt_{n_btags}btags_{btagger}_discrim"]
        valid_event = ~ak.is_none(discrim, axis=0)
        hist = find_hist(
            hists,
            lambda h_name: f"top_veto_discrim_reco_{n_btags}btags_{btagger}_{pairing}"
            == h_name,
        )
        if hist:
            logger.debug(hist.name)
            hist.fill(
                ak.to_numpy(discrim[valid_event]),
                weights=ak.to_numpy(events.event_weight[valid_event]),
            )


def fill_X_HH_histograms(events, hists, n_btags: int, btagger: str) -> None:
    """Fill X_HH histograms"""

    for pairing in pairing_methods:
        discrim = events[f"X_HH_{n_btags}btags_{btagger}_{pairing}_discrim"]
        valid_event = ~ak.is_none(discrim, axis=0)
        valid_event_paired_correct = (
            events[f"correct_hh_{n_btags}btags_{btagger}_{pairing}_mask"] == True
        )
        for cat, mask in zip(
            ["", "_correct_pairs", "_wrong_pairs"],
            [valid_event, valid_event_paired_correct, ~valid_event_paired_correct],
        ):
            hist = find_hist(
                hists,
                lambda h_name: f"hh_mass_discrim_reco_{n_btags}btags_{btagger}_{pairing}{cat}"
                == h_name,
            )
            if hist:
                logger.debug(hist.name)
                hist.fill(
                    ak.to_numpy(discrim[mask]),
                    weights=ak.to_numpy(events.event_weight[mask]),
                )


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


def fill_leading_jets_histograms(
    events,
    histograms: dict,
    jet_type,
    hist_prefix,
    selection_mask=None,
    vars=["pt"],
    n_leading=1,
) -> None:
    """Fill leading jets histograms"""

    jets = ak.zip({v: events[f"{jet_type}_{v}"] for v in vars})
    weights = events.event_weight
    if selection_mask is not None:
        jets = jets[selection_mask]
        weights = weights[selection_mask]
    pt_sort = ak.argsort(jets.pt, axis=1, ascending=False)
    jets_sorted = jets[pt_sort]

    for v in vars:
        for i in range(1, n_leading + 1):
            hist = histograms[f"{hist_prefix}_{i}_{v}"]
            logger.debug(f"Filling histogram {hist.name}")
            leading_jet = jets_sorted[v][:, :i]
            valid_event = ak.num(leading_jet) == 1
            leading_jet = leading_jet[valid_event]
            hist.fill(
                np.ravel(jets_sorted[v][:, :i]).to_numpy(),
                weights=weights[valid_event].to_numpy(),
            )


def fill_n_true_bjet_composition_histograms(
    events, histograms: dict, weights=None
) -> None:
    """Fill number of true b-jets composition histograms"""

    bjets_mask = events.jet_truth_label_ID == 5
    events_n_true_bjets = ak.sum(bjets_mask, axis=1)

    hist = histograms["events_geq_n_true_bjets"]
    logger.debug(f"Filling histogram {hist.name}")
    hist.fill(events_n_true_bjets.to_numpy(), weights=weights)


def fill_jet_flavor_composition_histograms(
    events, histograms: dict, n_btags: int, btag_model: str, btag_eff: float
) -> None:
    """Fill jet flavor composition histograms"""

    jet_flavs = ["u", "c", "b", "tau"]
    jet_flav_ids = [0, 4, 5, 6]

    discrim = lambda jet, fc=0.018, ftau=0: np.log(
        jet.pb / (fc * jet.pc + ftau * jet.ptau + (1 - fc - ftau) * jet.pu)
    )

    btagger = format_btagger_model_name(
        btag_model,
        btag_eff,
    )

    jet_flavor = events.jet_truth_label_ID
    jet_probs = ak.zip(
        {f"p{flav}": events[f"jet_btag_{btag_model}_p{flav}"] for flav in jet_flavs}
    )
    weights = events.event_weight
    for pairing in pairing_methods:
        hist = histograms[f"jet_flavor_signal_{n_btags}btags_{btagger}_{pairing}"]
        signal_event_mask = events[f"signal_{n_btags}btags_{btagger}_{pairing}_mask"]
        hh_jets_idx = events[f"hh_{n_btags}btags_{btagger}_jet_idx"]
        hh_jets_flavor_signal = jet_flavor[hh_jets_idx][signal_event_mask]
        wgts = ak.ones_like(hh_jets_flavor_signal) * weights[signal_event_mask]
        if hist:
            logger.debug(f"Filling histogram {hist.name}")
            hist.fill(
                ak.ravel(hh_jets_flavor_signal).to_numpy(),
                weights=ak.ravel(wgts).to_numpy(),
            )

        jet_probs_signal = jet_probs[hh_jets_idx][signal_event_mask]
        jet_discrim_signal = discrim(jet_probs_signal)
        for flav, flav_id in zip(jet_flavs, jet_flav_ids):
            hist = histograms[
                f"bjet_discrim_{flav}_signal_{n_btags}btags_{btagger}_{pairing}"
            ]
            if hist:
                flav_mask = hh_jets_flavor_signal == flav_id
                logger.debug(f"Filling histogram {hist.name}")
                hist.fill(
                    np.ravel(jet_discrim_signal[flav_mask]).to_numpy(),
                    weights=ak.ravel(wgts[flav_mask]).to_numpy(),
                )


def fill_HH_combined_pairing_histograms(
    events, hists: dict, n_btags: int, btagger: str
) -> None:
    """Fill HH combined pairing histograms"""

    jet_p4 = p4.zip(
        {var: events[f"jet_{var}"] for var in kin_labels},
    )
    weights = events.event_weight
    valid_event_mask = events[f"valid_{n_btags}btags_{btagger}_events"]

    h1_jets_idx_min_deltar = events[
        f"H1_{n_btags}btags_{btagger}_min_deltar_pairing_jet_idx"
    ]
    h2_jets_idx_min_deltar = events[
        f"H2_{n_btags}btags_{btagger}_min_deltar_pairing_jet_idx"
    ]
    h1_p4_min_deltar = ak.sum(jet_p4[h1_jets_idx_min_deltar], axis=1)
    h2_p4_min_deltar = ak.sum(jet_p4[h2_jets_idx_min_deltar], axis=1)
    hh_reco_p4_min_deltar = h1_p4_min_deltar + h2_p4_min_deltar
    h1_jets_idx_min_mass_optimized_1D_medium_pairing = events[
        f"H1_{n_btags}btags_{btagger}_min_mass_optimized_1D_medium_pairing_jet_idx"
    ]
    h2_jets_idx_min_mass_optimized_1D_medium_pairing = events[
        f"H2_{n_btags}btags_{btagger}_min_mass_optimized_1D_medium_pairing_jet_idx"
    ]
    h1_p4_min_mass_optimized_1D_medium_pairing = ak.sum(
        jet_p4[h1_jets_idx_min_mass_optimized_1D_medium_pairing], axis=1
    )
    h2_p4_min_mass_optimized_1D_medium_pairing = ak.sum(
        jet_p4[h2_jets_idx_min_mass_optimized_1D_medium_pairing], axis=1
    )
    hh_reco_p4_min_mass_optimized_1D_medium_pairing = (
        h1_p4_min_mass_optimized_1D_medium_pairing
        + h2_p4_min_mass_optimized_1D_medium_pairing
    )
    hh_jet_idx = events[f"hh_{n_btags}btags_{btagger}_jet_idx"]
    hh_p4 = ak.sum(jet_p4[hh_jet_idx], axis=1)
    low_mass_mask = (hh_p4.mass < 370_000) & valid_event_mask

    # create variable called hh_p4_combined_pairing where the pairing is done with the minimum deltaR for hh_p4.mass < 370 GeV and
    # minimum mass optimized 1D medium pairing for hh_p4.mass >= 370 GeV
    h1_p4_combined_pairing = ak.where(
        low_mass_mask,
        h1_p4_min_mass_optimized_1D_medium_pairing,
        h1_p4_min_deltar,
    )
    h2_p4_combined_pairing = ak.where(
        low_mass_mask,
        h2_p4_min_mass_optimized_1D_medium_pairing,
        h2_p4_min_deltar,
    )
    hh_p4_combined_pairing = h1_p4_combined_pairing + h2_p4_combined_pairing
    weights_combined_pairing = ak.where(
        low_mass_mask,
        weights,
        weights,
    )
    fill_H_histograms(
        h1=h1_p4_combined_pairing,
        h2=h2_p4_combined_pairing,
        weights=weights_combined_pairing,
        hists=find_hists_by_name(
            hists,
            f"h[12]_(pt|eta|phi|mass)_reco_{n_btags}btags_{btagger}_combined_pairing$",
        ),
    )
    fill_HH_histograms(
        hh=hh_p4_combined_pairing,
        weights=weights_combined_pairing,
        hists=find_hists_by_name(
            hists,
            "hh_(pt|eta|phi|mass)_reco_{n_btags}btags_{btagger}_combined_pairing$",
        ),
    )
    fill_mHH_2d_histograms(
        mh1=h1_p4_combined_pairing.mass,
        mh2=h2_p4_combined_pairing.mass,
        weights=weights_combined_pairing,
        hist=find_hist(
            hists,
            lambda h_name: f"mHH_plane_reco_{n_btags}btags_{btagger}_combined_pairing"
            == h_name,
        ),
    )
