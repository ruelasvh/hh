import logging
import numpy as np
import vector as p4
import awkward as ak
from hh.shared.utils import (
    logger,
    format_btagger_model_name,
    GeV,
)
from hh.shared.labels import kin_labels
from hh.shared.selection import (
    X_HH,
    R_CR,
    X_Wt,
    get_W_t_p4,
    classify_control_events,
)
from hh.nonresonantresolved.pairing import pairing_methods
from hh.nonresonantresolved.selection import (
    select_n_jets_events,
    select_n_bjets_events,
    select_events_passing_triggers,
    select_truth_matched_jets,
    select_hh_jet_candidates,
    reconstruct_hh_jet_pairs,
    select_correct_hh_pair_events,
    select_discrim_events,
    select_vbf_events,
)


def process_batch(
    events: ak.Record,
    selections: dict,
    is_mc: bool = False,
) -> ak.Record:
    """Apply analysis regions selections and append info to events."""

    logger.info(
        "Initial Events: %s (weighted: %s)", len(events), ak.sum(events["event_weight"])
    )

    events_4_more_true_bjets = ak.sum(events.jet_truth_label_ID == 5, axis=1) > 4
    logger.debug(
        "Events with 4 or more true b-jets: %s (weighted: %s)",
        ak.sum(events_4_more_true_bjets),
        ak.sum(events.event_weight[events_4_more_true_bjets]),
    )

    ## return early if selections is empty ##
    if not selections:
        logger.info("Selections empty. No object selections applied.")
        return events

    ## apply trigger selection ##
    if "trigs" in selections:
        trig_sel = selections["trigs"]
        trig_set, trig_op = (
            trig_sel.get("value"),
            trig_sel.get("operator"),
        )
        assert trig_set, "Invalid trigger set provided."
        passed_trigs = select_events_passing_triggers(events, operation=trig_op)
        events = events[passed_trigs]
        logger.info(
            "Events passing the %s of all triggers: %s (weighted: %s)",
            trig_op.upper() if trig_op is not None else "None",
            ak.sum(passed_trigs),
            ak.sum(events.event_weight),
        )
        if ak.sum(passed_trigs) == 0:
            return None

    ## apply jet selections ##
    if "jets" in selections:
        jets_sel = selections["jets"]
        ## set overall event filter ##
        valid_events_mask = np.ones(len(events), dtype=bool)
        ## apply jet b-tagging selections ##
        if "btagging" in jets_sel:
            bjets_sel = jets_sel["btagging"]
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
                valid_jets_mask = select_n_jets_events(
                    jets=ak.zip(
                        {k: events[f"jet_{k}"] for k in ["jvttag", *kin_labels.keys()]}
                    ),
                    selection=jets_sel,
                    do_jvt=True,
                )
                ## apply 2 b-tag pre-selection ##
                valid_jets_mask = select_n_bjets_events(
                    jets=valid_jets_mask,
                    btags=events[f"jet_btag_{btagger}"][valid_jets_mask],
                    selection={**i_bjets_sel, "count": {"operator": ">=", "value": 2}},
                )
                valid_events_mask = (
                    valid_events_mask & ~ak.is_none(valid_jets_mask).to_numpy()
                )
                events[f"valid_2btags_{btagger}_events"] = valid_events_mask
                logger.info(
                    "Events passing previous cut and %s %s jets with pT %s %s, |eta| %s %s and 2 b-tags: %s (weighted: %s)",
                    jets_sel["count"]["operator"],
                    jets_sel["count"]["value"],
                    jets_sel["pt"]["operator"],
                    jets_sel["pt"]["value"],
                    jets_sel["eta"]["operator"],
                    jets_sel["eta"]["value"],
                    ak.sum(valid_events_mask),
                    ak.sum(events.event_weight[valid_events_mask]),
                )

                if ak.sum(valid_events_mask) == 0:
                    return None

                if is_mc:
                    events["reco_truth_matched_jets"] = select_truth_matched_jets(
                        events.jet_truth_H_parent_mask != 0,
                        # (events.jet_truth_H_parent_mask == 1)
                        # | (events.jet_truth_H_parent_mask == 2),
                        valid_jets_mask,
                    )
                    ### jets truth matched with HadronConeExclTruthLabelID ###
                    events["reco_truth_matched_jets_v2"] = select_truth_matched_jets(
                        events.jet_truth_label_ID == 5, valid_jets_mask
                    )
                    logger.debug(
                        "Events passing previous cuts and 4 truth-matched jets: %s (weighted: %s)",
                        ak.sum(~ak.is_none(events.reco_truth_matched_jets)),
                        ak.sum(
                            events.event_weight[
                                ~ak.is_none(events.reco_truth_matched_jets)
                            ]
                        ),
                    )
                    logger.debug(
                        "Events passing previous cuts and 4 truth-matched jets using HadronConeExclTruthLabelID: %s (weighted: %s)",
                        ak.sum(~ak.is_none(events.reco_truth_matched_jets_v2)),
                        ak.sum(
                            events.event_weight[
                                ~ak.is_none(events.reco_truth_matched_jets_v2)
                            ]
                        ),
                    )

                ## select and save events with >= n central b-jets ##
                valid_btagged_jets = select_n_bjets_events(
                    jets=valid_jets_mask,
                    btags=events[f"jet_btag_{btagger}"][valid_jets_mask],
                    selection=i_bjets_sel,
                )
                events[f"valid_{n_btags}btags_{btagger}_jets"] = valid_btagged_jets
                valid_events_mask = (
                    valid_events_mask & ~ak.is_none(valid_btagged_jets).to_numpy()
                )
                events[f"valid_{n_btags}btags_{btagger}_events"] = valid_events_mask
                logger.info(
                    "Events passing previous cut and %s %s b-tags with %s and %s efficiency: %s (weighted: %s)",
                    i_bjets_sel["count"]["operator"],
                    i_bjets_sel["count"]["value"],
                    i_bjets_sel["model"],
                    i_bjets_sel["efficiency"],
                    ak.sum(valid_events_mask),
                    ak.sum(events.event_weight[valid_events_mask]),
                )
                if ak.sum(valid_events_mask) == 0:
                    return None

                ### Do truth matching with b-tagging requirement ###
                if is_mc:
                    reco_truth_matched_btagged_jets = select_truth_matched_jets(
                        events.jet_truth_H_parent_mask != 0,
                        # (events.jet_truth_H_parent_mask == 1)
                        # | (events.jet_truth_H_parent_mask == 2),
                        valid_btagged_jets,
                    )
                    events[f"reco_truth_matched_{n_btags}btags_{btagger}_jets"] = (
                        reco_truth_matched_btagged_jets
                    )
                    ### jets truth matched with HadronConeExclTruthLabelID ###
                    reco_truth_matched_btagged_jets_v2 = select_truth_matched_jets(
                        events.jet_truth_label_ID == 5,
                        valid_btagged_jets,
                    )
                    events[
                        f"reco_truth_matched_central_{n_btags}btags_{btagger}_jets_v2"
                    ] = reco_truth_matched_btagged_jets_v2
                    logger.debug(
                        "Events passing previous cuts and 4 truth-matched b-tagged jets with %s %s btags: %s (weighted: %s)",
                        i_bjets_sel["count"]["operator"],
                        i_bjets_sel["count"]["value"],
                        ak.sum(~ak.is_none(reco_truth_matched_btagged_jets)),
                        ak.sum(
                            events.event_weight[
                                ~ak.is_none(reco_truth_matched_btagged_jets)
                            ]
                        ),
                    )
                    logger.debug(
                        "Events passing previous cuts and 4 truth-matched b-tagged jets using HadronConeExclTruthLabelID: %s (weighted: %s)",
                        ak.sum(~ak.is_none(reco_truth_matched_btagged_jets_v2)),
                        ak.sum(
                            events.event_weight[
                                ~ak.is_none(reco_truth_matched_btagged_jets_v2)
                            ]
                        ),
                    )

                ###################################
                # select and save HH jet candidates
                ###################################
                jets = p4.zip(
                    {
                        "btag": events[f"jet_btag_{btagger}"],
                        **{k: events[f"jet_{k}"] for k in kin_labels},
                    }
                )
                ###################################
                # Do VBF selection
                ###################################
                selections["VBF"] = {
                    "pt": {"operator": ">", "value": 30},
                    "eta_min": {"operator": ">", "value": 2.5},
                    "eta_max": {"operator": "<", "value": 4.5},
                    "m_jj": {"operator": ">", "value": 1_000},
                    "delta_eta_jj": {"operator": ">", "value": 3},
                    "pT_vector_sum": {"operator": "<", "value": 65},
                }
                if "VBF" in selections:
                    vbf_sel = selections["VBF"]
                    passed_vbf, vbf_jets = select_vbf_events(
                        jets=jets,
                        valid_central_jets_mask=valid_btagged_jets,
                        selection=vbf_sel,
                    )
                    events["passed_vbf"] = passed_vbf
                    logger.info(
                        "Events passing previous cuts and VBF selection: %s (weighted: %s)",
                        ak.sum(valid_events_mask & passed_vbf),
                        ak.sum(events.event_weight[valid_events_mask & passed_vbf]),
                    )

                hh_jet_idx, non_hh_jet_idx = select_hh_jet_candidates(
                    jets=jets,
                    valid_jets_mask=valid_btagged_jets,
                )
                events[f"hh_{n_btags}btags_{btagger}_jet_idx"] = hh_jet_idx
                events[f"non_hh_{n_btags}btags_{btagger}_jet_idx"] = non_hh_jet_idx
                ## select and save HH jet candidates that are truth-matched ##
                if is_mc:
                    hh_truth_matched_jet_idx, non_hh_truth_matched_jet_idx = (
                        select_hh_jet_candidates(
                            jets=jets,
                            valid_jets_mask=reco_truth_matched_btagged_jets,
                        )
                    )
                    events[f"hh_{n_btags}btags_{btagger}_truth_matched_jet_idx"] = (
                        hh_truth_matched_jet_idx
                    )
                    events[f"non_hh_{n_btags}btags_{btagger}_truth_matched_jet_idx"] = (
                        non_hh_truth_matched_jet_idx
                    )

                ############################################
                ## Calculate top veto X_Wt
                ############################################
                if "X_Wt_discriminant" in selections:
                    top_veto_sel = selections["X_Wt_discriminant"]
                    # reconstruct W and top candidates
                    W_candidates_p4, top_candidates_p4 = get_W_t_p4(
                        jets,
                        hh_jet_idx,
                        non_hh_jet_idx,
                    )
                    # calculate X_Wt discriminant
                    X_Wt_discrim = X_Wt(
                        W_candidates_p4.mass * GeV,
                        top_candidates_p4.mass * GeV,
                    )
                    # select only the minimum X_Wt for each event
                    X_Wt_discrim = ak.min(X_Wt_discrim, axis=1)
                    events[f"X_Wt_{n_btags}btags_{btagger}_discrim"] = X_Wt_discrim
                    X_Wt_mask = select_discrim_events(
                        (X_Wt_discrim,), selection=top_veto_sel
                    )
                    events[f"X_Wt_{n_btags}btags_{btagger}_mask"] = X_Wt_mask

                ############################################
                # Apply different HH jet candidate pairings
                ############################################
                ###### HH jet nominal pairing ######
                if is_mc:
                    correct_hh_nominal_pairs_mask = select_correct_hh_pair_events(
                        h1_jets_idx=hh_truth_matched_jet_idx[:, 0:2],
                        h2_jets_idx=hh_truth_matched_jet_idx[:, 2:4],
                        jet_truth_H_parent_mask=events.jet_truth_H_parent_mask,
                    )
                    events[
                        f"correct_hh_{n_btags}btags_{btagger}_nominal_pairs_mask"
                    ] = correct_hh_nominal_pairs_mask
                    logger.debug(
                        "Events passing previous cuts and correct H1 and H2 jet pairs using nominal pairing: %s (weighted: %s)",
                        ak.sum(correct_hh_nominal_pairs_mask),
                        ak.sum(events.event_weight[correct_hh_nominal_pairs_mask]),
                    )

                for pairing, pairing_info in pairing_methods.items():
                    ## NEED TO MASK THE EVENTS WITH THE X_Wt MASK ##
                    valid_events_pairing_mask = np.copy(valid_events_mask)
                    ###### reconstruct H1 and H2 ######
                    H1_jet_idx, H2_jet_idx = reconstruct_hh_jet_pairs(
                        jets,
                        hh_jet_idx=hh_jet_idx,
                        loss=pairing_info["loss"],
                        optimizer=pairing_info["optimizer"],
                    )
                    events[f"H1_{n_btags}btags_{btagger}_{pairing}_jet_idx"] = (
                        H1_jet_idx
                    )
                    events[f"H2_{n_btags}btags_{btagger}_{pairing}_jet_idx"] = (
                        H2_jet_idx
                    )
                    if is_mc:
                        ###### Truth match the H1 and H2 ######
                        H1_truth_matched_jet_idx, H2_truth_matched_jet_idx = (
                            reconstruct_hh_jet_pairs(
                                jets,
                                hh_jet_idx=hh_truth_matched_jet_idx,
                                loss=pairing_info["loss"],
                                optimizer=pairing_info["optimizer"],
                            )
                        )
                        events[
                            f"H1_{n_btags}btags_{btagger}_{pairing}_truth_matched_jet_idx"
                        ] = H1_truth_matched_jet_idx
                        events[
                            f"H2_{n_btags}btags_{btagger}_{pairing}_truth_matched_jet_idx"
                        ] = H2_truth_matched_jet_idx

                        correct_hh_pairs_mask = select_correct_hh_pair_events(
                            h1_jets_idx=H1_truth_matched_jet_idx,
                            h2_jets_idx=H2_truth_matched_jet_idx,
                            jet_truth_H_parent_mask=events.jet_truth_H_parent_mask,
                        )
                        events[
                            f"correct_hh_{n_btags}btags_{btagger}_{pairing}_mask"
                        ] = correct_hh_pairs_mask
                        logger.debug(
                            "Events passing previous cuts and correct H1 and H2 jet pairs using %s pairing: %s (weighted: %s)",
                            pairing.replace("_", " "),
                            ak.sum(correct_hh_pairs_mask),
                            ak.sum(events.event_weight[correct_hh_pairs_mask]),
                        )

                    #################################################
                    # Perform event selection for each pairing method
                    #################################################
                    h1_p4 = ak.sum(jets[H1_jet_idx], axis=1)
                    h2_p4 = ak.sum(jets[H2_jet_idx], axis=1)
                    ###### Delta eta ∆𝜂 HH veto ######
                    if "Delta_eta_HH_discriminant" in selections:
                        deltaeta_HH_discrim_sel = selections[
                            "Delta_eta_HH_discriminant"
                        ]
                        deltaeta_HH = np.abs(h1_p4.eta - h2_p4.eta)
                        events[
                            f"deltaeta_HH_{n_btags}btags_{btagger}_{pairing}_discrim"
                        ] = deltaeta_HH
                        Delta_eta_HH_mask = select_discrim_events(
                            (deltaeta_HH,),
                            selection=deltaeta_HH_discrim_sel,
                        )
                        events[
                            f"deltaeta_HH_{n_btags}btags_{btagger}_{pairing}_mask"
                        ] = Delta_eta_HH_mask
                        valid_events_pairing_mask = (
                            valid_events_pairing_mask & Delta_eta_HH_mask
                        )
                        logger.info(
                            "Events passing previous cuts and ∆𝜂 HH %s %s using %s pairing: %s (weighted: %s)",
                            deltaeta_HH_discrim_sel["operator"],
                            deltaeta_HH_discrim_sel["value"],
                            pairing.replace("_", " "),
                            ak.sum(valid_events_pairing_mask),
                            ak.sum(events.event_weight[valid_events_pairing_mask]),
                        )
                        if ak.sum(valid_events_pairing_mask) == 0:
                            return None

                    ###### Apply X_Wt cut ######
                    if "X_Wt_discriminant" in selections:
                        valid_events_pairing_mask = (
                            valid_events_pairing_mask & X_Wt_mask
                        )
                        logger.info(
                            "Events passing previous cuts and X_Wt %s %s using %s pairing: %s (weighted: %s)",
                            top_veto_sel["operator"],
                            top_veto_sel["value"],
                            pairing.replace("_", " "),
                            ak.sum(valid_events_pairing_mask),
                            ak.sum(events.event_weight[valid_events_pairing_mask]),
                        )
                        if ak.sum(valid_events_pairing_mask) == 0:
                            return None

                    ###### X_HH mass veto ######
                    if "X_HH_discriminant" in selections:
                        hh_mass_discrim_sel = selections["X_HH_discriminant"]
                        X_HH_discrim = X_HH(h1_p4.m * GeV, h2_p4.m * GeV)
                        events[f"X_HH_{n_btags}btags_{btagger}_{pairing}_discrim"] = (
                            X_HH_discrim
                        )
                        R_CR_discrim = R_CR(h1_p4.m * GeV, h2_p4.m * GeV)
                        events[f"R_CR_{n_btags}btags_{btagger}_{pairing}_discrim"] = (
                            R_CR_discrim
                        )
                        regions = ["signal", "control"]
                        for region in regions:
                            if region in hh_mass_discrim_sel:
                                region_mask = select_discrim_events(
                                    (X_HH_discrim, R_CR_discrim),
                                    selection=hh_mass_discrim_sel[region],
                                )
                                events[
                                    f"X_HH_{region}_{n_btags}btags_{btagger}_{pairing}_mask"
                                ] = region_mask
                                logger.info(
                                    "Events passing previous cuts and X_HH veto for %s region using %s pairing: %s (weighted: %s)",
                                    region,
                                    pairing.replace("_", " "),
                                    ak.sum(region_mask),
                                    ak.sum(events.event_weight[region_mask]),
                                )

                    signal_event_mask = np.zeros(len(events), dtype=bool)
                    control_event_mask = np.zeros(len(events), dtype=bool)
                    if "X_Wt_discriminant" in selections:
                        signal_event_mask = events[
                            f"X_Wt_{n_btags}btags_{btagger}_mask"
                        ]
                    if "Delta_eta_HH_discriminant" in selections:
                        signal_event_mask = (
                            signal_event_mask
                            & events[
                                f"deltaeta_HH_{n_btags}btags_{btagger}_{pairing}_mask"
                            ]
                        )
                    if "X_HH_discriminant" in selections:
                        # needs to go before re-assigment of signal_event_mask to ensure the final signal region mask not applied
                        control_event_mask = (
                            signal_event_mask
                            & events[
                                f"X_HH_control_{n_btags}btags_{btagger}_{pairing}_mask"
                            ]
                        )
                        signal_event_mask = (
                            signal_event_mask
                            & events[
                                f"X_HH_signal_{n_btags}btags_{btagger}_{pairing}_mask"
                            ]
                        )
                    events[f"signal_{n_btags}btags_{btagger}_{pairing}_mask"] = (
                        signal_event_mask
                    )
                    events[f"control_{n_btags}btags_{btagger}_{pairing}_mask"] = (
                        control_event_mask
                    )
                    logger.info(
                        "Events passing previous cuts and signal region using %s pairing: %s (weighted: %s)",
                        pairing.replace("_", " "),
                        ak.sum(signal_event_mask),
                        ak.sum(events.event_weight[signal_event_mask]),
                    )
                    logger.info(
                        "Events passing previous cuts and control region using %s pairing: %s (weighted: %s)",
                        pairing.replace("_", " "),
                        ak.sum(control_event_mask),
                        ak.sum(events.event_weight[control_event_mask]),
                    )
                    signal_event_correct_pairs_mask = ak.mask(
                        correct_hh_pairs_mask, signal_event_mask
                    )
                    events[
                        f"signal_correct_pairs_{n_btags}btags_{btagger}_{pairing}_mask"
                    ] = signal_event_correct_pairs_mask
                    logger.debug(
                        "Events passing previous cuts and signal region with wrong pairs using %s pairing: %s (weighted: %s)",
                        pairing.replace("_", " "),
                        ak.sum(~signal_event_correct_pairs_mask),
                        ak.sum(events.event_weight[~signal_event_correct_pairs_mask]),
                    )
                    control_event_correct_pairs_mask = ak.mask(
                        correct_hh_pairs_mask, control_event_mask
                    )
                    events[
                        f"control_correct_pairs_{n_btags}btags_{btagger}_{pairing}_mask"
                    ] = control_event_correct_pairs_mask
                    logger.debug(
                        "Events passing previous cuts and control region with wrong pairs using %s pairing: %s (weighted: %s)",
                        pairing.replace("_", " "),
                        ak.sum(~control_event_correct_pairs_mask),
                        ak.sum(events.event_weight[~control_event_correct_pairs_mask]),
                    )

    return events
