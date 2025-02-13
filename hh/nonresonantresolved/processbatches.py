import vector
import numpy as np
import awkward as ak
import onnxruntime as ort
from hh.shared.utils import (
    GeV,
    logger,
    get_op,
    format_btagger_model_name,
)
from hh.shared.labels import kin_labels
from hh.shared.selection import (
    X_HH,
    R_CR,
    X_Wt,
    get_W_t_p4,
)
from hh.nonresonantresolved.pairing import pairing_methods as all_pairing_methods
from hh.nonresonantresolved.selection import (
    select_events_passing_triggers,
    select_n_jets_events,
    select_n_bjets_events,
    select_truth_matched_jets,
    select_hh_jet_candidates,
    reconstruct_hh_jet_pairs,
    select_correct_hh_pair_events,
    select_discrim_events,
    select_vbf_events,
)
from hh.shared.clahh_utils import get_inferences, get_deepset_inputs
from hh.nonresonantresolved.branches import get_trigger_branch_aliases

vector.register_awkward()


def process_batch(
    events: ak.Record,
    selections: dict,
    is_mc: bool = False,
    year: int = None,
    clahh_model_ort: ort.InferenceSession = None,
) -> ak.Record:
    """Apply analysis regions selections and append info to events."""

    logger.info(
        "Analysis initial events: %s (weighted: %s)",
        len(events),
        ak.sum(events.event_weight),
    )

    events_4_more_true_bjets = ak.sum(events.jet_truth_label_ID == 5, axis=1) > 4
    logger.debug(
        "Analysis events with 4 or more true b-jets: %s (weighted: %s)",
        ak.sum(events_4_more_true_bjets),
        ak.sum(events.event_weight[events_4_more_true_bjets]),
    )

    ## return early if selections is empty ##
    if not selections:
        logger.info("Analysis selections empty. No object selections applied.")
        return events

    ## create a mask to keep track of valid events
    valid_events_mask = np.ones(len(events), dtype=bool)

    ## apply trigger selection ##
    if "trigs" in selections:
        trig_sel = selections["trigs"]
        trig_set, trig_op = (
            trig_sel.get("value"),
            trig_sel.get("operator"),
        )
        assert trig_set, "Invalid trigger set provided."
        triggers = get_trigger_branch_aliases(trig_set, year=year)
        passed_trigs = select_events_passing_triggers(
            events, triggers=triggers.keys(), operator=trig_op
        )
        events[f"passed_{trig_op}_trigs"] = passed_trigs
        valid_events_mask = passed_trigs
        logger.info(
            "Analysis events passing the %s of all triggers: %s (weighted: %s)",
            trig_op.upper() if trig_op is not None else "None",
            ak.sum(passed_trigs),
            ak.sum(events.event_weight[passed_trigs]),
        )
        if ak.sum(passed_trigs) == 0:
            return None

    ## apply jet selections ##
    if "jets" in selections:
        jets_sel = selections["jets"]
        ## apply jet b-tagging selections ##
        if "btagging" in jets_sel:
            bjets_sel = jets_sel["btagging"]
            if isinstance(bjets_sel, dict):
                bjets_sel = [bjets_sel]
            for i_bjets_sel in bjets_sel:
                valid_events_mask = np.copy(valid_events_mask)
                btag_model = i_bjets_sel["model"]
                btag_eff = i_bjets_sel["efficiency"]
                n_btags = i_bjets_sel["count"]["value"]
                btagger = format_btagger_model_name(
                    btag_model,
                    btag_eff,
                )
                jets_p4 = ak.zip(
                    {
                        **{k: events[f"jet_{k}"] for k in kin_labels},
                        "btag": events[f"jet_btag_{btagger}"],
                        "jvttag": events["jet_jvttag"],
                    },
                    with_name="Momentum4D",
                )
                valid_jets_mask = select_n_jets_events(
                    jets=jets_p4.mask[valid_events_mask],
                    selection=jets_sel,
                    do_jvt=True,
                )
                ## apply 2 b-tag pre-selection ##
                valid_jets_2bjets_mask = select_n_bjets_events(
                    jets=valid_jets_mask,
                    btags=jets_p4.btag.mask[valid_jets_mask],
                    selection={**i_bjets_sel, "count": {"operator": ">=", "value": 2}},
                )
                events[f"valid_2btags_{btagger}_jets"] = valid_jets_2bjets_mask
                valid_events_mask = ~ak.is_none(valid_jets_2bjets_mask).to_numpy()
                logger.info(
                    "Analysis events passing previous cut and %s %s jets with pT %s %s, |eta| %s %s and 2 b-tags: %s (weighted: %s)",
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
                    reco_truth_matched_jets = select_truth_matched_jets(
                        events.jet_truth_H_parent_mask != 0,
                        valid_jets_2bjets_mask,
                    )
                    events["reco_truth_matched_jets"] = reco_truth_matched_jets
                    logger.debug(
                        "Analysis events passing previous cuts and 4 truth-matched jets: %s (weighted: %s)",
                        ak.sum(~ak.is_none(reco_truth_matched_jets)),
                        ak.sum(
                            events.event_weight[~ak.is_none(reco_truth_matched_jets)]
                        ),
                    )
                    ### jets truth matched with HadronConeExclTruthLabelID ###
                    reco_truth_matched_jets_v2 = select_truth_matched_jets(
                        events.jet_truth_label_ID == 5, valid_jets_2bjets_mask
                    )
                    events["reco_truth_matched_jets_v2"] = reco_truth_matched_jets_v2
                    logger.debug(
                        "Analysis events passing previous cuts and 4 truth-matched jets using HadronConeExclTruthLabelID: %s (weighted: %s)",
                        ak.sum(~ak.is_none(reco_truth_matched_jets_v2)),
                        ak.sum(
                            events.event_weight[~ak.is_none(reco_truth_matched_jets_v2)]
                        ),
                    )

                ## select and save events with >= n central b-jets ##
                valid_btagged_jets = select_n_bjets_events(
                    jets=valid_jets_2bjets_mask,
                    btags=jets_p4.btag.mask[valid_jets_2bjets_mask],
                    selection=i_bjets_sel,
                )
                events[f"valid_{n_btags}btags_{btagger}_jets"] = valid_btagged_jets
                valid_events_mask = ~ak.is_none(valid_btagged_jets).to_numpy()
                logger.info(
                    "Analysis events passing previous cut and %s %s b-tags with %s and %s efficiency: %s (weighted: %s)",
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
                        valid_btagged_jets,
                    )
                    events[f"reco_truth_matched_{n_btags}btags_{btagger}_jets"] = (
                        reco_truth_matched_btagged_jets
                    )
                    logger.debug(
                        "Analysis events passing previous cuts and 4 truth-matched b-tagged jets with %s %s btags: %s (weighted: %s)",
                        i_bjets_sel["count"]["operator"],
                        i_bjets_sel["count"]["value"],
                        ak.sum(~ak.is_none(reco_truth_matched_btagged_jets)),
                        ak.sum(
                            events.event_weight[
                                ~ak.is_none(reco_truth_matched_btagged_jets)
                            ]
                        ),
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
                        "Analysis events passing previous cuts and 4 truth-matched b-tagged jets using HadronConeExclTruthLabelID: %s (weighted: %s)",
                        ak.sum(~ak.is_none(reco_truth_matched_btagged_jets_v2)),
                        ak.sum(
                            events.event_weight[
                                ~ak.is_none(reco_truth_matched_btagged_jets_v2)
                            ]
                        ),
                    )

                ###################################
                # Do VBF selection
                ###################################
                if "VBF" in selections:
                    vbf_sel = selections["VBF"]
                    passed_vbf, vbf_jets = select_vbf_events(
                        jets=jets_p4,
                        valid_central_jets_mask=valid_btagged_jets,
                        selection=vbf_sel,
                    )
                    events["passed_vbf"] = passed_vbf
                    logger.info(
                        "Analysis events passing previous cuts and VBF selection: %s (weighted: %s)",
                        ak.sum(valid_events_mask & passed_vbf),
                        ak.sum(events.event_weight[valid_events_mask & passed_vbf]),
                    )

                ###################################################
                ######## Select and save HH jet candidates ########
                ###################################################
                hh_jet_idx, non_hh_jet_idx = select_hh_jet_candidates(
                    jets=jets_p4,
                    valid_jets_mask=valid_btagged_jets,
                )
                events[f"hh_{n_btags}btags_{btagger}_jet_idx"] = hh_jet_idx
                events[f"non_hh_{n_btags}btags_{btagger}_jet_idx"] = non_hh_jet_idx

                ## select and save HH jet candidates that are truth-matched ##
                if is_mc:
                    hh_truth_matched_jet_idx, non_hh_truth_matched_jet_idx = (
                        select_hh_jet_candidates(
                            jets=jets_p4,
                            valid_jets_mask=reco_truth_matched_btagged_jets,
                        )
                    )
                    events[f"hh_{n_btags}btags_{btagger}_truth_matched_jet_idx"] = (
                        hh_truth_matched_jet_idx
                    )
                    events[f"non_hh_{n_btags}btags_{btagger}_truth_matched_jet_idx"] = (
                        non_hh_truth_matched_jet_idx
                    )

                ####################################
                ######## CLAHH discriminant ########
                ####################################
                if "CLAHH_discriminant" in selections and clahh_model_ort:
                    clahh_discrim_sel = selections["CLAHH_discriminant"]
                    dataset = events[[f"jet_{k}" for k in kin_labels]]
                    dataset["jet_btag"] = events[f"jet_btag_{btagger}"]
                    dataset["hh_jet_idx"] = hh_jet_idx
                    input_jet, input_event = get_deepset_inputs(
                        dataset=dataset,
                        max_jets=20,
                    )
                    clahh_discrim = get_inferences(
                        ort_session=clahh_model_ort,
                        input_jet=input_jet,
                        input_event=input_event,
                    )
                    events[f"clahh_{n_btags}btags_{btagger}_discrim"] = clahh_discrim
                    clahh_predictions = np.where(
                        get_op(clahh_discrim_sel["operator"])(
                            clahh_discrim, clahh_discrim_sel["value"]
                        ),
                        1,
                        0,
                    )
                    clahh_wp_str = "{:.0f}".format(
                        clahh_discrim_sel["working_point"] * 100
                    )
                    events[f"clahh{clahh_wp_str}_{n_btags}btags_{btagger}"] = (
                        clahh_predictions
                    )
                    clahh_predictions_mask = clahh_predictions == 1
                    clahh_signal_events_mask = (
                        valid_events_mask & clahh_predictions_mask
                    )
                    logger.info(
                        "Analysis events passing previous cuts and CLAHH %s %s: %s (weighted: %s)",
                        clahh_discrim_sel["operator"],
                        clahh_discrim_sel["value"],
                        ak.sum(clahh_signal_events_mask),
                        ak.sum(events.event_weight[clahh_signal_events_mask]),
                    )
                    # if ak.sum(clahh_signal_events_mask) == 0:
                    #     return None

                ############################################
                ## Calculate top veto X_Wt
                ############################################
                if "X_Wt_discriminant" in selections:
                    top_veto_sel = selections["X_Wt_discriminant"]
                    # reconstruct W and top candidates
                    W_candidates_p4, top_candidates_p4 = get_W_t_p4(
                        jets_p4,
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

                # ###### scan m_X_lead and m_X_sub and reconstruct H1 and H2 ######
                # logger.info("Scanning m_X_lead and m_X_sub")
                # pairing = "min_mass_optimize_2D_pairing"
                # pairing_info = {
                #     "label": r"$\mathrm{arg\,min\,} ((m_{jj}^{lead}-m_\mathrm{X}^{lead})^2 + (m_{jj}^{sub}-m_\mathrm{X}^{sub})^2)$ pairing",
                #     "loss": lambda m_X_lead, m_X_sub: lambda jet_p4, jet_pair_1, jet_pair_2: (
                #         (
                #             np.maximum(
                #                 (
                #                     jet_p4[:, jet_pair_1[0]] + jet_p4[:, jet_pair_1[1]]
                #                 ).mass,
                #                 (
                #                     jet_p4[:, jet_pair_2[0]] + jet_p4[:, jet_pair_2[1]]
                #                 ).mass,
                #             )
                #             - m_X_lead
                #         )
                #         ** 2
                #         + (
                #             np.minimum(
                #                 (
                #                     jet_p4[:, jet_pair_1[0]] + jet_p4[:, jet_pair_1[1]]
                #                 ).mass,
                #                 (
                #                     jet_p4[:, jet_pair_2[0]] + jet_p4[:, jet_pair_2[1]]
                #                 ).mass,
                #             )
                #             - m_X_sub
                #         )
                #         ** 2
                #     ),
                #     "optimizer": np.argmin,
                #     "m_X_range": (np.linspace(0, 150, 16), np.linspace(0, 150, 16)),
                # }
                # m_X_lead_range, m_X_sub_range = pairing_info["m_X_range"]
                # for m_X_lead, m_X_sub in it.product(m_X_lead_range, m_X_sub_range):
                #     pairing_id = f"{pairing}_m_X_lead_{m_X_lead}_m_X_sub_{m_X_sub}"
                #     selection_name = f"{n_btags}btags_{btagger}_{pairing_id}"
                #     # use the m_X_lead and m_X_sub to scan over all possible pairings
                #     pairing_loss = pairing_info["loss"](m_X_lead * GeV, m_X_sub * GeV)
                #     H1_jet_idx, H2_jet_idx = reconstruct_hh_jet_pairs(
                #         jets,
                #         hh_jet_idx=hh_jet_idx,
                #         loss=pairing_loss,
                #         optimizer=pairing_info["optimizer"],
                #     )
                #     events[f"H1_{selection_name}_jet_idx"] = H1_jet_idx
                #     events[f"H2_{selection_name}_jet_idx"] = H2_jet_idx
                #     if is_mc:
                #         ###### Truth match the H1 and H2 ######
                #         H1_truth_matched_jet_idx, H2_truth_matched_jet_idx = (
                #             reconstruct_hh_jet_pairs(
                #                 jets,
                #                 hh_jet_idx=hh_truth_matched_jet_idx,
                #                 loss=pairing_loss,
                #                 optimizer=pairing_info["optimizer"],
                #             )
                #         )
                #         events[f"H1_{selection_name}_truth_matched_jet_idx"] = (
                #             H1_truth_matched_jet_idx
                #         )
                #         events[f"H2_{selection_name}_truth_matched_jet_idx"] = (
                #             H2_truth_matched_jet_idx
                #         )

                #         correct_hh_pairs_mask = select_correct_hh_pair_events(
                #             h1_jets_idx=H1_truth_matched_jet_idx,
                #             h2_jets_idx=H2_truth_matched_jet_idx,
                #             jet_truth_H_parent_mask=events.jet_truth_H_parent_mask,
                #         )
                #         events[f"correct_hh_{selection_name}_mask"] = (
                #             correct_hh_pairs_mask
                #         )
                #         logger.debug(
                #             "Events passing previous cuts and correct H1 and H2 jet pairs with %s: %s (weighted: %s)",
                #             pairing_id.replace("_", " "),
                #             ak.sum(correct_hh_pairs_mask),
                #             ak.sum(events.event_weight[correct_hh_pairs_mask]),
                #         )

                ############################################
                # Apply different HH jet candidate pairings
                ############################################
                pairing_methods = {}
                if "pairing" in selections:
                    pairing_methods = {
                        k: v
                        for k, v in all_pairing_methods.items()
                        if k in selections["pairing"]
                    }
                for pairing, pairing_info in pairing_methods.items():
                    ## NEED TO MASK THE EVENTS WITH THE X_Wt MASK ##
                    valid_events_pairing_mask = np.copy(valid_events_mask)

                    ###### reconstruct H1 and H2 ######
                    H1_jet_idx, H2_jet_idx = reconstruct_hh_jet_pairs(
                        jets_p4,
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
                                jets_p4,
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
                            "Analysis events passing previous cuts and correct H1 and H2 jet pairs using %s pairing: %s (weighted: %s)",
                            pairing.replace("_", " "),
                            ak.sum(correct_hh_pairs_mask),
                            ak.sum(events.event_weight[correct_hh_pairs_mask]),
                        )

                    #################################################
                    # Perform event selection for each pairing method
                    #################################################
                    h1_p4 = ak.sum(jets_p4[H1_jet_idx], axis=1)
                    h2_p4 = ak.sum(jets_p4[H2_jet_idx], axis=1)
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
                            "Analysis events passing previous cuts and ∆𝜂 HH %s %s using %s pairing: %s (weighted: %s)",
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
                            "Analysis events passing previous cuts and X_Wt %s %s using %s pairing: %s (weighted: %s)",
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
                                    "Analysis events passing previous cuts and X_HH veto for %s region using %s pairing: %s (weighted: %s)",
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
                        "Analysis events passing previous cuts and signal region using %s pairing: %s (weighted: %s)",
                        pairing.replace("_", " "),
                        ak.sum(signal_event_mask),
                        ak.sum(events.event_weight[signal_event_mask]),
                    )
                    logger.info(
                        "Analysis events passing previous cuts and control region using %s pairing: %s (weighted: %s)",
                        pairing.replace("_", " "),
                        ak.sum(control_event_mask),
                        ak.sum(events.event_weight[control_event_mask]),
                    )

                    if is_mc:
                        signal_event_correct_pairs_mask = ak.mask(
                            correct_hh_pairs_mask, signal_event_mask
                        )
                        events[
                            f"signal_correct_pairs_{n_btags}btags_{btagger}_{pairing}_mask"
                        ] = signal_event_correct_pairs_mask
                        logger.debug(
                            "Analysis events passing previous cuts and signal region with wrong pairs using %s pairing: %s (weighted: %s)",
                            pairing.replace("_", " "),
                            ak.sum(~signal_event_correct_pairs_mask),
                            ak.sum(
                                events.event_weight[~signal_event_correct_pairs_mask]
                            ),
                        )
                        control_event_correct_pairs_mask = ak.mask(
                            correct_hh_pairs_mask, control_event_mask
                        )
                        events[
                            f"control_correct_pairs_{n_btags}btags_{btagger}_{pairing}_mask"
                        ] = control_event_correct_pairs_mask
                        logger.debug(
                            "Analysis events passing previous cuts and control region with wrong pairs using %s pairing: %s (weighted: %s)",
                            pairing.replace("_", " "),
                            ak.sum(~control_event_correct_pairs_mask),
                            ak.sum(
                                events.event_weight[~control_event_correct_pairs_mask]
                            ),
                        )

    return events
