import awkward as ak
import numpy as np
from hh.shared.utils import (
    logger,
    format_btagger_model_name,
)
from hh.nonresonantresolved.triggers import trig_sets
from .selection import (
    select_n_jets_events,
    select_n_bjets_events,
    select_hc_jets,
    select_hh_jet_pairs,
    select_X_Wt_events,
    select_hh_events,
    select_correct_hh_pair_events,
    select_events_passing_triggers,
    select_truth_matched_jets,
)


def process_batch(
    events: ak.Record,
    selections: dict,
    sample_weight: float = 1.0,
    is_mc: bool = False,
) -> ak.Record:
    """Apply analysis regions selection and append info to events."""

    logger.info("Initial Events: %s", len(events))

    if "event_weight" not in events.fields:
        events["event_weight"] = np.ones(len(events), dtype=float) * sample_weight
        if is_mc:
            events["mc_event_weight"] = events.mc_event_weights[:, 0]
            events["event_weight"] = (
                np.prod([events.mc_event_weight, events.pileup_weight], axis=0)
                * sample_weight
            )

    # set overall event filter, up to signal and background selections
    events["valid_event"] = np.ones(len(events), dtype=bool)
    events["valid_central_jets"] = np.ones_like(events.jet_pt, dtype=bool)

    # check if selections is empty (i.e. no selection)
    if not selections:
        logger.info("No objects selection applied.")
        return events

    # start adding objects info
    events["jet_num"] = ak.num(events.jet_pt)

    # apply trigger selection
    if "trigs" in selections:
        trig_set, trig_op = (
            selections["trigs"].get("value"),
            selections["trigs"].get("operator"),
        )
        assert trig_set, "Invalid trigger selection. Please provide a trigger set."
        passed_trigs_mask = select_events_passing_triggers(events, op=trig_op)
        events["valid_event"] = events.valid_event & passed_trigs_mask
        logger.info(
            "Events passing the %s of all triggers: %s",
            trig_op.upper() if trig_op is not None else "None",
            ak.sum(events.valid_event),
        )
        if ak.sum(events.valid_event) == 0:
            return events

    # select and save events with >= n central jets
    if "central_jets" in selections:
        central_jets_sel = selections["central_jets"]
        valid_central_jets, valid_event_central_jets = select_n_jets_events(
            jets=ak.zip(
                {
                    k: events[v]
                    for k, v in zip(
                        ["pt", "eta", "jvttag"], ["jet_pt", "jet_eta", "jet_jvttag"]
                    )
                    if v in events.fields
                }
            ),
            selection=central_jets_sel,
            do_jvt="jet_jvttag" in events.fields,
        )
        events["valid_central_jets"] = valid_central_jets
        events["valid_event"] = events.valid_event & valid_event_central_jets
        logger.info(
            "Events passing previous cut and %s %s central jets with pT %s %s, |eta| %s %s and Jvt tag: %s",
            central_jets_sel["count"]["operator"],
            central_jets_sel["count"]["value"],
            central_jets_sel["pt"]["operator"],
            central_jets_sel["pt"]["value"],
            central_jets_sel["eta"]["operator"],
            central_jets_sel["eta"]["value"],
            ak.sum(events.valid_event),
        )
        if ak.sum(events.valid_event) == 0:
            return events

    if is_mc:
        truth_matched_jets = select_truth_matched_jets(
            jets=ak.zip(
                {
                    "pt": events.jet_pt,
                    "eta": events.jet_eta,
                    "phi": events.jet_phi,
                    "mass": events.jet_mass,
                }
            ),
            hh_truth_mask=events.jet_truth_H_parents,
            valid_jets_mask=events.valid_central_jets,
        )
        events["reco_truth_matched_jets"] = truth_matched_jets
        logger.info(
            "Events passing previous cuts and truth matched: %s",
            ak.sum(~ak.is_none(events.reco_truth_matched_jets, axis=0)),
        )

    # add jet and b-tagging info
    if "btagging" in selections:
        bjets_sel = selections["btagging"]
        btagger = format_btagger_model_name(
            bjets_sel["model"],
            bjets_sel["efficiency"],
        )
        events["jet_btag"] = events[f"jet_btag_{btagger}"]
        events["btag_num"] = ak.sum(events.jet_btag, axis=1)

        # select and save events with >= n central b-jets
        valid_central_jets, valid_event_central_jets = select_n_bjets_events(
            jets=ak.zip(
                {
                    "btag": events.jet_btag,
                    "valid": events.valid_central_jets,
                }
            ),
            selection=bjets_sel,
        )
        events["valid_central_jets"] = valid_central_jets
        events["valid_event"] = events.valid_event & valid_event_central_jets
        logger.info(
            "Events passing previous cut and %s %s central b-jets tagged with %s and %s efficiency: %s",
            bjets_sel["count"]["operator"],
            bjets_sel["count"]["value"],
            bjets_sel["model"],
            bjets_sel["efficiency"],
            ak.sum(events.valid_event),
        )
        if ak.sum(events.valid_event) == 0:
            return events

    # get the diHiggs candidate jets indices
    hh_jet_idx, non_hh_jet_idx = select_hc_jets(
        jets=ak.zip(
            {
                "pt": events.jet_pt,
                "valid": events.valid_central_jets,
            }
        )
    )
    events["hh_jet_idx"] = hh_jet_idx
    events["non_hh_jet_idx"] = non_hh_jet_idx
    # reconstruct higgs candidates using the minimum deltaR
    leading_h_jet_idx, subleading_h_jet_idx = select_hh_jet_pairs(
        jets=ak.zip(
            {
                "pt": events.jet_pt,
                "eta": events.jet_eta,
                "phi": events.jet_phi,
                "mass": events.jet_mass,
            }
        ),
        hh_jet_idx=hh_jet_idx,
    )
    events["leading_h_jet_idx"] = leading_h_jet_idx
    events["subleading_h_jet_idx"] = subleading_h_jet_idx
    # # correctly paired Higgs bosons, might need revision
    # if is_mc:
    #     correct_hh_pairs_from_truth = select_correct_hh_pair_events(
    #         events, events.valid_event
    #     )
    #     events["correct_hh_pairs_from_truth"] = correct_hh_pairs_from_truth
    #     logger.info(
    #         "Events with correct HH pairs: %s",
    #         ak.sum(correct_hh_pairs_from_truth),
    #     )

    # calculate top veto discriminant
    if "top_veto" in selections:
        top_veto_sel = selections["top_veto"]
        passed_top_veto_mask, X_Wt_discriminant_min = select_X_Wt_events(
            events,
            selection=top_veto_sel,
        )
        events["X_Wt_discriminant_min"] = X_Wt_discriminant_min
        events["valid_event"] = events.valid_event & passed_top_veto_mask
        logger.info(
            "Events passing central jet selection and top-veto discriminant %s %s: %s",
            top_veto_sel["operator"],
            top_veto_sel["value"],
            ak.sum(events.valid_event),
        )
        if ak.sum(events.valid_event) == 0:
            return events

    if "hh_deltaeta_veto" in selections:
        hh_deltaeta_sel = selections["hh_deltaeta_veto"]
        passed_hh_deltaeta_mask, hh_deltaeta_discrim = select_hh_events(
            events, deltaeta_sel=hh_deltaeta_sel
        )
        events["hh_deltaeta_discriminant"] = hh_deltaeta_discrim
        events["valid_event"] = events.valid_event & passed_hh_deltaeta_mask
        logger.info(
            "Events passing central jet selection and |deltaEta_HH| %s %s: %s",
            hh_deltaeta_sel["operator"],
            hh_deltaeta_sel["value"],
            ak.sum(events.valid_event),
        )
        if ak.sum(events.valid_event) == 0:
            return events

    ############################################################
    # Calculate mass discriminant for signal and control regions
    ############################################################

    # signal region
    if "hh_mass_veto" in selections and "signal" in selections["hh_mass_veto"]:
        signal_hh_mass_selection = selections["hh_mass_veto"]["signal"]
        (
            passed_signal_hh_mass_mask,
            hh_mass_discrim_signal,
        ) = select_hh_events(
            events,
            mass_sel=signal_hh_mass_selection,
        )
        events["hh_mass_discriminant_signal"] = hh_mass_discrim_signal
        events["signal_event"] = events.valid_event & passed_signal_hh_mass_mask
        logger.info(
            "Signal events passing previous cut and di-Higgs mass discriminant %s %s: %s",
            signal_hh_mass_selection["inner_boundry"]["operator"],
            signal_hh_mass_selection["inner_boundry"]["value"],
            ak.sum(events.signal_event),
        )

    # control region
    if "hh_mass_veto" in selections and "control" in selections["hh_mass_veto"]:
        control_hh_mass_selection = selections["hh_mass_veto"]["control"]
        (
            passed_control_hh_mass_mask,
            hh_mass_discrim_control,
        ) = select_hh_events(
            events,
            mass_sel=control_hh_mass_selection,
        )
        events["hh_mass_discriminant_control"] = hh_mass_discrim_control
        # keep track of control region events
        events["control_event"] = events.valid_event & passed_control_hh_mass_mask
        logger.info(
            "Control events passing previous cut and di-Higgs mass discriminant between %s %s and %s %s: %s",
            control_hh_mass_selection["inner_boundry"]["operator"],
            control_hh_mass_selection["inner_boundry"]["value"],
            control_hh_mass_selection["outer_boundry"]["operator"],
            control_hh_mass_selection["outer_boundry"]["value"],
            ak.sum(events.control_event),
        )

    return events
