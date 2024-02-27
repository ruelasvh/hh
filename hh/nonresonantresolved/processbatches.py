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
    reconstruct_hh_mindeltar,
    select_X_Wt_events,
    select_hh_events,
    select_correct_hh_pair_events,
    select_events_passing_triggers,
)


def process_batch(
    events: ak.Record,
    event_selection: dict,
    sample_weight: float = 1.0,
    is_mc: bool = False,
) -> ak.Record:
    """Apply analysis regions selection and append info to events."""

    logger.info("Initial Events: %s", len(events))

    if "event_weight" not in events.fields:
        events["event_weight"] = np.ones(len(events), dtype=float)
    if is_mc:
        # events["ftag_sf"] = calculate_scale_factors(events)
        events["event_weight"] = (
            np.prod([events.mc_event_weights[:, 0], events.pileup_weight], axis=0)
            * sample_weight
        )

    # check if event_selection is empty (i.e. no selection)
    if not event_selection:
        logger.info("No event selection applied.")
        return events

    # set overall event filter, up to signal and background selections
    events["valid_event"] = np.ones(len(events), dtype=bool)

    # add jet and b-tagging info
    events["jet_num"] = ak.num(events.jet_pt)
    if "jet_btag" not in events.fields:
        btagger = format_btagger_model_name(
            event_selection["btagging"]["model"],
            event_selection["btagging"]["efficiency"],
        )
        events["jet_btag"] = events[f"jet_btag_{btagger}"]
    events["btag_num"] = ak.sum(events.jet_btag, axis=1)

    # select and save events passing the OR of all triggers
    if "trigs" in event_selection:
        trig_op, trig_set = (
            event_selection["trigs"].get("operator"),
            event_selection["trigs"].get("value"),
        )
        assert trig_op and trig_set, (
            "Invalid trigger selection. Please provide both operator and value. "
            f"Possible operators: AND, OR. Possible values: {trig_sets.keys()}"
        )
        passed_trigs_mask = select_events_passing_triggers(events, op=trig_op)
        events["passed_triggers"] = passed_trigs_mask
        events["valid_event"] = events.valid_event & passed_trigs_mask
        logger.info(
            "Events passing the %s of all triggers: %s",
            trig_op.upper(),
            ak.sum(events.valid_event),
        )
        if len(events[events.valid_event]) == 0:
            return events

    # select and save events with >= n central jets
    central_jets_sel = event_selection["central_jets"]
    n_central_jets_mask, with_n_central_jets = select_n_jets_events(
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
    events["n_central_jets"] = n_central_jets_mask
    events["with_n_central_jets"] = with_n_central_jets
    events["valid_event"] = events.valid_event & with_n_central_jets
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
    if len(events[events.valid_event]) == 0:
        return events

    # select and save events with >= n central b-jets
    bjets_sel = event_selection["btagging"]
    n_central_bjets_mask, with_n_central_bjets = select_n_bjets_events(
        jets=ak.zip({"btag": events.jet_btag, "valid": n_central_jets_mask}),
        selection=bjets_sel,
    )
    events["n_central_bjets"] = n_central_bjets_mask
    events["with_n_central_bjets"] = with_n_central_bjets
    # keep track of valid events
    events["valid_event"] = events.valid_event & with_n_central_bjets
    logger.info(
        "Events passing previous cut and %s %s central b-jets tagged with %s and %s efficiency: %s",
        bjets_sel["count"]["operator"],
        bjets_sel["count"]["value"],
        bjets_sel["model"],
        bjets_sel["efficiency"],
        ak.sum(events.valid_event),
    )
    if len(events[events.valid_event]) == 0:
        return events

    # get the higgs candidate jets indices
    hc_jet_idx, non_hc_jet_idx = select_hc_jets(
        jets=ak.zip(
            {
                "valid": n_central_bjets_mask,
                "pt": events.jet_pt,
                "btag": events.jet_btag,
            }
        )
    )
    events["hc_jet_idx"] = hc_jet_idx
    events["non_hc_jet_idx"] = non_hc_jet_idx
    # reconstruct higgs candidates using the minimum deltaR
    leading_h_jet_idx, subleading_h_jet_idx = reconstruct_hh_mindeltar(
        jets=ak.zip(
            {
                "pt": events.jet_pt,
                "eta": events.jet_eta,
                "phi": events.jet_phi,
                "mass": events.jet_mass,
            }
        ),
        hc_jet_idx=hc_jet_idx,
    )
    events["leading_h_jet_idx"] = leading_h_jet_idx
    events["subleading_h_jet_idx"] = subleading_h_jet_idx
    # correctly paired Higgs bosons
    if is_mc:
        correct_hh_pairs_from_truth = select_correct_hh_pair_events(events)
        logger.info(
            "Events with correct HH pairs: %s",
            ak.sum(correct_hh_pairs_from_truth),
        )

    # calculate top veto discriminant
    if "top_veto" in event_selection:
        top_veto_sel = event_selection["top_veto"]
        passed_top_veto_mask, X_Wt_discriminant_min = select_X_Wt_events(
            events,
            selection=top_veto_sel,
        )
        events["X_Wt_discriminant_min"] = X_Wt_discriminant_min
        events["passed_top_veto"] = passed_top_veto_mask
        # keep track of valid events
        events["valid_event"] = events.valid_event & passed_top_veto_mask
        logger.info(
            "Events passing central jet selection and top-veto discriminant %s %s: %s",
            top_veto_sel["operator"],
            top_veto_sel["value"],
            ak.sum(events.valid_event),
        )
        if len(events[events.valid_event]) == 0:
            return events

    if "hh_deltaeta_veto" in event_selection:
        hh_deltaeta_sel = event_selection["hh_deltaeta_veto"]
        passed_hh_deltaeta_mask, hh_deltaeta_discrim = select_hh_events(
            events, deltaeta_sel=hh_deltaeta_sel
        )
        events["hh_deltaeta_discriminant"] = hh_deltaeta_discrim
        events["passed_hh_deltaeta"] = passed_hh_deltaeta_mask
        # keep track of valid events
        events["valid_event"] = events.valid_event & passed_hh_deltaeta_mask
        logger.info(
            "Events passing central jet selection and |deltaEta_HH| %s %s: %s",
            hh_deltaeta_sel["operator"],
            hh_deltaeta_sel["value"],
            ak.sum(events.valid_event),
        )
        if len(events[events.valid_event]) == 0:
            return events

    ############################################################
    # Calculate mass discriminant for signal and control regions
    ############################################################

    # signal region
    if (
        "hh_mass_veto" in event_selection
        and "signal" in event_selection["hh_mass_veto"]
    ):
        signal_hh_mass_selection = event_selection["hh_mass_veto"]["signal"]
        (
            passed_signal_hh_mass_mask,
            hh_mass_discrim_signal,
        ) = select_hh_events(
            events,
            mass_sel=signal_hh_mass_selection,
        )
        events["hh_mass_discriminant_signal"] = hh_mass_discrim_signal
        events["passed_signal_hh_mass"] = passed_signal_hh_mass_mask
        # keep track of signal region events
        events["signal_event"] = events.valid_event & passed_signal_hh_mass_mask
        logger.info(
            "Signal events passing previous cut and di-Higgs mass discriminant %s %s: %s",
            signal_hh_mass_selection["inner_boundry"]["operator"],
            signal_hh_mass_selection["inner_boundry"]["value"],
            ak.sum(events.signal_event),
        )

    # control region
    if (
        "hh_mass_veto" in event_selection
        and "control" in event_selection["hh_mass_veto"]
    ):
        control_hh_mass_selection = event_selection["hh_mass_veto"]["control"]
        (
            passed_control_hh_mass_mask,
            hh_mass_discrim_control,
        ) = select_hh_events(
            events,
            mass_sel=control_hh_mass_selection,
        )
        events["hh_mass_discriminant_control"] = hh_mass_discrim_control
        events["passed_control_hh_mass"] = passed_control_hh_mass_mask
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
