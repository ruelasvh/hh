import awkward as ak
import numpy as np
from src.shared.utils import (
    logger,
    format_btagger_model_name,
)
from .selection import (
    select_n_jets_events,
    select_n_bjets_events,
    select_hc_jets,
    reconstruct_hh_mindeltar,
    select_X_Wt_events,
    select_hh_events,
    select_correct_hh_pair_events,
    select_events_passing_all_triggers_OR,
)


def process_batch(
    events: ak.Record,
    event_selection: dict,
    total_weight: float = 1.0,
    is_mc: bool = True,
) -> ak.Record:
    """Apply analysis regions selection and append info to events."""

    logger.info("Initial Events: %s", len(events))
    # set some default columns
    btagger = format_btagger_model_name(
        event_selection["btagging"]["model"], event_selection["btagging"]["efficiency"]
    )
    events["jet_btag"] = events[f"jet_btag_{btagger}"]
    events["btag_num"] = ak.sum(events.jet_btag, axis=1)
    events["jet_num"] = ak.num(events.jet_pt)
    events["event_weight"] = np.ones_like(events.event_number)
    if is_mc:
        events["mc_event_weight"] = ak.firsts(events.mc_event_weights, axis=1)
        # events["ftag_sf"] = calculate_scale_factors(events)
        events["event_weight"] = (
            np.prod([events.mc_event_weight, events.pileup_weight], axis=0)
            * total_weight
        )
    # select and save events passing the OR of all triggers
    passed_trigs_mask = select_events_passing_all_triggers_OR(events)
    logger.info("Events passing the OR of all triggers: %s", ak.sum(passed_trigs_mask))
    events["passed_triggers"] = passed_trigs_mask
    # keep track of valid events
    events["valid_event"] = passed_trigs_mask
    # select and save events with >= n central jets
    central_jets_sel = event_selection["central_jets"]
    n_central_jets_mask, with_n_central_jets = select_n_jets_events(
        jets=ak.zip(
            {
                "pt": events.jet_pt,
                "eta": events.jet_eta,
                "jvttag": events.jet_jvttag,
            }
        ),
        selection=central_jets_sel,
    )
    events["n_central_jets"] = n_central_jets_mask
    events["with_n_central_jets"] = with_n_central_jets
    # keep track of valid events
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
    # select and save events with >= n central b-jets
    bjets_sel = event_selection["btagging"]
    n_central_bjets_mask, with_n_central_bjets = select_n_bjets_events(
        jets=ak.zip({"btags": events.jet_btag, "valid": n_central_jets_mask}),
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
    # # logger.info("Events with >= 6 central or forward jets", len(events_with_central_or_forward_jets))
    # get the higgs candidate jets indices
    hc_jet_idx, non_hc_jet_idx = select_hc_jets(events)
    events["hc_jet_idx"] = hc_jet_idx
    events["non_hc_jet_idx"] = non_hc_jet_idx
    # calculate hh delta eta
    leading_h_jet_idx, subleading_h_jet_idx = reconstruct_hh_mindeltar(events)
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
    passed_top_veto_mask, X_Wt_discriminant_min = select_X_Wt_events(
        events,
        selection=event_selection["top_veto"],
    )
    events["X_Wt_discriminant_min"] = X_Wt_discriminant_min
    events["passed_top_veto"] = passed_top_veto_mask
    # keep track of valid events
    events["valid_event"] = events.valid_event & passed_top_veto_mask
    top_veto_sel = event_selection["top_veto"]
    logger.info(
        "Events passing central jet selection and top-veto discriminant %s %s: %s",
        top_veto_sel["operator"],
        top_veto_sel["value"],
        ak.sum(events.valid_event),
    )
    hh_deltaeta_sel = event_selection["hh_deltaeta_veto"]
    passed_hh_deltaeta_mask, hh_deltaeta_discrim = select_hh_events(
        events, deltaeta_sel=hh_deltaeta_sel
    )
    events["hh_deltaeta_discriminant"] = hh_deltaeta_discrim
    events["passed_hh_deltaeta"] = passed_hh_deltaeta_mask
    # keep track of valid events
    events["valid_event"] = events.valid_event & passed_hh_deltaeta_mask
    logger.info(
        "Events passing passing central jet selection and |deltaEta_HH| %s %s: %s",
        hh_deltaeta_sel["operator"],
        hh_deltaeta_sel["value"],
        ak.sum(events.valid_event),
    )
    # calculate mass discriminant for signal and control regions
    # signal region
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
