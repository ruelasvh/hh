import awkward as ak
import awkward_pandas as akpd
import pandas as pd
import numpy as np
import vector as p4
from shared.utils import logger
from .triggers import run3_all as triggers_run3_all
from .utils import (
    find_hist,
    find_hists_by_name,
    format_btagger_model_name,
)
from .selection import (
    select_n_jets_events,
    select_n_bjets_events,
    select_hc_jets,
    reconstruct_hh_mindeltar,
    select_X_Wt_events,
    select_X_Wt_events_nicole,
    select_hh_events,
    select_correct_hh_pair_events,
    sort_jets_by_pt,
    select_events_passing_all_triggers_OR,
)
from src.nonresonantresolved.branches import (
    get_jet_branch_alias_names,
)


def extract_and_append_analysis_regions_info(
    events: pd.DataFrame, luminosity_weight, config, is_mc: bool = True
) -> pd.DataFrame:
    """Apply analysis regions selection and append info to events DataFrame."""

    logger.info("Initial Events: %s", len(events))
    # get event selection from config
    event_selection = config["event_selection"]
    # set some default columns
    btagger = format_btagger_model_name(
        event_selection["btagging"]["model"], event_selection["btagging"]["efficiency"]
    )
    events["jet_btag_default"] = events[f"jet_btag_{btagger}"]
    events["btag_num"] = events.jet_btag_default.ak.sum(axis=1)
    events["jet_num"] = events.jet_pt.ak.num(axis=1)
    events["jet_pt_sorted_idx"] = akpd.from_awkward(
        events.jet_pt.ak.argsort(axis=1, ascending=False),
        name="jet_pt_sorted_idx",
    )
    breakpoint()
    events["jet_p4"] = akpd.from_awkward(
        p4.zip(
            {
                "pt": events.jet_pt.ak.array,
                "eta": events.jet_eta.ak.array,
                "phi": events.jet_phi.ak.array,
                "mass": events.jet_m.ak.array,
            }
        ),
        name="jet_p4",
    )
    events["mc_event_weight"] = ak.firsts(events.mc_event_weights.ak.array, axis=1)
    # select and save events passing the OR of all triggers
    passed_trigs_mask = select_events_passing_all_triggers_OR(
        events,
        triggers_run3_all,
    )
    events.loc[:, "passed_triggers"] = passed_trigs_mask
    events.loc[
        :, "valid_event"
    ] = passed_trigs_mask  # start keeping track of signal events
    logger.info(
        "Events passing the OR of all triggers: %s", events.passed_triggers.sum()
    )
    # select and save events with >= n central jets
    with_n_central_jets = select_n_jets_events(
        events,
        selection=event_selection["central_jets"],
    )
    events.loc[:, "with_n_central_jets"] = with_n_central_jets
    events.loc[:, "valid_event"] = events.valid_event.to_numpy() & with_n_central_jets
    central_jets_sel = event_selection["central_jets"]
    logger.info(
        "Events passing previous cut and with %s %s central jets, pT %s %s and |eta| %s %s: %s",
        central_jets_sel["count"]["operator"],
        central_jets_sel["count"]["value"],
        central_jets_sel["pt"]["operator"],
        central_jets_sel["pt"]["value"],
        central_jets_sel["eta"]["operator"],
        central_jets_sel["eta"]["value"],
        events.valid_event.sum(),
    )
    # select and save events with >= n central b-jets
    with_n_bjets = select_n_bjets_events(
        events,
        selection=event_selection["btagging"],
    )
    events.loc[:, "valid_event"] = events.valid_event.to_numpy() & with_n_bjets
    events.loc[:, "with_n_central_bjets"] = (
        events.with_n_central_jets.to_numpy() & with_n_bjets
    )
    bjets_sel = event_selection["btagging"]
    logger.info(
        "Events passing previous cut and with %s %s central b-jets tagged with %s and %s efficiency: %s",
        bjets_sel["count"]["operator"],
        bjets_sel["count"]["value"],
        bjets_sel["model"],
        bjets_sel["efficiency"],
        events.valid_event.sum(),
    )
    # # logger.info("Events with >= 6 central or forward jets", len(events_with_central_or_forward_jets))
    # get the higgs candidate jets indices
    hc_jets_idx = select_hc_jets(events)
    events.loc[:, "hc_jets_idx"] = akpd.from_awkward(hc_jets_idx, name="hc_jets_idx")
    # calculate top veto discriminant
    passed_top_veto_mask, X_Wt_discriminant_min = select_X_Wt_events(
        events,
        selection=event_selection["top_veto"],
    )
    events.loc[:, "valid_event"] = events.valid_event.to_numpy() & passed_top_veto_mask
    events.loc[:, "passed_top_veto"] = passed_top_veto_mask
    events.loc[:, "X_Wt_discriminant_min"] = X_Wt_discriminant_min
    top_veto_sel = event_selection["top_veto"]
    logger.info(
        "Events passing previous cut and with top-veto discriminant %s %s: %s",
        top_veto_sel["operator"],
        top_veto_sel["value"],
        events.valid_event.sum(),
    )
    # calculate hh delta eta
    leading_h_jets_idx, subleading_h_jets_idx = reconstruct_hh_mindeltar(events)
    events.loc[:, "leading_h_jets_idx"] = akpd.from_awkward(
        leading_h_jets_idx, name="leading_h_jets_idx"
    )
    events.loc[:, "subleading_h_jets_idx"] = akpd.from_awkward(
        subleading_h_jets_idx, name="subleading_h_jets_idx"
    )
    hh_deltaeta_sel = event_selection["hh_deltaeta_veto"]
    passed_hh_deltaeta_mask, hh_deltaeta_discrim = select_hh_events(
        events, deltaeta_sel=hh_deltaeta_sel
    )
    events.loc[:, "valid_event"] = (
        events.valid_event.to_numpy() & passed_hh_deltaeta_mask
    )
    events.loc[:, "passed_hh_deltaeta"] = passed_hh_deltaeta_mask
    events.loc[:, "hh_deltaeta_discriminant"] = hh_deltaeta_discrim
    logger.info(
        "Events passing previous cut and with |deltaEta_HH| %s %s: %s",
        hh_deltaeta_sel["operator"],
        hh_deltaeta_sel["value"],
        events.valid_event.sum(),
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
    events.loc[:, "passed_signal_hh_mass"] = passed_signal_hh_mass_mask
    events.loc[:, "hh_mass_discriminant_signal"] = hh_mass_discrim_signal
    events.loc[:, "signal_event"] = (
        events.valid_event.to_numpy() & passed_signal_hh_mass_mask
    )
    logger.info(
        "Signal events passing previous cut and with mass discriminant %s %s: %s",
        signal_hh_mass_selection["inner_boundry"]["operator"],
        signal_hh_mass_selection["inner_boundry"]["value"],
        events.signal_event.sum(),
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
    events.loc[:, "passed_control_hh_mass"] = passed_control_hh_mass_mask
    events.loc[:, "hh_mass_discriminant_control"] = hh_mass_discrim_control
    events.loc[:, "control_event"] = (
        events.valid_event.to_numpy() & passed_control_hh_mass_mask
    )
    logger.info(
        "Control events passing previous cut and with mass discriminant between %s %s and %s %s: %s",
        control_hh_mass_selection["inner_boundry"]["operator"],
        control_hh_mass_selection["inner_boundry"]["value"],
        control_hh_mass_selection["outer_boundry"]["operator"],
        control_hh_mass_selection["outer_boundry"]["value"],
        events.control_event.sum(),
    )
    ################################ WIP ################################

    # # # WIP correctly paired Higgs bosons
    # # correct_hh_pairs_from_truth = select_correct_hh_pair_events(
    # #     valid_events, leading_h_jet_indices, subleading_h_jet_indices, args.signal
    # # )
    # # logger.info(
    # #     "Events with correct HH pairs: %s",
    # #     ak.sum(correct_hh_pairs_from_truth),
    # # )

    return events
