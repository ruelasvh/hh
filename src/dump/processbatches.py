import awkward as ak
import numpy as np
import vector as p4
from src.shared.utils import (
    logger,
    format_btagger_model_name,
)
from src.nonresonantresolved.selection import (
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
    events: ak.Array,
    selections: dict,
    features: dict,
    class_label: str,
    total_weight: float = 1.0,
    is_mc: bool = True,
) -> ak.Array:
    """Apply analysis regions selection and append info to events."""

    logger.info("Initial Events: %s", len(events))
    # set some default columns
    features_out = features["out"]
    class_names = features["classes"]
    jet_selection = selections["jets"]
    bjet_selection = jet_selection["btagging"]
    btagger = format_btagger_model_name(
        bjet_selection["model"], bjet_selection["efficiency"]
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
    events["valid_event"] = passed_trigs_mask
    logger.info("Events passing the OR of all triggers: %s", ak.sum(events.valid_event))
    # convert jets to cartesian coordinates
    jets_p4 = p4.zip(
        {
            "pt": events.jet_pt,
            "eta": events.jet_eta,
            "phi": events.jet_phi,
            "mass": events.jet_mass,
        }
    )
    jets_xyz = {"jet": ak.zip({"x": jets_p4.x, "y": jets_p4.y, "z": jets_p4.z})}
    for obj, var in jets_xyz.items():
        for coord in var.fields:
            events[f"{obj}_{coord}"] = var[coord]

    # append class_names to events and set them to 0 or 1
    for class_name in class_names:
        if class_name == class_label:
            events[class_name] = np.ones_like(events.event_number)
        else:
            events[class_name] = np.zeros_like(events.event_number)

    # select and save jet selections
    n_jets_mask, n_jets_event_mask = select_n_jets_events(
        jets=ak.zip(
            {
                "pt": events.jet_pt,
                "eta": events.jet_eta,
                "jvttag": events.jet_jvttag,
            }
        ),
        selection=jet_selection,
    )
    events["valid_event"] = events.valid_event & n_jets_event_mask
    logger.info(
        "Events passing previous cuts and jets selection: %s",
        ak.sum(events.valid_event),
    )

    # select and save b-jet selections
    n_bjets_mask, n_bjets_event_mask = select_n_bjets_events(
        jets=ak.zip({"btags": events.jet_btag, "valid": n_jets_mask}),
        selection=bjet_selection,
    )
    events["valid_event"] = events.valid_event & n_bjets_event_mask
    logger.info(
        "Events passing previous cuts and b-jets selection: %s",
        ak.sum(events.valid_event),
    )

    return events[events.valid_event][[*features_out, *class_names]]
