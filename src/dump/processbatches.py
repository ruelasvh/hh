import awkward as ak
import numpy as np
import vector as p4
from src.shared.utils import (
    logger,
    inv_GeV,
    format_btagger_model_name,
)
from src.dump.output import Features, Labels
from src.nonresonantresolved.selection import (
    select_n_jets_events,
    select_n_bjets_events,
    select_hc_jets,
    reconstruct_hh_mindeltar,
    select_X_Wt_events,
    select_hh_events,
    select_correct_hh_pair_events,
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

    # check that required configs else exit
    required_configs = ["jets", "btagging"]
    provided_configs = list(selections.keys()) + list([k for k in selections.keys()])
    remaining_configs = list(set(required_configs) - set(provided_configs.keys()))
    assert (
        not remaining_configs
    ), f"Missing configs: {remaining_configs}. Required configs: {required_configs}"

    logger.info("Initial Events: %s", len(events))

    # get features and class names to be saved
    features_out = features["out"]
    class_names = features["classes"]

    # get jet and b-jet selections
    jet_selection = selections["jets"]
    bjet_selection = jet_selection["btagging"]
    btagger = format_btagger_model_name(
        bjet_selection["model"], bjet_selection["efficiency"]
    )

    # set event filter
    events["valid_event"] = np.ones(len(events), dtype=bool)

    # start setting data to be saved
    events[Features.JET_NUM.value] = ak.num(events.jet_pt, axis=1)
    events[Features.JET_BTAG.value] = events[f"jet_btag_{btagger}"]
    events[Features.JET_NBTAGS.value] = ak.sum(events.jet_btag, axis=1)
    if is_mc:
        events[Features.EVENT_MCWEIGHT.value] = events.mc_event_weights[:, 0]
        events[Features.EVENT_PUWEIGHT.value] = events.pileup_weight
        events[Features.EVENT_XWEIGHT.value] = (
            np.ones(len(events), dtype=float) * total_weight
        )
        # events["ftag_sf"] = calculate_scale_factors(events)
    # convert jets to cartesian coordinates and save them
    jets_p4 = p4.zip(
        {
            "pt": events.jet_pt * inv_GeV,
            "eta": events.jet_eta,
            "phi": events.jet_phi,
            "mass": events.jet_mass * inv_GeV,
        }
    )
    jets_xyz = ak.zip(
        {
            Features.JET_X.value: jets_p4.x,
            Features.JET_Y.value: jets_p4.y,
            Features.JET_Z.value: jets_p4.z,
        }
    )
    for f in jets_xyz.fields:
        events[Features(f).value] = jets_xyz[f]

    # append class_names to events and set them to 0 or 1
    for class_name in class_names:
        if class_name == class_label:
            events[Labels(class_name).value] = np.ones_like(
                events.event_number, dtype=int
            )
        else:
            events[Labels(class_name).value] = np.zeros_like(
                events.event_number, dtype=int
            )

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
    if (
        bjet_selection
        and bjet_selection.get("count")
        and bjet_selection["count"]["value"] > 0
    ):
        n_bjets_mask, n_bjets_event_mask = select_n_bjets_events(
            jets=ak.zip({"btags": events[f"jet_btag_{btagger}"], "valid": n_jets_mask}),
            selection=bjet_selection,
        )
        events["valid_event"] = events.valid_event & n_bjets_event_mask
        logger.info(
            "Events passing previous cuts and b-jets selection: %s",
            ak.sum(events.valid_event),
        )

    return events[events.valid_event][[*features_out, *class_names]]
