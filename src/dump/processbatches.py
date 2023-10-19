import awkward as ak
import numpy as np
import vector as p4
from src.shared.utils import (
    logger,
    make_4jet_comb_array,
    format_btagger_model_name,
)
from src.dump.output import Features, Labels
from src.nonresonantresolved.selection import (
    select_events_passing_all_triggers_OR,
    select_n_jets_events,
    select_n_bjets_events,
    select_hc_jets,
    reconstruct_hh_mindeltar,
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
    provided_configs = list(selections.keys()) + list(
        [k for v in selections.values() for k in v.keys()]
    )
    remaining_configs = list(set(required_configs) - set(provided_configs))
    assert (
        not remaining_configs
    ), f"Missing configs: {remaining_configs}. Required configs: {required_configs}"

    logger.info("Initial Events: %s", len(events))

    # get features and class names to be saved
    features_out = features["out"]
    class_names = features["classes"]

    # set event wide selections variables
    event_selection = selections.get("events")
    # get jet and b-jet selections
    jet_selection = selections["jets"]
    bjet_selection = jet_selection["btagging"]
    btagger = format_btagger_model_name(
        bjet_selection["model"], bjet_selection["efficiency"]
    )

    # set overall event filter
    events["valid_event"] = np.ones(len(events), dtype=bool)

    if event_selection and event_selection.get("trigs"):
        # select and save events passing the OR of all triggers
        passed_trigs_mask = select_events_passing_all_triggers_OR(events)
        # keep track of valid events
        events["valid_event"] = events.valid_event & passed_trigs_mask
        logger.info(
            "Events passing the OR of all triggers: %s", ak.sum(events.valid_event)
        )

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

    jets_p4 = p4.zip(
        {
            "pt": events.jet_pt,
            "eta": events.jet_eta,
            "phi": events.jet_phi,
            "mass": events.jet_mass,
        }
    )
    # convert jets to cartesian coordinates and save them
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
            events[Labels(class_name).value] = np.ones(len(events))
        else:
            events[Labels(class_name).value] = np.zeros(len(events))

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
    if bjet_selection:
        n_bjets_mask, n_bjets_event_mask = select_n_bjets_events(
            jets=ak.zip(
                {"btag": events[Features.JET_BTAG.value], "valid": n_jets_mask}
            ),
            selection=bjet_selection,
        )
        events["valid_event"] = events.valid_event & n_bjets_event_mask
        logger.info(
            "Events passing previous cuts and b-jets selection: %s",
            ak.sum(events.valid_event),
        )
        # select and save hc jets
        hc_jet_idx, non_hc_jet_idx = select_hc_jets(
            jets=ak.zip(
                {
                    "valid": n_bjets_mask,
                    "pt": events.jet_pt,
                    "btag": events[Features.JET_BTAG.value],
                }
            )
        )
        four_bjets_p4 = ak.mask(jets_p4[hc_jet_idx], events.valid_event)
        events[Features.EVENT_M_4B.value] = (
            four_bjets_p4[:, 0]
            + four_bjets_p4[:, 1]
            + four_bjets_p4[:, 2]
            + four_bjets_p4[:, 3]
        ).mass
        # four_bjet_combs = ak.combinations(hc_jet_idx, 2, axis=1)
        # left_bjet, right_bjet = ak.unzip(four_bjet_combs)
        # events[Features.EVENT_BB_RMH.value] = (
        #     jets_p4[left_bjet] + jets_p4[right_bjet]
        # ).mass / 125.0
        # events[Features.EVENT_BB_DR.value] = jets_p4[left_bjet].deltaR(
        #     jets_p4[right_bjet]
        # )
        # events[Features.EVENT_BB_DETA.value] = abs(
        #     jets_p4[left_bjet].eta - jets_p4[right_bjet].eta
        # )
        events[Features.EVENT_BB_RMH.value] = (
            make_4jet_comb_array(four_bjets_p4, lambda x, y: (x + y).mass) / 125.0
        )
        events[Features.EVENT_BB_DR.value] = make_4jet_comb_array(
            four_bjets_p4, lambda x, y: x.deltaR(y)
        )
        events[Features.EVENT_BB_DETA.value] = make_4jet_comb_array(
            four_bjets_p4, lambda x, y: abs(x.eta - y.eta)
        )
        # correctly paired Higgs bosons to further clean up labels
        if is_mc:
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
            correct_hh_pairs_from_truth = select_correct_hh_pair_events(
                events["jet_truth_H_parents"], leading_h_jet_idx, subleading_h_jet_idx
            )
            logger.info(
                "Events with correct HH pairs: %s",
                ak.sum(correct_hh_pairs_from_truth),
            )
            events["valid_event"] = events.valid_event & correct_hh_pairs_from_truth

    return events[events.valid_event][[*features_out, *class_names]]
