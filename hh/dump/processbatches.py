import awkward as ak
import numpy as np
import vector as p4
from hh.shared.utils import (
    logger,
    inv_GeV,
    make_4jet_comb_array,
    format_btagger_model_name,
)
from hh.dump.output import Features, Labels
from hh.nonresonantresolved.selection import (
    select_events_passing_triggers,
    select_n_jets_events,
    select_n_bjets_events,
    select_hc_jets,
    reconstruct_hh_mindeltar,
    select_correct_hh_pair_events,
    get_W_t_p4,
)
from hh.nonresonantresolved.triggers import trig_sets
from hh.shared.selection import X_HH, X_Wt


def process_batch(
    events: ak.Array,
    selections: dict,
    features: dict,
    class_label: str,
    sample_weight: float = 1.0,
    is_mc: bool = True,
) -> ak.Array:
    """Apply analysis regions selection and append info to events."""

    logger.info("Initial Events: %s", len(events))

    # get features and class names to be saved
    features_out = features["out"]
    class_names = features["classes"]

    # append class_names to events and set them to 0 or 1
    for class_name in class_names:
        if class_name == class_label:
            events[Labels(class_name).value] = np.ones(len(events))
        else:
            events[Labels(class_name).value] = np.zeros(len(events))

    if is_mc:
        # events["ftag_sf"] = calculate_scale_factors(events)
        events[Features.EVENT_WEIGHT.value] = (
            events.mc_event_weights[:, 0] * sample_weight
        )
    else:
        events[Features.EVENT_WEIGHT.value] = np.ones(len(events), dtype=float)

    # start adding jet features
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

    events[Features.JET_NUM.value] = ak.num(events.jet_pt, axis=1)

    # check if selections is empty (i.e. no selection)
    if not selections:
        logger.info("No selections applied.")
        return events[[*features_out, *class_names]]

    # check that required configs else exit
    required_configs = ["jets", "btagging"]
    provided_configs = list(selections.keys()) + list(
        [k for v in selections.values() for k in v.keys()]
    )
    remaining_configs = list(set(required_configs) - set(provided_configs))
    assert (
        not remaining_configs
    ), f"Missing configs: {remaining_configs}. Required configs: {required_configs}"

    # set overall event filter
    events["valid_event"] = np.ones(len(events), dtype=bool)

    # get event level selections
    if "events" in selections:
        event_selection = selections["events"]
        if "trigs" in event_selection:
            trig_op, trig_set = (
                event_selection["trigs"].get("operator"),
                event_selection["trigs"].get("value"),
            )
            assert trig_op and trig_set, (
                "Invalid trigger selection. Please provide both operator and value. "
                f"Possible operators: AND, OR. Possible values: {trig_sets.keys()}"
            )
            # select and save events passing the triggers
            passed_trigs_mask = select_events_passing_triggers(events, op=trig_op)
            # keep track of valid events
            events["passed_triggers"] = passed_trigs_mask
            events["valid_event"] = events.valid_event & passed_trigs_mask
            logger.info(
                "Events passing the %s of all triggers: %s",
                trig_op.upper(),
                ak.sum(events.valid_event),
            )
            if len(events[events.valid_event]) == 0:
                return events[events.valid_event]

    # get jet selections
    jet_selection = selections["jets"]
    # select and save jet selections
    n_jets_mask, n_jets_event_mask = select_n_jets_events(
        jets=ak.zip(
            {
                k: events[v]
                for k, v in zip(
                    ["pt", "eta", "jvttag"], ["jet_pt", "jet_eta", "jet_jvttag"]
                )
                if v in events.fields
            }
        ),
        selection=jet_selection,
        do_jvt="jet_jvttag" in events.fields,
    )
    events["valid_event"] = events.valid_event & n_jets_event_mask
    logger.info(
        "Events passing previous cuts and jets selection: %s",
        ak.sum(events.valid_event),
    )
    if len(events[events.valid_event]) == 0:
        return events[events.valid_event]

    # select and save b-jet selections
    bjet_selection = jet_selection["btagging"]
    btagger = format_btagger_model_name(
        bjet_selection["model"], bjet_selection["efficiency"]
    )
    events[Features.JET_BTAG.value] = events[f"jet_btag_{btagger}"]
    events[Features.JET_NBTAGS.value] = ak.sum(events[Features.JET_BTAG.value], axis=1)
    n_bjets_mask, n_bjets_event_mask = select_n_bjets_events(
        jets=ak.zip({"btag": events[Features.JET_BTAG.value], "valid": n_jets_mask}),
        selection=bjet_selection,
    )
    events["valid_event"] = events.valid_event & n_bjets_event_mask
    logger.info(
        "Events passing previous cuts and b-jets selection: %s",
        ak.sum(events.valid_event),
    )
    if len(events[events.valid_event]) == 0:
        return events[events.valid_event]

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
    # calculate bb features
    if Features.EVENT_BB_RMH.value in features_out:
        events[Features.EVENT_BB_RMH.value] = (
            make_4jet_comb_array(four_bjets_p4, lambda x, y: (x + y).mass) / 125.0
        )
    if Features.EVENT_BB_DR.value in features_out:
        events[Features.EVENT_BB_DR.value] = make_4jet_comb_array(
            four_bjets_p4, lambda x, y: x.deltaR(y)
        )
    if Features.EVENT_BB_DETA.value in features_out:
        events[Features.EVENT_BB_DETA.value] = make_4jet_comb_array(
            four_bjets_p4, lambda x, y: abs(x.eta - y.eta)
        )

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

    # correctly paired Higgs bosons to further clean up labels
    if is_mc:
        correct_hh_pairs_from_truth = select_correct_hh_pair_events(events)
        events["valid_event"] = events.valid_event & correct_hh_pairs_from_truth
        logger.info(
            "Events passing previous cuts and truth-matched to HH: %s",
            ak.sum(events.valid_event),
        )
        if len(events[events.valid_event]) == 0:
            return events[events.valid_event]

    # calculate X_Wt
    if Features.EVENT_X_WT.value in features_out:
        W_candidates_p4, top_candidates_p4 = get_W_t_p4(
            ak.zip(
                {
                    "pt": events.jet_pt,
                    "eta": events.jet_eta,
                    "phi": events.jet_phi,
                    "mass": events.jet_mass,
                    "btag": events.jet_btag,
                }
            ),
            hc_jet_idx,
            non_hc_jet_idx,
        )
        X_Wt_discriminant = X_Wt(
            W_candidates_p4.mass * inv_GeV,
            top_candidates_p4.mass * inv_GeV,
        )
        # select only the minimum X_Wt for each event
        X_Wt_discriminant = ak.min(X_Wt_discriminant, axis=1)
        events[Features.EVENT_X_WT.value] = X_Wt_discriminant

    # calculate HH features
    h1_jet1_idx, h1_jet2_idx = (
        leading_h_jet_idx[:, 0, np.newaxis],
        leading_h_jet_idx[:, 1, np.newaxis],
    )
    h2_jet1_idx, h2_jet2_idx = (
        subleading_h_jet_idx[:, 0, np.newaxis],
        subleading_h_jet_idx[:, 1, np.newaxis],
    )
    h1 = jets_p4[h1_jet1_idx] + jets_p4[h1_jet2_idx]
    h2 = jets_p4[h2_jet1_idx] + jets_p4[h2_jet2_idx]
    if Features.EVENT_DELTAETA_HH.value in features_out:
        events[Features.EVENT_DELTAETA_HH.value] = np.abs(
            ak.firsts(h1.eta) - ak.firsts(h2.eta)
        )
    if Features.EVENT_X_HH.value in features_out:
        events[Features.EVENT_X_HH.value] = X_HH(
            ak.firsts(h1.m) * inv_GeV, ak.firsts(h2.m) * inv_GeV
        )

    return events[events.valid_event][[*features_out, *class_names]]
