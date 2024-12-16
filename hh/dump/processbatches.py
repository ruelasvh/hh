import awkward as ak
import numpy as np
import vector
from hh.shared.utils import (
    logger,
    GeV,
    make_4jet_comb_array,
    format_btagger_model_name,
    get_common,
    update_cutflow,
)
from hh.shared.labels import kin_labels
from hh.dump.output import Features, Labels
from hh.nonresonantresolved.selection import (
    select_events_passing_triggers,
    select_n_jets_events,
    select_n_bjets_events,
    select_hh_jet_candidates,
    reconstruct_hh_jet_pairs,
)
from hh.nonresonantresolved.triggers import trig_sets
from hh.shared.selection import X_HH, X_Wt, get_W_t_p4
from hh.nonresonantresolved.pairing import pairing_methods

vector.register_awkward()


def process_batch(
    events: ak.Record,
    selections: dict,
    outputs: dict,
    class_label: str,
    sample_weight: float = 1.0,
    is_mc: bool = True,
    cutflow: dict = None,
) -> ak.Record:
    """Apply analysis regions selection and append info to events."""

    # get features and class names to be saved
    feature_names = outputs["features"]
    label_names = outputs["labels"]

    # append label_names to events and set them to 0 or 1
    for class_name in label_names:
        if class_name == class_label:
            events[Labels(class_name).value] = np.ones(len(events))
        else:
            events[Labels(class_name).value] = np.zeros(len(events))

    events[Features.EVENT_WEIGHT.value] = (
        np.ones(len(events), dtype=float) * sample_weight
    )
    if is_mc:
        events["event_weight"] = (
            np.prod([events.mc_event_weights[:, 0], events.pileup_weight], axis=0)
            * sample_weight
        )
        events[Features.EVENT_WEIGHT.value] = events["event_weight"]

    logger.info(
        "Initial Events: %s (weighted: %s)", len(events), ak.sum(events.event_weight)
    )
    if cutflow is not None:
        cutname = "initial_events"
        update_cutflow(cutflow, cutname, np.ones(len(events)), events.event_weight)

    # start adding jet features
    events[Features.JET_NUM.value] = ak.num(events.jet_pt, axis=-1)
    # build 4-momentum vectors for jets
    jets_p4 = ak.zip(
        {k: events[f"jet_{k}"] for k in ["jvttag", *kin_labels.keys()]},
        with_name="Momentum4D",
    )
    # convert jets to cartesian coordinates and save them
    for f, v in zip(
        [Features.JET_PX, Features.JET_PY, Features.JET_PZ],
        [jets_p4.px, jets_p4.py, jets_p4.pz],
    ):
        events[f.value] = v

    # check if selections is empty (i.e. no selection)
    if not selections:
        logger.info("No objects selection applied.")
        features_out = get_common(events.fields, feature_names)
        if cutflow is None:
            return events[[*features_out, *label_names]]
        else:
            return events[[*features_out, *label_names]], cutflow

    # get event level selections
    if "trigs" in selections:
        trig_op, trig_set = (
            selections["trigs"].get("operator"),
            selections["trigs"].get("value"),
        )
        assert trig_op and trig_set, (
            "Invalid trigger selection. Please provide both operator and value. "
            f"Possible operators: AND, OR. Possible values: {trig_sets.keys()}"
        )
        # select and save events passing the triggers
        passed_trigs_mask = select_events_passing_triggers(events, op=trig_op)
        events = events[passed_trigs_mask]
        logger.info(
            "Events passing the %s of all triggers: %s (weighted: %s)",
            trig_op.upper(),
            len(events),
            ak.sum(events.event_weight),
        )
        if cutflow is not None:
            cutname = "_".join(["triggers", f"pass_{trig_op.lower()}"]).replace(
                ".", "p"
            )
            update_cutflow(cutflow, cutname, passed_trigs_mask, events.event_weight)
        if len(events) == 0:
            features_out = get_common(events.fields, feature_names)
            if cutflow is None:
                return events[[*features_out, *label_names]]
            else:
                return events[[*features_out, *label_names]], cutflow

    # get jet selections
    if "jets" in selections:
        jet_selection = selections["jets"]
        valid_jets = select_n_jets_events(
            jets=jets_p4,
            selection=jet_selection,
            do_jvt=False,
        )
        valid_events_mask = ~ak.is_none(valid_jets, axis=0)
        valid_jets = valid_jets[valid_events_mask]
        events = events[valid_events_mask]
        logger.info(
            "Events passing previous cuts and jets selection: %s (weighted: %s)",
            len(events),
            ak.sum(events.event_weight),
        )
        if cutflow is not None:
            cutname = "_".join(
                [
                    "jets",
                    f"pt_{jet_selection['pt']['value']}",
                    f"eta_{jet_selection['eta']['value']}",
                    f"count_{jet_selection['count']['value']}",
                ]
            ).replace(".", "p")
            update_cutflow(cutflow, cutname, valid_events_mask, events.event_weight)
        if len(events) == 0:
            features_out = get_common(events.fields, feature_names)
            if cutflow is None:
                return events[[*features_out, *label_names]]
            else:
                return events[[*features_out, *label_names]], cutflow

        # select and save b-jet selections
        if "btagging" in jet_selection:
            bjet_selection = jet_selection["btagging"]
            btagger = format_btagger_model_name(
                bjet_selection["model"], bjet_selection["efficiency"]
            )
            events[Features.JET_BTAG.value] = events[f"jet_btag_{btagger}"]
            events[Features.JET_NBTAGS.value] = ak.sum(
                events[Features.JET_BTAG.value], axis=1
            )
            valid_bjets = select_n_bjets_events(
                jets=valid_jets,
                btags=events[Features.JET_BTAG.value],
                selection=bjet_selection,
            )
            valid_events_mask = ~ak.is_none(valid_bjets, axis=0)
            valid_bjets = valid_bjets[valid_events_mask]
            events = events[valid_events_mask]
            logger.info(
                "Events passing previous cuts and b-jets selection: %s (weighted: %s)",
                len(events),
                ak.sum(events.event_weight),
            )
            if cutflow is not None:
                cutname = "_".join(
                    [
                        "bjets",
                        f"tagger_{btagger}",
                        f"count_{bjet_selection['count']['value']}",
                    ]
                ).replace(".", "p")
                update_cutflow(cutflow, cutname, valid_events_mask, events.event_weight)
            if len(events) == 0:
                features_out = get_common(events.fields, feature_names)
                if cutflow is None:
                    return events[[*features_out, *label_names]]
                else:
                    return events[[*features_out, *label_names]], cutflow

            # select and save diHiggs candidates jets
            jets_p4 = ak.zip(
                {
                    "btag": events[f"jet_btag_{btagger}"],
                    **{k: events[f"jet_{k}"] for k in kin_labels},
                },
                with_name="Momentum4D",
            )
            hh_jet_idx, non_hh_jet_idx = select_hh_jet_candidates(
                jets=jets_p4, valid_jets_mask=valid_bjets
            )
            events["hh_jet_idx"] = hh_jet_idx
            events["non_hh_jet_idx"] = non_hh_jet_idx
            four_bjets_p4 = jets_p4[events.hh_jet_idx]
            events[Features.EVENT_M_4B.value] = (
                four_bjets_p4[:, 0]
                + four_bjets_p4[:, 1]
                + four_bjets_p4[:, 2]
                + four_bjets_p4[:, 3]
            ).mass
            # calculate bb features
            if Features.EVENT_BB_RMH.value in feature_names:
                events[Features.EVENT_BB_RMH.value] = (
                    make_4jet_comb_array(four_bjets_p4, lambda x, y: (x + y).mass * GeV)
                    / 125.0
                )
            if Features.EVENT_BB_DR.value in feature_names:
                events[Features.EVENT_BB_DR.value] = make_4jet_comb_array(
                    four_bjets_p4, lambda x, y: x.deltaR(y)
                )
            if Features.EVENT_BB_DETA.value in feature_names:
                events[Features.EVENT_BB_DETA.value] = make_4jet_comb_array(
                    four_bjets_p4, lambda x, y: abs(x.eta - y.eta)
                )

            # reconstruct higgs candidates using the minimum deltaR
            pairing_method = selections["pairing"]
            pairing_info = pairing_methods[pairing_method]
            H1_jet_idx, H2_jet_idx = reconstruct_hh_jet_pairs(
                jets=jets_p4,
                hh_jet_idx=hh_jet_idx,
                loss=pairing_info["loss"],
                optimizer=pairing_info["optimizer"],
            )
            events["leading_h_jet_idx"] = H1_jet_idx
            events["subleading_h_jet_idx"] = H2_jet_idx

            # calculate X_Wt
            if Features.EVENT_X_WT.value in feature_names:
                W_candidates_p4, top_candidates_p4 = get_W_t_p4(
                    jets=jets_p4,
                    hh_jet_idx=events.hh_jet_idx,
                    non_hh_jet_idx=events.non_hh_jet_idx,
                )
                X_Wt_discriminant = X_Wt(
                    W_candidates_p4.mass * GeV,
                    top_candidates_p4.mass * GeV,
                )
                # select only the minimum X_Wt for each event
                X_Wt_discriminant = ak.min(X_Wt_discriminant, axis=1)
                events[Features.EVENT_X_WT.value] = X_Wt_discriminant

            # calculate HH features
            h1_jet1_idx, h1_jet2_idx = (
                events.leading_h_jet_idx[:, 0, np.newaxis],
                events.leading_h_jet_idx[:, 1, np.newaxis],
            )
            h2_jet1_idx, h2_jet2_idx = (
                events.subleading_h_jet_idx[:, 0, np.newaxis],
                events.subleading_h_jet_idx[:, 1, np.newaxis],
            )
            h1 = jets_p4[h1_jet1_idx] + jets_p4[h1_jet2_idx]
            h2 = jets_p4[h2_jet1_idx] + jets_p4[h2_jet2_idx]
            if Features.EVENT_DELTAETA_HH.value in feature_names:
                events[Features.EVENT_DELTAETA_HH.value] = np.abs(
                    ak.firsts(h1.eta) - ak.firsts(h2.eta)
                )
            if Features.EVENT_X_HH.value in feature_names:
                events[Features.EVENT_X_HH.value] = X_HH(
                    ak.firsts(h1.m) * GeV, ak.firsts(h2.m) * GeV
                )

    features_out = get_common(events.fields, feature_names)
    if cutflow is None:
        return events[[*features_out, *label_names]]
    else:
        return events[[*features_out, *label_names]], cutflow
