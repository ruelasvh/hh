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
    find_matching_field,
)
from hh.shared.labels import kin_labels
from hh.dump.output import Features, Labels, Spectators
from hh.nonresonantresolved.selection import (
    select_events_passing_triggers,
    select_n_jets_events,
    select_n_bjets_events,
    select_hh_jet_candidates,
    reconstruct_hh_jet_pairs,
)
from hh.nonresonantresolved.triggers import trig_sets
from hh.nonresonantresolved.processbatches import (
    process_batch as analysis_process_batch,
)
from hh.nonresonantresolved.branches import get_trigger_branch_aliases

vector.register_awkward()


def process_batch(
    events: ak.Record,
    selections: dict,
    outputs: dict,
    class_label: str,
    sample_weight: float = 1.0,
    is_mc: bool = True,
    year: int = None,
) -> ak.Record:
    """Apply analysis regions selection and append info to events."""

    # get features and class names to be saved
    feature_names = outputs["features"]
    label_names = outputs["labels"]
    spectator_names = outputs["spectators"]
    train_selections = selections["training"]
    analysis_selections = selections["analysis"]
    cutflow = {}

    events[Spectators.EVENT_NUMBER.value] = events.event_number

    # append label_names to events and set them to 0 or 1
    for class_name in label_names:
        if class_name == class_label:
            events[Labels(class_name).value] = np.ones(len(events))
        else:
            events[Labels(class_name).value] = np.zeros(len(events))

    events["event_weight"] = np.ones(len(events), dtype=float) * sample_weight
    if is_mc:
        if getattr(events.mc_event_weights, "ndim", 1) > 1:
            mc_w = events.mc_event_weights[:, 0]
        else:
            mc_w = events.mc_event_weights
        events["event_weight"] = np.prod(
            [mc_w, events.pileup_weight, events.event_weight],
            axis=0,
        )
    events[Spectators.EVENT_WEIGHT.value] = events["event_weight"]

    logger.info(
        "Initial events: %s (weighted: %s)", len(events), ak.sum(events.event_weight)
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
    # save jet kinematics
    for f, v in zip(
        [
            Spectators.JET_PT,
            Spectators.JET_ETA,
            Spectators.JET_PHI,
            Spectators.JET_MASS,
        ],
        [jets_p4.pt, jets_p4.eta, jets_p4.phi, jets_p4.mass],
    ):
        events[f.value] = v
    # convert jets to cartesian coordinates and save them
    for f, v in zip(
        [Features.JET_PX, Features.JET_PY, Features.JET_PZ],
        [jets_p4.px, jets_p4.py, jets_p4.pz],
    ):
        events[f.value] = v

    # check if selections is empty (i.e. no selection)
    if not selections:
        logger.info("No objects selection applied.")
        out_fields = get_common(
            events.fields, [*label_names, *feature_names, *spectator_names]
        )
        if cutflow is None:
            return events[out_fields]
        else:
            return events[out_fields], cutflow

    ############################################
    # Train selections
    ############################################
    # get event level selections
    if "trigs" in train_selections:
        trig_op, trig_set = (
            train_selections["trigs"].get("operator"),
            train_selections["trigs"].get("value"),
        )
        assert trig_op and trig_set, (
            "Invalid trigger selection. Please provide both operator and value. "
            f"Possible operators: AND, OR. Possible values: {trig_sets.keys()}"
        )
        triggers = get_trigger_branch_aliases(trig_set, year)
        # select and save events passing the triggers
        passed_trigs_mask = select_events_passing_triggers(
            events, triggers=triggers.keys(), operator=trig_op
        )
        events = events[passed_trigs_mask]
        logger.info(
            "Events passing the %s of triggers %s: %s (weighted: %s)",
            trig_op.upper(),
            list(triggers.values()),
            len(events),
            ak.sum(events.event_weight),
        )
        if cutflow is not None:
            cutname = "_".join(["triggers", f"pass_{trig_op.lower()}"]).replace(
                ".", "p"
            )
            update_cutflow(cutflow, cutname, passed_trigs_mask, events.event_weight)
        if len(events) == 0:
            out_fields = get_common(
                events.fields, [*label_names, *feature_names, *spectator_names]
            )
            if cutflow is None:
                return events[out_fields]
            else:
                return events[out_fields], cutflow

    # apply jet train selections
    if "jets" in train_selections:
        jet_selection = train_selections["jets"]
        jets_p4 = ak.zip(
            {k: events[f"jet_{k}"] for k in ["jvttag", *kin_labels.keys()]},
            with_name="Momentum4D",
        )
        valid_jets = select_n_jets_events(
            jets=jets_p4,
            selection=jet_selection,
            do_jvt=True,
        )
        valid_events_mask = ~ak.is_none(valid_jets, axis=0)
        valid_jets = valid_jets[valid_events_mask]
        events = events[valid_events_mask]
        logger.info(
            "Events passing previous cut and %s %s jets with pT %s %s, |eta| %s %s and 2 b-tags: %s (weighted: %s)",
            jet_selection["count"]["operator"],
            jet_selection["count"]["value"],
            jet_selection["pt"]["operator"],
            jet_selection["pt"]["value"],
            jet_selection["eta"]["operator"],
            jet_selection["eta"]["value"],
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
            out_fields = get_common(
                events.fields, [*label_names, *feature_names, *spectator_names]
            )
            if cutflow is None:
                return events[out_fields]
            else:
                return events[out_fields], cutflow

        # apply b-jet train selections
        if "btagging" in jet_selection:
            bjet_selection = jet_selection["btagging"]
            btagger = format_btagger_model_name(
                bjet_selection["model"], bjet_selection["efficiency"]
            )
            events[Features.JET_NBTAGS.value] = ak.sum(
                events[f"jet_btag_{btagger}"], axis=1
            )
            valid_bjets = select_n_bjets_events(
                jets=valid_jets,
                btags=events[f"jet_btag_{btagger}"][valid_jets],
                selection=bjet_selection,
            )
            valid_events_mask = ~ak.is_none(valid_bjets, axis=0)
            valid_bjets = valid_bjets[valid_events_mask]
            events = events[valid_events_mask]
            logger.info(
                "Events passing previous cut and %s %s b-tags with %s and %s efficiency: %s (weighted: %s)",
                bjet_selection["count"]["operator"],
                bjet_selection["count"]["value"],
                bjet_selection["model"],
                bjet_selection["efficiency"],
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
                out_fields = get_common(
                    events.fields, [*label_names, *feature_names, *spectator_names]
                )
                if cutflow is None:
                    return events[out_fields]
                else:
                    return (
                        events[out_fields],
                        cutflow,
                    )

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
                ak.sum(four_bjets_p4, axis=1, keepdims=True).mass * GeV
            )
            # calculate bb features
            if Features.EVENT_BB_DM.value in feature_names:
                events[Features.EVENT_BB_DM.value] = (
                    make_4jet_comb_array(four_bjets_p4, lambda x, y: (x + y).mass) * GeV
                )
            if Features.EVENT_BB_DR.value in feature_names:
                events[Features.EVENT_BB_DR.value] = make_4jet_comb_array(
                    four_bjets_p4, lambda x, y: x.deltaR(y)
                )
            if Features.EVENT_BB_DETA.value in feature_names:
                events[Features.EVENT_BB_DETA.value] = make_4jet_comb_array(
                    four_bjets_p4, lambda x, y: abs(x.eta - y.eta)
                )

    ############################################
    # Analysis selections
    ############################################
    analysis_events = analysis_process_batch(
        events[events.fields], analysis_selections, is_mc, year
    )

    X_Wt_discriminant_name = find_matching_field(analysis_events, "X_Wt", "discrim")
    if X_Wt_discriminant_name is not None:
        events[Spectators.EVENT_X_WT.value] = analysis_events[X_Wt_discriminant_name]
    else:
        events[Spectators.EVENT_X_WT.value] = ak.values_astype(
            ak.mask(events.event_number, np.zeros(len(events), dtype=bool)),
            np.float32,
        )

    delta_eta_HH_name = find_matching_field(analysis_events, "deltaeta_HH", "discrim")
    if delta_eta_HH_name is not None:
        events[Spectators.EVENT_DELTAETA_HH.value] = analysis_events[delta_eta_HH_name]
    else:
        events[Spectators.EVENT_DELTAETA_HH.value] = ak.values_astype(
            ak.mask(events.event_number, np.zeros(len(events), dtype=bool)),
            np.float32,
        )

    X_HH_discrim_name = find_matching_field(analysis_events, "X_HH", "discrim")
    if X_HH_discrim_name is not None:
        events[Spectators.EVENT_X_HH.value] = analysis_events[X_HH_discrim_name]
    else:
        events[Spectators.EVENT_X_HH.value] = ak.values_astype(
            ak.mask(events.event_number, np.zeros(len(events), dtype=bool)),
            np.float32,
        )

    out_fields = get_common(
        events.fields, [*label_names, *feature_names, *spectator_names]
    )
    if cutflow is None:
        return events[out_fields]
    else:
        return events[out_fields], cutflow
