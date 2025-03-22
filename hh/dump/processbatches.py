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
    truncate_jets,
)
from hh.shared.labels import kin_labels
from hh.dump.output import OutputVariables
from hh.nonresonantresolved.selection import (
    select_events_passing_triggers,
    select_n_jets_events,
    select_n_bjets_events,
    select_hh_jet_candidates,
)
from hh.nonresonantresolved.processbatches import (
    process_batch as analysis_process_batch,
)

vector.register_awkward()


def process_batch(
    events: ak.Record,
    selections: dict,
    output: dict,
    class_label: str,
    sample_weight: float = 1.0,
    is_mc: bool = True,
    year: int = None,
) -> ak.Record:
    """Apply analysis regions selection and append info to events."""

    # get features and class names to be saved
    output_variable_names = output["variables"]
    output_label_names = output["labels"]
    train_selections = selections["training"]
    analysis_selections = selections["analysis"]
    cutflow = {}

    events[OutputVariables.EVENT_NUMBER.value] = events.event_number
    events[OutputVariables.YEAR.value] = np.ones(len(events), dtype=int) * year

    # append output_label_names to events and set them to 0 or 1
    for class_name in output_label_names:
        if class_name == class_label:
            events[OutputVariables(class_name).value] = np.ones(len(events))
        else:
            events[OutputVariables(class_name).value] = np.zeros(len(events))

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
    events[OutputVariables.EVENT_WEIGHT.value] = events["event_weight"]

    logger.info(
        "Initial events: %s (weighted: %s)", len(events), ak.sum(events.event_weight)
    )
    cutname = "initial_events"
    update_cutflow(cutflow, cutname, np.ones(len(events)), events.event_weight)

    # start adding jet features
    events[OutputVariables.N_JETS.value] = ak.num(events.jet_pt, axis=-1)
    # build 4-momentum vectors for jets
    jets_p4 = ak.zip(
        {k: events[f"jet_{k}"] for k in ["jvttag", *kin_labels.keys()]},
        with_name="Momentum4D",
    )
    # save jet kinematics
    for f, v in zip(
        [
            OutputVariables.JET_PT,
            OutputVariables.JET_ETA,
            OutputVariables.JET_PHI,
            OutputVariables.JET_MASS,
        ],
        [jets_p4.pt, jets_p4.eta, jets_p4.phi, jets_p4.mass],
    ):
        events[f.value] = v
    # convert jets to cartesian coordinates and save them
    for f, v in zip(
        [OutputVariables.JET_PX, OutputVariables.JET_PY, OutputVariables.JET_PZ],
        [jets_p4.px, jets_p4.py, jets_p4.pz],
    ):
        events[f.value] = v

    # check if selections is empty (i.e. no selection)
    if not selections:
        logger.info("No objects selection applied.")
        out_fields = get_common(
            events.fields, [*output_variable_names, *output_label_names]
        )
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
            "Possible operators: AND, OR."
        )
        trigs = trig_set[str(year)]
        # select and save events passing the triggers
        passed_trigs_mask = select_events_passing_triggers(
            events, triggers=[f"trig_{trig}" for trig in trigs], operator=trig_op
        )
        events = events[passed_trigs_mask]
        logger.info(
            "Events passing the %s of triggers %s: %s (weighted: %s)",
            trig_op.upper(),
            list(trigs.values()),
            len(events),
            ak.sum(events.event_weight),
        )
        cutname = "_".join(["triggers", f"pass_{trig_op.lower()}"]).replace(".", "p")
        update_cutflow(cutflow, cutname, passed_trigs_mask, events.event_weight)
        if len(events) == 0:
            out_fields = get_common(
                events.fields, [*output_variable_names, *output_label_names]
            )
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
                events.fields, [*output_variable_names, *output_label_names]
            )
            return events[out_fields], cutflow

        if "btagging" in jet_selection:
            bjet_selection = jet_selection["btagging"]
            btagger = format_btagger_model_name(bjet_selection["model"])
            if "efficiency" in bjet_selection:
                btagger = format_btagger_model_name(
                    btagger, bjet_selection["efficiency"]
                )
                events[OutputVariables.N_BTAGS.value] = ak.sum(
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
                        events.fields, [*output_variable_names, *output_label_names]
                    )
                    return events[out_fields], cutflow

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
                events[OutputVariables.M_4B.value] = (
                    ak.sum(four_bjets_p4, axis=1).mass * GeV
                )
                events[OutputVariables.PT_4B.value] = (
                    ak.sum(four_bjets_p4, axis=1, keepdims=True).pt * GeV
                )
                events[OutputVariables.ETA_4B.value] = ak.sum(
                    four_bjets_p4, axis=1, keepdims=True
                ).eta
                events[OutputVariables.PHI_4B.value] = ak.sum(
                    four_bjets_p4, axis=1, keepdims=True
                ).phi
                # calculate bb features
                if OutputVariables.BB_DM.value in output_variable_names:
                    events[OutputVariables.BB_DM.value] = (
                        make_4jet_comb_array(four_bjets_p4, lambda x, y: (x + y).mass)
                        * GeV
                    )
                if OutputVariables.BB_DR.value in output_variable_names:
                    events[OutputVariables.BB_DR.value] = make_4jet_comb_array(
                        four_bjets_p4, lambda x, y: x.deltaR(y)
                    )
                if OutputVariables.BB_DETA.value in output_variable_names:
                    events[OutputVariables.BB_DETA.value] = make_4jet_comb_array(
                        four_bjets_p4, lambda x, y: abs(x.eta - y.eta)
                    )
            else:
                # Form Higgs candidates from jets with highest b-tagging probabilities
                highest_btag_jets = ak.argsort(
                    events[f"jet_btag_{btagger}_pb"], axis=1, ascending=False
                )[:, :4]
                jets_p4 = ak.zip(
                    {
                        **{k: events[f"jet_{k}"] for k in kin_labels},
                    },
                    with_name="Momentum4D",
                )
                four_bjets_p4 = jets_p4[highest_btag_jets]
                events[OutputVariables.M_4B.value] = (
                    ak.sum(four_bjets_p4, axis=1).mass * GeV
                )
                events[OutputVariables.PT_4B.value] = (
                    ak.sum(four_bjets_p4, axis=1, keepdims=True).pt * GeV
                )
                events[OutputVariables.ETA_4B.value] = ak.sum(
                    four_bjets_p4, axis=1, keepdims=True
                ).eta
                events[OutputVariables.PHI_4B.value] = ak.sum(
                    four_bjets_p4, axis=1, keepdims=True
                ).phi
                # calculate bb features
                if OutputVariables.BB_DM.value in output_variable_names:
                    events[OutputVariables.BB_DM.value] = (
                        make_4jet_comb_array(four_bjets_p4, lambda x, y: (x + y).mass)
                        * GeV
                    )
                if OutputVariables.BB_DR.value in output_variable_names:
                    events[OutputVariables.BB_DR.value] = make_4jet_comb_array(
                        four_bjets_p4, lambda x, y: x.deltaR(y)
                    )
                if OutputVariables.BB_DETA.value in output_variable_names:
                    events[OutputVariables.BB_DETA.value] = make_4jet_comb_array(
                        four_bjets_p4, lambda x, y: abs(x.eta - y.eta)
                    )

    ############################################
    # Analysis selections
    ############################################
    analysis_events, analysis_cutflow = analysis_process_batch(
        events[events.fields],
        analysis_selections,
        is_mc=is_mc,
        year=year,
        return_cutflow=True,
    )
    cutflow.update(analysis_cutflow)

    analysis_bjet_selection = analysis_selections["jets"]["btagging"]
    analysis_btagger = format_btagger_model_name(
        analysis_bjet_selection["model"], analysis_bjet_selection["efficiency"]
    )
    analysis_btag_count = analysis_bjet_selection["count"]["value"]

    # TODO: Make this dynamic to handle all the possible pairing options
    H1_reco_p4_name = find_matching_field(
        analysis_events, f"H1_{analysis_btag_count}btags_{analysis_btagger}", "p4"
    )
    if H1_reco_p4_name is not None:
        for var in kin_labels:
            events[OutputVariables[f"H1_RECO_{var.upper()}"].value] = getattr(
                analysis_events[H1_reco_p4_name], var
            )
    else:
        for var in kin_labels:
            events[OutputVariables[f"H1_RECO_{var.upper()}"].value] = ak.values_astype(
                ak.mask(events.event_number, np.zeros(len(events), dtype=bool)),
                np.float32,
            )

    H2_reco_p4_name = find_matching_field(
        analysis_events, f"H2_{analysis_btag_count}btags_{analysis_btagger}", "p4"
    )
    if H2_reco_p4_name is not None:
        for var in kin_labels:
            events[OutputVariables[f"H2_RECO_{var.upper()}"].value] = getattr(
                analysis_events[H2_reco_p4_name], var
            )
    else:
        for var in kin_labels:
            events[OutputVariables[f"H2_RECO_{var.upper()}"].value] = ak.values_astype(
                ak.mask(events.event_number, np.zeros(len(events), dtype=bool)),
                np.float32,
            )

    HH_reco_p4_name = find_matching_field(
        analysis_events, f"H2_{analysis_btag_count}btags_{analysis_btagger}", "p4"
    )
    if HH_reco_p4_name is not None:
        for var in kin_labels:
            events[OutputVariables[f"HH_RECO_{var.upper()}"].value] = getattr(
                analysis_events[HH_reco_p4_name], var
            )
    else:
        for var in kin_labels:
            events[OutputVariables[f"HH_RECO_{var.upper()}"].value] = ak.values_astype(
                ak.mask(events.event_number, np.zeros(len(events), dtype=bool)),
                np.float32,
            )

    X_Wt_discriminant_name = find_matching_field(analysis_events, "X_Wt", "discrim")
    if X_Wt_discriminant_name is not None:
        events[OutputVariables.X_WT.value] = analysis_events[X_Wt_discriminant_name]
    else:
        events[OutputVariables.X_WT.value] = ak.values_astype(
            ak.mask(events.event_number, np.zeros(len(events), dtype=bool)),
            np.float32,
        )

    delta_eta_HH_name = find_matching_field(analysis_events, "deltaeta_HH", "discrim")
    if delta_eta_HH_name is not None:
        events[OutputVariables.DELTAETA_HH.value] = analysis_events[delta_eta_HH_name]
    else:
        events[OutputVariables.DELTAETA_HH.value] = ak.values_astype(
            ak.mask(events.event_number, np.zeros(len(events), dtype=bool)),
            np.float32,
        )

    X_HH_discrim_name = find_matching_field(analysis_events, "X_HH", "discrim")
    if X_HH_discrim_name is not None:
        events[OutputVariables.X_HH.value] = analysis_events[X_HH_discrim_name]
    else:
        events[OutputVariables.X_HH.value] = ak.values_astype(
            ak.mask(events.event_number, np.zeros(len(events), dtype=bool)),
            np.float32,
        )

    max_jets = output["max_jets"]
    pad_value = output["pad_value"]
    if max_jets > 0:
        jet_field_names = [f for f in events.fields if f.startswith("jet_")]
        for f in jet_field_names:
            events[f] = truncate_jets(events[f], max_jets, pad_value)

    out_fields = get_common(
        events.fields, [*output_variable_names, *output_label_names]
    )

    return events[out_fields], cutflow
