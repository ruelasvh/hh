import awkward as ak
import numpy as np
import vector
from hh.shared.utils import (
    logger,
    inv_GeV,
    make_4jet_comb_array,
    format_btagger_model_name,
    get_common,
    kin_labels,
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

vector.register_awkward()


def get_jet_p4(events):
    return ak.zip({k: events[f"jet_{k}"] for k in kin_labels}, with_name="Momentum4D")


def process_batch(
    events: ak.Record,
    selections: dict,
    outputs: dict,
    class_label: str,
    sample_weight: float = 1.0,
    is_mc: bool = True,
) -> ak.Record:
    """Apply analysis regions selection and append info to events."""

    logger.info("Initial Events: %s", len(events))

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
        events[Features.MC_EVENT_WEIGHT.value] = events.mc_event_weights[:, 0]
        events[Features.EVENT_WEIGHT.value] = (
            np.prod([events.mc_event_weight, events.pileup_weight], axis=0)
            * sample_weight
        )

    # start adding jet features
    events[Features.JET_NUM.value] = ak.num(events.jet_pt, axis=-1)
    # build 4-momentum vectors for jets
    j4 = get_jet_p4(events)
    # convert jets to cartesian coordinates and save them
    for f, v in zip(
        [Features.JET_X, Features.JET_Y, Features.JET_Z], [j4.px, j4.py, j4.pz]
    ):
        events[f.value] = v

    # check if selections is empty (i.e. no selection)
    if not selections:
        logger.info("No objects selection applied.")
        features_out = get_common(events.fields, feature_names)
        return events[[*features_out, *label_names]]

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
            events = events[passed_trigs_mask]
            logger.info(
                "Events passing the %s of all triggers: %s",
                trig_op.upper(),
                len(events),
            )
            if len(events) == 0:
                features_out = get_common(events.fields, feature_names)
                return events[[*features_out, *label_names]]

    # get jet selections
    if "jets" in selections:
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
        events = events[n_jets_event_mask]
        logger.info(
            "Events passing previous cuts and jets selection: %s",
            len(events),
        )
        if len(events) == 0:
            features_out = get_common(events.fields, feature_names)
            return events[[*features_out, *label_names]]

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
            n_bjets_mask, n_bjets_event_mask = select_n_bjets_events(
                jets=ak.zip(
                    {
                        "btag": events[Features.JET_BTAG.value],
                        "valid": n_jets_mask[n_jets_event_mask],
                    }
                ),
                selection=bjet_selection,
            )
            events = events[n_bjets_event_mask]
            logger.info(
                "Events passing previous cuts and b-jets selection: %s",
                len(events),
            )
            if len(events) == 0:
                features_out = get_common(events.fields, feature_names)
                return events[[*features_out, *label_names]]

            # select and save hc jets
            hh_c_jets = select_hc_jets(
                jets=ak.zip(
                    {
                        "pt": events.jet_pt,
                        "btag": events[Features.JET_BTAG.value],
                        "valid": n_bjets_mask[n_bjets_event_mask],
                    }
                )
            )
            events["hc_jet_idx"] = hh_c_jets[0]
            events["non_hc_jet_idx"] = hh_c_jets[1]
            j4 = get_jet_p4(events)
            four_bjets_p4 = j4[events.hc_jet_idx]
            events[Features.EVENT_M_4B.value] = (
                four_bjets_p4[:, 0]
                + four_bjets_p4[:, 1]
                + four_bjets_p4[:, 2]
                + four_bjets_p4[:, 3]
            ).mass
            # calculate bb features
            if Features.EVENT_BB_RMH.value in feature_names:
                events[Features.EVENT_BB_RMH.value] = (
                    make_4jet_comb_array(four_bjets_p4, lambda x, y: (x + y).mass)
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
            hh_jet = reconstruct_hh_mindeltar(
                jets=ak.zip(
                    {
                        "pt": events.jet_pt,
                        "eta": events.jet_eta,
                        "phi": events.jet_phi,
                        "mass": events.jet_mass,
                    }
                ),
                hc_jet_idx=events.hc_jet_idx,
            )
            events["leading_h_jet_idx"] = hh_jet[0]
            events["subleading_h_jet_idx"] = hh_jet[1]

            # correctly paired Higgs bosons to further clean up labels
            if is_mc:
                correct_hh_pairs_from_truth = select_correct_hh_pair_events(events)
                events = events[correct_hh_pairs_from_truth]
                logger.info(
                    "Events passing previous cuts and truth-matched to HH: %s",
                    len(events),
                )
                if len(events) == 0:
                    features_out = get_common(events.fields, feature_names)
                    return events[[*features_out, *label_names]]

            # calculate X_Wt
            if Features.EVENT_X_WT.value in feature_names:
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
                    events.hc_jet_idx,
                    events.non_hc_jet_idx,
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
                events.leading_h_jet_idx[:, 0, np.newaxis],
                events.leading_h_jet_idx[:, 1, np.newaxis],
            )
            h2_jet1_idx, h2_jet2_idx = (
                events.subleading_h_jet_idx[:, 0, np.newaxis],
                events.subleading_h_jet_idx[:, 1, np.newaxis],
            )
            j4 = get_jet_p4(events)
            h1 = j4[h1_jet1_idx] + j4[h1_jet2_idx]
            h2 = j4[h2_jet1_idx] + j4[h2_jet2_idx]
            if Features.EVENT_DELTAETA_HH.value in feature_names:
                events[Features.EVENT_DELTAETA_HH.value] = np.abs(
                    ak.firsts(h1.eta) - ak.firsts(h2.eta)
                )
            if Features.EVENT_X_HH.value in feature_names:
                events[Features.EVENT_X_HH.value] = X_HH(
                    ak.firsts(h1.m) * inv_GeV, ak.firsts(h2.m) * inv_GeV
                )

    features_out = get_common(events.fields, feature_names)
    return events[[*features_out, *label_names]]
