import awkward as ak
import vector as p4
import numpy as np
import itertools
from hh.shared.utils import (
    logger,
    format_btagger_model_name,
    kin_labels,
    optimizer_mass_pairing,
)
from hh.nonresonantresolved.selection import (
    select_n_jets_events,
    select_n_bjets_events,
    select_events_passing_triggers,
    select_truth_matched_jets,
    select_hh_jet_candidates,
    reconstruct_hh_jet_pairs,
    select_correct_hh_pair_events,
)


def process_batch(
    events: ak.Record,
    selections: dict,
    sample_weight: float = 1.0,
    is_mc: bool = False,
) -> ak.Record:
    """Apply analysis regions selections and append info to events."""

    logger.info("Initial Events: %s", len(events))

    # return early if selections is empty
    if not selections:
        logger.info("Selections empty. No objects selection applied.")
        return events

    # set overall event filter, up to signal and background selections
    events["valid_event"] = np.ones(len(events), dtype=bool)

    # set the total event weight
    events["event_weight"] = (
        np.prod([events.mc_event_weights[:, 0], events.pileup_weight], axis=0)
        if is_mc
        else np.ones(len(events), dtype=float)
    ) * sample_weight

    # apply trigger selection
    if "trigs" in selections:
        trig_sel = selections["trigs"]
        trig_set, trig_op = (
            trig_sel.get("value"),
            trig_sel.get("operator"),
        )
        assert trig_set, "Invalid trigger set provided."
        passed_trigs_mask = select_events_passing_triggers(events, op=trig_op)
        events["valid_event"] = events.valid_event & passed_trigs_mask
        logger.info(
            "Events passing the %s of all triggers: %s (from %s)",
            trig_op.upper() if trig_op is not None else "None",
            ak.sum(events.valid_event),
            ak.sum(passed_trigs_mask),
        )
        if ak.sum(events.valid_event) == 0:
            return events

    # select and save events with >= n central jets
    if "central_jets" in selections:
        central_jets_sel = selections["central_jets"]
        valid_central_jets = select_n_jets_events(
            jets=ak.zip(
                {k: events[f"jet_{k}"] for k in ["jvttag", *kin_labels.keys()]}
            ),
            selection=central_jets_sel,
            do_jvt=True,
        )
        events["valid_central_jets"] = valid_central_jets
        events["valid_event"] = events.valid_event & ~ak.is_none(valid_central_jets)
        logger.info(
            "Events passing previous cut and %s %s central jets with pT %s %s, |eta| %s %s and Jvt tag: %s (from %s)",
            central_jets_sel["count"]["operator"],
            central_jets_sel["count"]["value"],
            central_jets_sel["pt"]["operator"],
            central_jets_sel["pt"]["value"],
            central_jets_sel["eta"]["operator"],
            central_jets_sel["eta"]["value"],
            ak.sum(events.valid_event),
            ak.sum(~ak.is_none(valid_central_jets)),
        )
        if ak.sum(events.valid_event) == 0:
            return events

        if is_mc:
            events["reco_truth_matched_jets"] = select_truth_matched_jets(
                events.truth_jet_H_parent_mask != 0,
                # (events.truth_jet_H_parent_mask == 1)
                # | (events.truth_jet_H_parent_mask == 2),
                events.valid_central_jets,
            )
            logger.info(
                "Events passing previous cuts and truth matching: %s",
                ak.sum(~ak.is_none(events.reco_truth_matched_jets, axis=0)),
            )
            ### jets truth matched with HadronConeExclTruthLabelID ###
            events["reco_truth_matched_jets_v2"] = select_truth_matched_jets(
                events.jet_truth_label_ID == 5, events.valid_central_jets
            )
            logger.info(
                "Events passing previous cuts and truth matching using HadronConeExclTruthLabelID: %s",
                ak.sum(~ak.is_none(events.reco_truth_matched_jets_v2, axis=0)),
            )

    # add jet and b-tagging info
    if "btagging" in selections:
        bjets_sel = selections["btagging"]
        btagger = format_btagger_model_name(
            bjets_sel["model"],
            bjets_sel["efficiency"],
        )
        events["jet_btag"] = events[f"jet_btag_{btagger}"]
        # select and save events with >= n central b-jets
        valid_central_btagged_jets = select_n_bjets_events(
            jets=(events.valid_central_jets & (events.jet_btag == 1)),
            selection=bjets_sel,
        )
        events["valid_central_btagged_jets"] = valid_central_btagged_jets
        events["valid_event"] = events.valid_event & ~ak.is_none(
            valid_central_btagged_jets
        )
        logger.info(
            "Events passing previous cut and %s %s b-jets tagged with %s and %s efficiency: %s (from total %s)",
            bjets_sel["count"]["operator"],
            bjets_sel["count"]["value"],
            bjets_sel["model"],
            bjets_sel["efficiency"],
            ak.sum(events.valid_event),
            ak.sum(~ak.is_none(valid_central_btagged_jets)),
        )
        if ak.sum(events.valid_event) == 0:
            return events
        ### Do truth matching with b-tagging requirement ###
        if is_mc:
            events["reco_truth_matched_btagged_jets"] = select_truth_matched_jets(
                events.truth_jet_H_parent_mask != 0,
                # (events.truth_jet_H_parent_mask == 1)
                # | (events.truth_jet_H_parent_mask == 2),
                events.valid_central_btagged_jets,
            )
            logger.info(
                "Events passing previous cuts and truth matched with %s %s btags: %s",
                bjets_sel["count"]["operator"],
                bjets_sel["count"]["value"],
                ak.sum(~ak.is_none(events.reco_truth_matched_btagged_jets, axis=0)),
            )

        ### baseline for 4 b-tagged jets ###
        bjets_sel_4_btags = {**bjets_sel, "count": {"operator": ">=", "value": 4}}
        valid_central_4_btagged_jets = select_n_bjets_events(
            jets=(events.valid_central_jets & (events.jet_btag == 1)),
            selection=bjets_sel_4_btags,
        )
        events["valid_central_4_btagged_jets"] = valid_central_4_btagged_jets
        logger.info(
            "Events passing previous cut and %s %s b-jets tagged with %s and %s efficiency: %s (from total %s)",
            bjets_sel_4_btags["count"]["operator"],
            bjets_sel_4_btags["count"]["value"],
            bjets_sel_4_btags["model"],
            bjets_sel_4_btags["efficiency"],
            ak.sum(events.valid_event & ~ak.is_none(valid_central_4_btagged_jets)),
            ak.sum(~ak.is_none(valid_central_4_btagged_jets)),
        )
        ### Do truth matching with b-tagging requirement ###
        if is_mc:
            ### baseline 4 b-tagged jets ###
            events["reco_truth_matched_4_btagged_jets"] = select_truth_matched_jets(
                events.truth_jet_H_parent_mask != 0,
                # (events.truth_jet_H_parent_mask == 1)
                # | (events.truth_jet_H_parent_mask == 2),
                events.valid_central_4_btagged_jets,
            )
            logger.info(
                "Events passing previous cuts and truth matched with %s %s btags: %s",
                bjets_sel_4_btags["count"]["operator"],
                bjets_sel_4_btags["count"]["value"],
                ak.sum(~ak.is_none(events.reco_truth_matched_4_btagged_jets, axis=0)),
            )

    # select and save HH jet candidates
    events["hh_jet_idx"], events["non_hh_jet_idx"] = select_hh_jet_candidates(
        jets=ak.zip({k: events[f"jet_{k}"] for k in ["btag", *kin_labels.keys()]}),
        valid_jets_mask=events.valid_central_4_btagged_jets,
    )

    ######
    # Apply different HH jet candidate pairings
    ######
    ###### HH jet min deltaR pairing ######
    events["H1_min_dR_jet_idx"], events["H2_min_dR_jet_idx"] = reconstruct_hh_jet_pairs(
        jets=p4.zip({v: events[f"jet_{v}"] for v in kin_labels}),
        hh_jet_idx=events.hh_jet_idx,
        loss=lambda j_1, j_2: j_1.deltaR(j_2),
    )
    events["correct_hh_min_dR_pairs_mask"] = select_correct_hh_pair_events(
        h1_jets_idx=events.H1_min_dR_jet_idx,
        h2_jets_idx=events.H2_min_dR_jet_idx,
        truth_jet_H_parent_mask=events.truth_jet_H_parent_mask,
    )
    logger.info(
        "Events passing previous cuts and correct HH jet pairing with min deltaR: %s",
        ak.sum(events.correct_hh_min_dR_pairs_mask),
    )

    ###### HH jet max deltaR pairing ######
    events["H1_max_dR_jet_idx"], events["H2_max_dR_jet_idx"] = reconstruct_hh_jet_pairs(
        jets=p4.zip({v: events[f"jet_{v}"] for v in kin_labels}),
        hh_jet_idx=events.hh_jet_idx,
        loss=lambda j_1, j_2: j_1.deltaR(j_2),
        optimizer=np.argmax,
    )
    events["correct_hh_max_dR_pairs_mask"] = select_correct_hh_pair_events(
        h1_jets_idx=events.H1_max_dR_jet_idx,
        h2_jets_idx=events.H2_max_dR_jet_idx,
        truth_jet_H_parent_mask=events.truth_jet_H_parent_mask,
    )
    logger.info(
        "Events passing previous cuts and correct HH jet pairing with max deltaR: %s",
        ak.sum(events.correct_hh_max_dR_pairs_mask),
    )

    events["H1_min_mass_jet_idx"], events["H2_min_mass_jet_idx"] = (
        reconstruct_hh_jet_pairs(
            jets=p4.zip({v: events[f"jet_{v}"] for v in kin_labels}),
            hh_jet_idx=events.hh_jet_idx,
            loss=lambda j_1, j_2: ((j_1 + j_2).mass - 125) ** 2,
            optimizer=optimizer_mass_pairing,
        )
    )
    events["correct_hh_min_mass_pairs_mask"] = select_correct_hh_pair_events(
        h1_jets_idx=events.H1_min_mass_jet_idx,
        h2_jets_idx=events.H2_min_mass_jet_idx,
        truth_jet_H_parent_mask=events.truth_jet_H_parent_mask,
    )
    logger.info(
        "Events passing previous cuts and correct HH jet pairing with min deviation from Higgs mass: %s",
        ak.sum(events.correct_hh_min_mass_pairs_mask),
    )
    return events
