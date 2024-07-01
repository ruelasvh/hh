import awkward as ak
import vector as p4
import numpy as np
from hh.shared.utils import (
    logger,
    format_btagger_model_name,
    kin_labels,
    GeV,
)
from hh.shared.selection import X_HH, R_CR, X_Wt
from hh.nonresonantresolved.pairing import pairing_methods
from hh.nonresonantresolved.selection import (
    select_n_jets_events,
    select_n_bjets_events,
    select_events_passing_triggers,
    select_truth_matched_jets,
    select_hh_jet_candidates,
    reconstruct_hh_jet_pairs,
    select_correct_hh_pair_events,
    select_hh_events,
)


def process_batch(
    events: ak.Record,
    selections: dict,
    sample_weight: float = 1.0,
    is_mc: bool = False,
) -> ak.Record:
    """Apply analysis regions selections and append info to events."""

    logger.info("Initial Events: %s", len(events))

    ## return early if selections is empty ##
    if not selections:
        logger.info("Selections empty. No objects selection applied.")
        return events

    ## set overall event filter, up to signal and background selections ##
    valid_event = np.ones(len(events), dtype=bool)

    ## set the total event weight ##
    events["event_weight"] = (
        np.prod([events.mc_event_weights[:, 0], events.pileup_weight], axis=0)
        if is_mc
        else np.ones(len(events), dtype=float)
    ) * sample_weight

    ## apply trigger selection ##
    if "trigs" in selections:
        trig_sel = selections["trigs"]
        trig_set, trig_op = (
            trig_sel.get("value"),
            trig_sel.get("operator"),
        )
        assert trig_set, "Invalid trigger set provided."
        events["passed_trigs"] = select_events_passing_triggers(events, op=trig_op)
        valid_event = valid_event & events.passed_trigs
        logger.info(
            "Events passing the %s of all triggers: %s",
            trig_op.upper() if trig_op is not None else "None",
            ak.sum(valid_event),
        )
        if ak.sum(valid_event) == 0:
            return events

    ## select and save events with >= n central jets ##
    if "central_jets" in selections:
        central_jets_sel = selections["central_jets"]
        jets = ak.zip({k: events[f"jet_{k}"] for k in ["jvttag", *kin_labels.keys()]})
        events["valid_central_jets"] = select_n_jets_events(
            jets=ak.mask(jets, events.passed_trigs),
            selection=central_jets_sel,
            do_jvt=False,
        )
        valid_event = valid_event & ~ak.is_none(events.valid_central_jets)
        logger.info(
            "Events passing previous cut and %s %s jets with pT %s %s, |eta| %s %s: %s",
            central_jets_sel["count"]["operator"],
            central_jets_sel["count"]["value"],
            central_jets_sel["pt"]["operator"],
            central_jets_sel["pt"]["value"],
            central_jets_sel["eta"]["operator"],
            central_jets_sel["eta"]["value"],
            ak.sum(valid_event),
        )
        if ak.sum(valid_event) == 0:
            return events

        if is_mc:
            events["reco_truth_matched_central_jets"] = select_truth_matched_jets(
                events.truth_jet_H_parent_mask != 0,
                # (events.truth_jet_H_parent_mask == 1)
                # | (events.truth_jet_H_parent_mask == 2),
                events.valid_central_jets,
            )
            logger.info(
                "Events passing previous cuts and 4 truth-matched central jets: %s",
                ak.sum(~ak.is_none(events.reco_truth_matched_central_jets, axis=0)),
            )
            ### jets truth matched with HadronConeExclTruthLabelID ###
            events["reco_truth_matched_central_jets_v2"] = select_truth_matched_jets(
                events.jet_truth_label_ID == 5, events.valid_central_jets
            )
            logger.info(
                "Events passing previous cuts and 4 truth-matched central jets using HadronConeExclTruthLabelID: %s",
                ak.sum(~ak.is_none(events.reco_truth_matched_central_jets_v2, axis=0)),
            )

    ## add jet and b-tagging info ##
    if "btagging" in selections:
        bjets_sel = selections["btagging"]
        if isinstance(bjets_sel, dict):
            bjets_sel = [bjets_sel]
        for i_bjets_sel in bjets_sel:
            btag_model = i_bjets_sel["model"]
            btag_eff = i_bjets_sel["efficiency"]
            btag_count = i_bjets_sel["count"]["value"]
            btag_op = i_bjets_sel["count"]["operator"]
            btagger = format_btagger_model_name(
                btag_model,
                btag_eff,
            )
            jet_btag_key = f"jet_btag_{btagger}"
            # select and save events with >= n central b-jets
            valid_central_btagged_jets = select_n_bjets_events(
                jets=(events.valid_central_jets & (events[jet_btag_key] == 1)),
                selection=i_bjets_sel,
            )
            events[f"valid_central_{btag_count}_btag_{btagger}_jets"] = (
                valid_central_btagged_jets
            )
            valid_event = valid_event & ~ak.is_none(valid_central_btagged_jets)
            logger.info(
                "Events passing previous cut and %s %s b-tags with %s and %s efficiency: %s",
                i_bjets_sel["count"]["operator"],
                i_bjets_sel["count"]["value"],
                i_bjets_sel["model"],
                i_bjets_sel["efficiency"],
                ak.sum(valid_event),
            )
            if ak.sum(valid_event) == 0:
                return events

            ### Do truth matching with b-tagging requirement ###
            if is_mc:
                reco_truth_matched_btagged_jets = select_truth_matched_jets(
                    events.truth_jet_H_parent_mask != 0,
                    # (events.truth_jet_H_parent_mask == 1)
                    # | (events.truth_jet_H_parent_mask == 2),
                    valid_central_btagged_jets,
                )
                events[
                    f"reco_truth_matched_central_{btag_count}_btag_{btagger}_jets"
                ] = reco_truth_matched_btagged_jets
                logger.info(
                    "Events passing previous cuts and 4 truth-matched central b-tagged jets with %s %s btags: %s",
                    i_bjets_sel["count"]["operator"],
                    i_bjets_sel["count"]["value"],
                    ak.sum(~ak.is_none(reco_truth_matched_btagged_jets, axis=0)),
                )
                ### jets truth matched with HadronConeExclTruthLabelID ###
                reco_truth_matched_btagged_jets_v2 = select_truth_matched_jets(
                    events.jet_truth_label_ID == 5,
                    valid_central_btagged_jets,
                )
                events[
                    f"reco_truth_matched_central_{btag_count}_btag_{btagger}_jets_v2"
                ] = reco_truth_matched_btagged_jets_v2
                logger.info(
                    "Events passing previous cuts and 4 truth-matched central b-tagged jets using HadronConeExclTruthLabelID: %s",
                    ak.sum(~ak.is_none(reco_truth_matched_btagged_jets_v2, axis=0)),
                )

            # ###################################
            # # select and save HH jet candidates
            # ###################################
            jets = {k: events[f"jet_{k}"] for k in kin_labels}
            jets["btag"] = events[jet_btag_key]
            jets = ak.zip(jets)
            jets_p4 = p4.zip({v: events[f"jet_{v}"] for v in kin_labels})
            hh_jet_idx, non_hh_jet_idx = select_hh_jet_candidates(
                jets=jets,
                valid_jets_mask=valid_central_btagged_jets,
            )
            events[f"hh_{btag_count}b_{btagger}_jet_idx"] = hh_jet_idx
            events[f"non_hh_{btag_count}b_{btagger}_jet_idx"] = non_hh_jet_idx
            ## select and save HH jet candidates that are truth-matched ##
            hh_truth_matched_jet_idx, non_hh_truth_matched_jet_idx = (
                select_hh_jet_candidates(
                    jets=jets,
                    valid_jets_mask=reco_truth_matched_btagged_jets,
                )
            )
            events[f"hh_{btag_count}b_{btagger}_truth_matched_jet_idx"] = (
                hh_truth_matched_jet_idx
            )
            events[f"non_hh_{btag_count}b_{btagger}_truth_matched_jet_idx"] = (
                non_hh_truth_matched_jet_idx
            )
            ############################################
            # Apply different HH jet candidate pairings
            ############################################
            ###### HH jet nominal pairing ######
            if is_mc:
                correct_hh_nominal_pairs_mask = select_correct_hh_pair_events(
                    h1_jets_idx=hh_truth_matched_jet_idx[:, 0:2],
                    h2_jets_idx=hh_truth_matched_jet_idx[:, 2:4],
                    truth_jet_H_parent_mask=events.truth_jet_H_parent_mask,
                )
                events[f"correct_hh_{btag_count}b_{btagger}_nominal_pairs_mask"] = (
                    correct_hh_nominal_pairs_mask
                )
                logger.info(
                    "Events passing previous cuts and correct H1 and H2 jet pairs with nominal pairing: %s",
                    ak.sum(correct_hh_nominal_pairs_mask),
                )
            for pairing, pairing_info in pairing_methods.items():
                ###### reconstruct H1 and H2 ######
                H1_jet_idx, H2_jet_idx = reconstruct_hh_jet_pairs(
                    jets=jets_p4,
                    hh_jet_idx=hh_jet_idx,
                    loss=pairing_info["loss"],
                    optimizer=pairing_info["optimizer"],
                )
                events[f"H1_{btag_count}b_{btagger}_{pairing}_jet_idx"] = H1_jet_idx
                events[f"H2_{btag_count}b_{btagger}_{pairing}_jet_idx"] = H2_jet_idx
                ###### Truth match the H1 and H2 ######
                H1_truth_matched_jet_idx, H2_truth_matched_jet_idx = (
                    reconstruct_hh_jet_pairs(
                        jets=jets_p4,
                        hh_jet_idx=hh_truth_matched_jet_idx,
                        loss=pairing_info["loss"],
                        optimizer=pairing_info["optimizer"],
                    )
                )
                events[
                    f"H1_{btag_count}b_{btagger}_{pairing}_truth_matched_jet_idx"
                ] = H1_truth_matched_jet_idx
                events[
                    f"H2_{btag_count}b_{btagger}_{pairing}_truth_matched_jet_idx"
                ] = H2_truth_matched_jet_idx
                correct_hh_pairs_mask = select_correct_hh_pair_events(
                    h1_jets_idx=H1_truth_matched_jet_idx,
                    h2_jets_idx=H2_truth_matched_jet_idx,
                    truth_jet_H_parent_mask=events.truth_jet_H_parent_mask,
                )
                events[f"correct_hh_{btag_count}b_{btagger}_{pairing}_mask"] = (
                    correct_hh_pairs_mask
                )
                logger.info(
                    "Events passing previous cuts and correct H1 and H2 jet pairs with %s: %s",
                    pairing.replace("_", " "),
                    ak.sum(correct_hh_pairs_mask),
                )
                #################################################
                # Perform event selection for each pairing method
                #################################################
                regions = ["signal", "control"]
                ###### X_HH mass veto ######
                h1_p4 = ak.sum(jets_p4[H1_jet_idx], axis=1)
                h2_p4 = ak.sum(jets_p4[H2_jet_idx], axis=1)
                if "hh_mass_discriminat" in selections:
                    X_HH_discrim = X_HH(h1_p4.m * GeV, h2_p4.m * GeV)
                    events[f"X_HH_{btag_count}b_{btagger}_{pairing}_discrim"] = (
                        X_HH_discrim
                    )
                    R_CR_discrim = R_CR(h1_p4.m * GeV, h2_p4.m * GeV)
                    events[f"R_CR_{btag_count}b_{btagger}_{pairing}_discrim"] = (
                        R_CR_discrim
                    )
                    hh_mass_discrim_sel = selections["hh_mass_discriminat"]
                    # for region in regions:
                    #     if region in hh_mass_discrim_sel:
                    #         X_HH_mask, X_HH_discrim = select_hh_events(
                    #             jets_p4,
                    #             H1_jet_idx,
                    #             H2_jet_idx,
                    #             mass_sel=hh_mass_discrim_sel[region],
                    #         )
                    #         events[
                    #             f"X_HH_{region}_{btag_count}b_{btagger}_{pairing}_mask"
                    #         ] = X_HH_mask
                    #         events[
                    #             f"X_HH_{region}_{btag_count}b_{btagger}_{pairing}_discrim"
                    #         ] = X_HH_discrim
                    #         logger.info(
                    #             "Events passing previous cuts and X_HH veto for %s region with %s pairing: %s",
                    #             region,
                    #             pairing.replace("_", " "),
                    #             ak.sum(X_HH_mask),
                    #         )
                    for region in regions:
                        if region in hh_mass_discrim_sel:
                            X_HH_mask = select_hh_events(
                                (X_HH_discrim, R_CR_discrim),
                                mass_sel=hh_mass_discrim_sel[region],
                            )
                            events[
                                f"X_HH_{region}_{btag_count}b_{btagger}_{pairing}_mask"
                            ] = X_HH_mask
                            logger.info(
                                "Events passing previous cuts and X_HH veto for %s region with %s pairing: %s",
                                region,
                                pairing.replace("_", " "),
                                ak.sum(X_HH_mask),
                            )

    return events
