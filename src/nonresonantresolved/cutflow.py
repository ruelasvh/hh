import awkward as ak
from shared.utils import logger
from .triggers import run3_all as triggers_run3_all
from .utils import (
    find_hist,
    find_all_hists,
    format_btagger_model_name,
    get_all_trigs_or,
)
from .selection import (
    select_n_jets_events,
    select_n_bjets,
    select_hc_jets,
    reconstruct_hh_mindeltar,
    select_X_Wt_events,
    select_hh_events,
    select_correct_hh_pair_events,
)
from src.nonresonantresolved.branches import (
    get_jet_branch_alias_names,
)
from src.nonresonantresolved.fillhists import (
    fill_jet_kin_histograms,
    fill_leading_jets_histograms,
    fill_reco_mH_histograms,
    fill_reco_mH_2d_histograms,
    fill_hh_deltaeta_histograms,
    fill_top_veto_histograms,
    fill_hh_mass_discrim_histograms,
)


def cut_flow(events, hists, luminosity_weight, config, args) -> None:
    """Fill histograms with data"""

    # print("Jets with truth H parents decorations")
    # events["jet_truth_H_parents"].show()

    logger.info("Filling histograms")
    logger.info("Initial Events: %s", len(events))

    all_trigs_or_decicions = get_all_trigs_or(events, triggers_run3_all)
    passed_trigs_or_events = events[all_trigs_or_decicions]
    logger.info(
        "Events passing the OR of all triggers: %s", len(passed_trigs_or_events)
    )

    event_selection = config["event_selection"]
    central_jets_selection = event_selection["central_jets"]
    baseline_events = select_n_jets_events(
        passed_trigs_or_events,
        jet_vars=get_jet_branch_alias_names(args.signal),
        pt_cut=central_jets_selection["min_pt"],
        eta_cut=central_jets_selection["max_eta"],
        njets_cut=central_jets_selection["min_count"],
    )
    logger.info(
        "Events with >= %s central jets and kinematic requirements pT > %s and |eta| < %s: %s",
        central_jets_selection["min_count"],
        central_jets_selection["min_pt"],
        central_jets_selection["max_eta"],
        len(baseline_events),
    )
    fill_jet_kin_histograms(baseline_events, hists, luminosity_weight)
    fill_leading_jets_histograms(baseline_events, hists)

    btagging_selection = event_selection["btagging"]
    btagger_model_name = format_btagger_model_name(
        btagging_selection["model"], btagging_selection["efficiency"]
    )
    signal_events = select_n_bjets(
        baseline_events,
        btag_cut="jet_btag_DL1dv01_77",
        nbjets_cut=4,
    )
    hc_jets, hc_jets_indices = select_hc_jets(
        signal_events,
        jet_vars=get_jet_branch_alias_names(args.signal),
        btag_cut=f"jet_btag_{btagger_model_name}",
        nbjets_cut=btagging_selection["min_count"],
    )
    logger.info(
        "Events with >= %s b-tagged central jets: %s",
        btagging_selection["min_count"],
        len(hc_jets),
    )

    # logger.info("Events with >= 6 central or forward jets", len(events_with_central_or_forward_jets))

    (
        h_leading,
        h_subleading,
        leading_h_jet_indices,
        subleading_h_jet_indices,
    ) = reconstruct_hh_mindeltar(hc_jets)

    fill_reco_mH_histograms(
        mh1=h_leading.m,
        mh2=h_subleading.m,
        hists=find_all_hists(hists, "mH[12]_baseline"),
    )
    fill_reco_mH_2d_histograms(
        mh1=h_leading.m,
        mh2=h_subleading.m,
        hist=find_hist(hists, lambda h: "mH_plane_baseline" in h.name),
    )
    # # WIP correctly paired Higgs bosons
    # correct_hh_pairs_from_truth = select_correct_hh_pair_events(
    #     signal_events, leading_h_jet_indices, subleading_h_jet_indices, args.signal
    # )
    # logger.info(
    #     "Events with correct HH pairs: %s",
    #     ak.sum(correct_hh_pairs_from_truth),
    # )

    hh_deltaeta_selection = event_selection["hh_deltaeta_veto"]["ggF"]
    (
        h1_events_with_deltaeta_cut,
        h2_events_with_deltaeta_cut,
        hh_deltar,
        hh_events_deltaeta_cut_keep_mask,
    ) = select_hh_events(
        h_leading, h_subleading, deltaeta_cut=hh_deltaeta_selection["max_value"]
    )
    logger.info(
        "Events with |deltaEta_HH| < %s: %s",
        hh_deltaeta_selection["max_value"],
        len(h1_events_with_deltaeta_cut),
    )
    fill_hh_deltaeta_histograms(
        hh_deltar, hists=find_all_hists(hists, "hh_deltaeta_baseline")
    )

    # calculate top veto discriminant
    top_veto_selection = event_selection["top_veto"]["ggF"]
    hc_jets_with_hh_deltar_cut_indices = hc_jets_indices[
        hh_events_deltaeta_cut_keep_mask
    ]
    events_with_hh_deltar_cut = signal_events[hh_events_deltaeta_cut_keep_mask]
    (
        top_veto_pass_events,
        top_veto_discrim,
        top_veto_events_keep_mask,
    ) = select_X_Wt_events(
        events_with_hh_deltar_cut,
        hc_jets_with_hh_deltar_cut_indices,
        discriminant_cut=top_veto_selection["min_value"],
    )
    fill_top_veto_histograms(
        top_veto_discrim, hists=find_all_hists(hists, "top_veto_baseline")
    )
    logger.info(
        "Events with top-veto discriminant > %s: %s",
        top_veto_selection["min_value"],
        len(top_veto_pass_events),
    )

    # calculate mass discriminant
    h1_events_with_top_veto_cut = (
        h1_events_with_deltaeta_cut
        if len(h1_events_with_deltaeta_cut) == 0
        else h1_events_with_deltaeta_cut[top_veto_events_keep_mask]
    )
    h2_events_with_top_veto_cut = (
        h2_events_with_deltaeta_cut
        if len(h2_events_with_deltaeta_cut) == 0
        else h2_events_with_deltaeta_cut[top_veto_events_keep_mask]
    )
    # signal region
    hh_mass_selection = event_selection["hh_mass_veto"]["ggF"]
    (
        h1_events_with_mass_discrim_cut,
        h2_events_with_mass_discrim_cut,
        hh_mass_discrim,
        hh_events_keep_mask,
    ) = select_hh_events(
        h1_events_with_top_veto_cut,
        h2_events_with_top_veto_cut,
        mass_discriminant_cut=hh_mass_selection["signal"]["max_value"],
    )
    logger.info(
        "Signal events with mass discriminant < %s: %s",
        hh_mass_selection["signal"]["max_value"],
        len(h1_events_with_mass_discrim_cut),
    )
    fill_hh_mass_discrim_histograms(
        hh_mass_discrim, hists=find_all_hists(hists, "hh_mass_discrim_baseline")
    )
    fill_reco_mH_histograms(
        mh1=h1_events_with_mass_discrim_cut.m,
        mh2=h2_events_with_mass_discrim_cut.m,
        hists=find_all_hists(hists, "mH[12]_baseline_signal_region"),
    )
    fill_reco_mH_2d_histograms(
        mh1=h1_events_with_mass_discrim_cut.m,
        mh2=h2_events_with_mass_discrim_cut.m,
        hist=find_hist(hists, lambda h: "mH_plane_baseline_signal_region" in h.name),
    )
    # control region
    (
        h1_events_control_region,
        h2_events_control_region,
        _,
        _,
    ) = select_hh_events(
        h1_events_with_top_veto_cut,
        h2_events_with_top_veto_cut,
        mass_discriminant_cut=(
            hh_mass_selection["signal"]["max_value"],
            hh_mass_selection["control"]["max_value"],
        ),
    )
    logger.info(
        "Control region events: %s",
        len(h1_events_control_region),
    )
    fill_reco_mH_histograms(
        mh1=h1_events_control_region.m,
        mh2=h2_events_control_region.m,
        hists=find_all_hists(hists, "mH[12]_baseline_control_region"),
    )
    fill_reco_mH_2d_histograms(
        mh1=h1_events_control_region.m,
        mh2=h2_events_control_region.m,
        hist=find_hist(hists, lambda h: "mH_plane_baseline_control_region" in h.name),
    )
    # if args.signal:
    #     fill_reco_mH_truth_pairing_histograms(events, hists)
    #     fill_truth_matched_mjj_histograms(events, hists)
    #     fill_truth_matched_mjj_passed_pairing_histograms(events, hists)
