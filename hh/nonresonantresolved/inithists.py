from argparse import Namespace
from hh.nonresonantresolved.pairing import pairing_methods
from hh.shared.utils import (
    logger,
    format_btagger_model_name,
)
from hh.shared.labels import kin_labels
from hh.shared.hist import (
    Histogram,
    HistogramDynamic,
)


def init_hists(inputs: dict, selections: dict, args: Namespace) -> dict:
    """Initialize histograms for the different studies using the shape of inputs."""

    logger.info("Initializing hisotgrams")
    hists_dict = {}
    for sample in inputs:
        sample_name = sample["label"]
        hists_dict[sample_name] = {}
        hh_vars_binranges = {
            "pt": [20_000, 1_000_000],
            "eta": [-5, 5],
            "phi": [-3, 3],
            "mass": [20_000, 1_000_000],
            "sum_jet_pt": [20_000, 1_000_000],
            "delta_eta": [-5, 5],
        }

        ##################################################
        # Leading jet pT histograms ######################
        ##################################################
        hists_dict[sample_name].update(
            init_leading_jets_histograms(
                prefix="leading_truth_jet",
                binrange={"pt": [0, 6_000_000]},
                n_leading=1,
            )
        )

        ##################################################
        # Number of true bjets per event histograms ######
        ##################################################
        hists_dict[sample_name].update(
            init_n_true_bjet_composition_histograms(binrange=[2, 5])
        )

        ##################################################
        ### Truth H(H) histograms ########################
        ##################################################
        hists_dict[sample_name].update(init_H_histograms(postfix="_truth"))
        hists_dict[sample_name].update(init_HH_histograms(postfix="_truth"))

        # ##################################################
        # ### Reco H(H) histograms ########################
        # ##################################################
        # hists_dict[sample_name].update(
        #     init_HH_histograms(postfix="_truth_reco_central_jets_selection")
        # )
        # hists_dict[sample_name].update(
        #     init_HH_histograms(
        #         postfix="_truth_reco_central_truth_matched_jets_selection"
        #     )
        # )

        ##################################################
        ### Reco truth-matched H(H) histograms ###########
        ##################################################
        hists_dict[sample_name].update(
            init_HH_histograms(postfix="_reco_truth_matched")
        )
        hists_dict[sample_name].update(
            init_HH_histograms(postfix="_truth_reco_matched")
        )
        hists_dict[sample_name].update(
            init_HH_histograms(postfix="_reco_truth_matched_v2")
        )
        hists_dict[sample_name].update(
            init_HH_histograms(
                postfix="_reco_vs_truth_response",
                binrange={v: [-100, 100] for v in kin_labels},
            )
        )

        # ######################################################
        # ### HH jet candidate selections cutflow histograms ###
        # ######################################################
        # hists_dict[sample_name].update(
        #     init_leading_jets_histograms(prefix="hh_jet", postfix="_truth_matched")
        # )
        # hists_dict[sample_name].update(
        #     init_leading_jets_histograms(
        #         prefix="hh_jet", postfix="_truth_matched_4btags"
        #     )
        # )
        # hists_dict[sample_name].update(
        #     init_leading_jets_histograms(
        #         prefix="hh_jet", postfix="_truth_matched_2b2j_asym"
        #     )
        # )
        # hists_dict[sample_name].update(
        #     init_leading_jets_histograms(
        #         prefix="hh_jet", postfix="_truth_matched_2b2j_asym_nbtags"
        #     )
        # )
        # hists_dict[sample_name].update(
        #     init_leading_jets_histograms(
        #         prefix="hh_jet", postfix="_truth_matched_2b2j_asym_4btags"
        #     )
        # )

        if "jets" in selections and "btagging" in selections["jets"]:
            bjets_sel = selections["jets"]["btagging"]
            if isinstance(bjets_sel, dict):
                bjets_sel = [bjets_sel]
            for i_bjets_sel in bjets_sel:
                btag_model = i_bjets_sel["model"]
                btag_eff = i_bjets_sel["efficiency"]
                btag_count = i_bjets_sel["count"]["value"]
                btagger = format_btagger_model_name(
                    btag_model,
                    btag_eff,
                )
                hists_dict[sample_name].update(
                    init_leading_jets_histograms(
                        prefix=f"leading_resolved_{btag_count}btags_{btagger}_reco_jet",
                        binrange={"pt": [0, 6_000_000]},
                        n_leading=1,
                    )
                )
                ### HH histograms ########################
                hists_dict[sample_name].update(
                    init_HH_histograms(
                        postfix=f"_truth_reco_central_{btag_count}btags_{btagger}_jets_selection"
                    )
                )
                hists_dict[sample_name].update(
                    init_HH_histograms(
                        postfix=f"_truth_reco_central_{btag_count}btags_{btagger}_4_plus_truth_matched_jets_selection"
                    )
                )
                hists_dict[sample_name].update(
                    init_HH_histograms(
                        postfix=f"_truth_reco_central_{btag_count}btags_{btagger}_4_plus_truth_matched_jets_selection_v2"
                    )
                )
                for pairing in pairing_methods:
                    hists_dict[sample_name].update(
                        init_HH_histograms(
                            postfix=f"_truth_reco_central_{btag_count}btags_{btagger}_4_plus_truth_matched_jets_correct_{pairing}_selection"
                        )
                    )
                    hists_dict[sample_name].update(init_HH_histograms(postfix="_truth"))
                    hists_dict[sample_name].update(
                        init_HH_histograms(
                            hh_vars_binranges,
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}",
                        )
                    )
                    hists_dict[sample_name].update(
                        init_HH_histograms(
                            hh_vars_binranges,
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_correct_pairs",
                        )
                    )
                    hists_dict[sample_name].update(
                        init_HH_histograms(
                            hh_vars_binranges,
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_wrong_pairs",
                        )
                    )

                ###################################################
                ### Truth matched Pairing vs Reco HH histograms ###
                ###################################################
                for pairing in pairing_methods:
                    ### H1 and H2 histograms ########################
                    hists_dict[sample_name].update(
                        init_H_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}"
                        )
                    )
                    hists_dict[sample_name].update(
                        init_H_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_correct_pairs",
                        )
                    )
                    hists_dict[sample_name].update(
                        init_H_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_wrong_pairs",
                        )
                    )
                    hists_dict[sample_name].update(
                        init_H_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_lt_370_GeV"
                        )
                    )
                    hists_dict[sample_name].update(
                        init_H_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_geq_370_GeV"
                        )
                    )
                    hists_dict[sample_name].update(
                        init_H_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_lt_370_GeV"
                        )
                    )
                    hists_dict[sample_name].update(
                        init_H_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_geq_370_GeV"
                        )
                    )

                ####################################################
                ### Truth matched Pairing efficiency histograms ####
                ####################################################
                for pairing in pairing_methods:
                    hists_dict[sample_name].update(
                        init_HH_histograms(
                            hh_vars_binranges,
                            postfix=f"_reco_truth_matched_{btag_count}btags_{btagger}_{pairing}",
                        )
                    )
                    hists_dict[sample_name].update(
                        init_HH_histograms(
                            hh_vars_binranges,
                            postfix=f"_reco_truth_matched_{btag_count}btags_{btagger}_{pairing}_correct_pairs",
                        )
                    )
                ##################################################
                ### Mass plane for pairing methods histograms ####
                ##################################################
                for pairing in pairing_methods:
                    hists_dict[sample_name].update(
                        init_mH_2d_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}"
                        )
                    )
                    hists_dict[sample_name].update(
                        init_mH_2d_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_wrong_pairs"
                        )
                    )
                    hists_dict[sample_name].update(
                        init_mH_2d_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_lt_370_GeV"
                        )
                    )
                    hists_dict[sample_name].update(
                        init_mH_2d_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_geq_370_GeV"
                        )
                    )
                    for region in ["signal", "control"]:
                        hists_dict[sample_name].update(
                            init_mH_2d_histograms(
                                postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}"
                            )
                        )
                        hists_dict[sample_name].update(
                            init_mH_2d_histograms(
                                postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}_wrong_pairs"
                            )
                        )
                        hists_dict[sample_name].update(
                            init_mH_2d_histograms(
                                postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}_lt_370_GeV"
                            )
                        )
                        hists_dict[sample_name].update(
                            init_mH_2d_histograms(
                                postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}_geq_370_GeV"
                            )
                        )
                ##################################################
                ### HH discriminant histograms
                ##################################################
                for pairing in pairing_methods:
                    hists_dict[sample_name].update(
                        init_top_veto_discrim_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}"
                        )
                    )
                    hists_dict[sample_name].update(
                        init_HH_abs_deltaeta_discrim_histograms(
                            postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}"
                        )
                    )
                    hists_dict[sample_name].update(
                        init_HH_mass_discrim_histograms(
                            posfix=f"_reco_{btag_count}btags_{btagger}_{pairing}"
                        )
                    )
                    hists_dict[sample_name].update(
                        init_HH_mass_discrim_histograms(
                            posfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_correct_pairs"
                        )
                    )
                    hists_dict[sample_name].update(
                        init_HH_mass_discrim_histograms(
                            posfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_wrong_pairs"
                        )
                    )

                ############################################################
                ### Signal and Control region histograms after analysis cuts
                ############################################################
                for pairing in pairing_methods:
                    for region in ["signal", "control"]:
                        hists_dict[sample_name].update(
                            init_HH_histograms(
                                postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}",
                                binrange={
                                    "pt": [20_000, 500_000],
                                    "eta": [-5, 5],
                                    "phi": [-3, 3],
                                    "mass": [100_000, 1_100_000],
                                },
                                bins=21,
                            )
                        )
                        hists_dict[sample_name].update(
                            init_HH_histograms(
                                postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}_wrong_pairs",
                                binrange={
                                    "pt": [20_000, 500_000],
                                    "eta": [-5, 5],
                                    "phi": [-3, 3],
                                    "mass": [100_000, 1_100_000],
                                },
                                bins=21,
                            )
                        )
                        hists_dict[sample_name].update(
                            init_HH_histograms(
                                postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}_lt_370_GeV",
                                binrange={
                                    "pt": [20_000, 500_000],
                                    "eta": [-5, 5],
                                    "phi": [-3, 3],
                                    "mass": [100_000, 1_100_000],
                                },
                                bins=21,
                            )
                        )
                        hists_dict[sample_name].update(
                            init_HH_histograms(
                                postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}_geq_370_GeV",
                                binrange={
                                    "pt": [20_000, 500_000],
                                    "eta": [-5, 5],
                                    "phi": [-3, 3],
                                    "mass": [100_000, 1_100_000],
                                },
                                bins=21,
                            )
                        )

                ############################################################
                ### Signal region jet flavor composition histograms
                ############################################################
                for pairing in pairing_methods:
                    hists_dict[sample_name].update(
                        init_jet_flavor_composition_histograms(
                            postfix=f"_signal_{btag_count}btags_{btagger}_{pairing}"
                        )
                    )

    return hists_dict


def init_event_no_histograms(bins=101) -> list:
    """Initialize event number histograms"""

    hists = {"event_number": HistogramDynamic("event_number", bins=bins)}

    return hists


def init_event_weight_histograms(bins=101, postfix=None) -> list:
    """Initialize mc event weight histograms"""

    hists = {}
    for h_name in ["mc_event_weight", "total_event_weight"]:
        h_name += f"{postfix if postfix else ''}"
        hists[h_name] = HistogramDynamic(h_name, bins=bins, dtype=float)

    return hists


def init_jet_kin_histograms(
    binrange={
        "pt": [0, 1_300_000],
        "eta": [-5, 5],
        "phi": [-3, 3],
        "mass": [0, 20_000],
    },
    bins=101,
    postfix=None,
):
    """Initialize jet kinematics 1d histograms"""

    hists = {}
    for kin_var in kin_labels:
        h_name = f"jet_{kin_var}{postfix if postfix else ''}"
        hists[h_name] = Histogram(
            h_name,
            binrange[kin_var],
            bins,
        )

    return hists


def init_leading_jets_histograms(
    binrange={
        "pt": [0, 5_000_000],
        "eta": [-5, 5],
        "phi": [-3, 3],
        "mass": [0, 20_000],
    },
    bins=101,
    prefix="leading_jet",
    n_leading=4,
    postfix=None,
) -> list:
    """Initialize leading jets histograms"""

    hists = {}
    for i in range(1, n_leading + 1):
        for var in binrange:
            h_name = f"{prefix}_{i}_{var}{postfix if postfix else ''}"
            hists[h_name] = Histogram(
                h_name,
                binrange[var],
                bins,
            )

    return hists


def init_n_true_bjet_composition_histograms(
    prefix="events_geq_n_true_bjets", binrange=[2, 5], postfix=None
) -> list:
    """Initialize the number of true b-jets composition histograms"""

    hists = {}
    h_name = f"{prefix}{postfix if postfix is not None else ''}"
    hists[h_name] = Histogram(
        h_name,
        binrange=binrange,
        bins=len(range(binrange[0], binrange[1])) + 1,
    )

    return hists


def init_jet_flavor_composition_histograms(
    postfix=None,
) -> list:
    """Initialize the jet flavor and btagging 1d histograms"""

    hists = {}
    h_name = f"jet_flavor{postfix if postfix else ''}"
    binrange = [0, 6]
    hists[h_name] = Histogram(
        h_name,
        binrange=binrange,
        bins=len(range(binrange[0], binrange[1])) + 1,
    )

    for flav in ["u", "c", "b", "tau"]:
        h_name = f"bjet_discrim_{flav}{postfix if postfix else ''}"
        hists[h_name] = Histogram(
            h_name,
            binrange=[-2, 20],
            bins=51,
        )

    return hists


def init_H_histograms(
    binrange={
        "pt": [0, 800_000],
        "eta": [-5, 5],
        "phi": [-3, 3],
        "mass": [0, 400_000],
    },
    bins=101,
    postfix=None,
) -> list:
    """Initialize H1 and H2 kinematics 1d histograms"""

    hists = {}
    for h in [1, 2]:
        for kin_var in kin_labels:
            h_name = f"h{h}_{kin_var}{postfix if postfix else ''}"
            hists[h_name] = Histogram(
                h_name,
                binrange[kin_var],
                bins,
            )

    return hists


def init_HH_histograms(
    binrange={
        "pt": [0, 500_000],
        "eta": [-5, 5],
        "phi": [-3, 3],
        "mass": [0, 1_200_000],
    },
    bins=101,
    postfix=None,
) -> list:
    """Initialize HH kinematics 1d histograms"""

    hists = {}
    for var, var_range in binrange.items():
        h_name = f"hh_{var}{postfix if postfix else ''}"
        hists[h_name] = Histogram(
            h_name,
            var_range,
            bins,
        )

    return hists


def init_mH_2d_histograms(binrange=[0, 200_000], bins=51, postfix=None) -> list:
    """Initialize mH 2d histograms"""

    hists = {}
    h_name = f"mHH_plane{postfix if postfix else ''}"
    hists[h_name] = Histogram(
        h_name,
        binrange=binrange,
        bins=bins,
        dimensions=2,
    )

    return hists


def init_HH_abs_deltaeta_discrim_histograms(
    binrange=[0, 5], bins=51, postfix=None
) -> list:
    """Initialize hh absolute deltaEta 1d histograms"""

    hists = {}
    h_name = f"hh_abs_deltaeta_discrim{postfix if postfix else ''}"
    hists[h_name] = Histogram(
        h_name,
        binrange=binrange,
        bins=bins,
    )

    return hists


def init_HH_mass_discrim_histograms(binrange=[0, 20], bins=51, posfix=None) -> list:
    """Initialize hh mass discriminant 1d histograms"""

    hists = {}
    h_name = f"hh_mass_discrim{posfix if posfix else ''}"
    hists[h_name] = Histogram(
        h_name,
        binrange=binrange,
        bins=bins,
    )

    return hists


def init_top_veto_discrim_histograms(binrange=[0, 7.5], bins=51, postfix=None) -> list:
    """Initialize top veto 1d histograms"""

    hists = {}
    h_name = f"top_veto_discrim{postfix if postfix else ''}"
    hists[h_name] = Histogram(
        h_name,
        binrange=binrange,
        bins=bins,
    )

    return hists
