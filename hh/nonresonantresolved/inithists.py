from argparse import Namespace
from hh.nonresonantresolved.pairing import pairing_methods
from hh.shared.utils import (
    logger,
    format_btagger_model_name,
)
from hh.shared.labels import kin_labels
from hh.shared.hist import (
    Histogram,
    Histogram2d,
    HistogramDynamic,
    HistogramCategorical,
)


def init_hists(inputs: dict, selections: dict, args: Namespace) -> dict:
    """Initialize histograms for the different studies using the shape of inputs."""

    logger.info("Initializing hisotgrams")
    hists_dict = {}
    for sample in inputs:
        sample_name = sample["label"]
        hists_dict[sample_name] = []
        hh_vars_binranges = {
            "pt": [20_000, 1_000_000],
            "eta": [-5, 5],
            "phi": [-3, 3],
            "mass": [20_000, 1_000_000],
            "sum_jet_pt": [20_000, 1_000_000],
            "delta_eta": [-5, 5],
        }

        ##################################################
        ### Truth H(H) histograms ########################
        ##################################################
        hists_dict[sample_name] += init_H_histograms(postfix="_truth")
        hists_dict[sample_name] += init_H_histograms(postfix="_reco")
        hists_dict[sample_name] += init_H_histograms(
            postfix="_reco_H1_truth_mass_lt_350_GeV"
        )
        hists_dict[sample_name] += init_H_histograms(
            postfix="_reco_H1_truth_mass_geq_350_GeV"
        )
        hists_dict[sample_name] += init_HH_histograms(postfix="_truth")

        ##################################################
        ### Reco H(H) histograms ########################
        ##################################################
        hists_dict[sample_name] += init_HH_histograms(
            postfix="_truth_reco_central_jets_selection"
        )
        hists_dict[sample_name] += init_HH_histograms(
            postfix="_truth_reco_central_truth_matched_jets_selection"
        )

        ##################################################
        ### Reco truth-matched H(H) histograms ###########
        ##################################################
        hists_dict[sample_name] += init_HH_histograms(postfix="_reco_truth_matched")
        hists_dict[sample_name] += init_HH_histograms(postfix="_truth_reco_matched")
        hists_dict[sample_name] += init_HH_histograms(postfix="_reco_truth_matched_v2")
        hists_dict[sample_name] += init_HH_histograms(
            postfix="_reco_vs_truth_response",
            binrange={v: [-100, 100] for v in kin_labels},
        )

        # ######################################################
        # ### HH jet candidate selections cutflow histograms ###
        # ######################################################
        # hists_dict[sample_name] += init_leading_jets_histograms(
        #     prefix="hh_jet", postfix="_truth_matched"
        # )
        # hists_dict[sample_name] += init_leading_jets_histograms(
        #     prefix="hh_jet", postfix="_truth_matched_4btags"
        # )
        # hists_dict[sample_name] += init_leading_jets_histograms(
        #     prefix="hh_jet", postfix="_truth_matched_2b2j_asym"
        # )
        # hists_dict[sample_name] += init_leading_jets_histograms(
        #     prefix="hh_jet", postfix="_truth_matched_2b2j_asym_nbtags"
        # )
        # hists_dict[sample_name] += init_leading_jets_histograms(
        #     prefix="hh_jet", postfix="_truth_matched_2b2j_asym_4btags"
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
                hists_dict[sample_name] += init_HH_histograms(
                    postfix=f"_truth_reco_central_{btag_count}btags_{btagger}_jets_selection"
                )
                hists_dict[sample_name] += init_HH_histograms(
                    postfix=f"_truth_reco_central_{btag_count}btags_{btagger}_4_plus_truth_matched_jets_selection"
                )
                hists_dict[sample_name] += init_HH_histograms(
                    postfix=f"_truth_reco_central_{btag_count}btags_{btagger}_4_plus_truth_matched_jets_selection_v2"
                )
                for pairing in pairing_methods:
                    hists_dict[sample_name] += init_HH_histograms(
                        postfix=f"_truth_reco_central_{btag_count}btags_{btagger}_4_plus_truth_matched_jets_correct_{pairing}_selection"
                    )
                ###################################################
                ### Truth matched Pairing vs Reco HH histograms ###
                ###################################################
                for pairing in pairing_methods:
                    hists_dict[sample_name] += init_HH_histograms(
                        hh_vars_binranges,
                        postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}",
                    )
                    hists_dict[sample_name] += init_HH_histograms(
                        hh_vars_binranges,
                        postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_correct",
                    )

                ####################################################
                ### Truth matched Pairing efficiency histograms ####
                ####################################################
                for pairing in pairing_methods:
                    hists_dict[sample_name] += init_HH_histograms(
                        hh_vars_binranges,
                        postfix=f"_reco_truth_matched_{btag_count}btags_{btagger}_{pairing}",
                    )
                    hists_dict[sample_name] += init_HH_histograms(
                        hh_vars_binranges,
                        postfix=f"_reco_truth_matched_{btag_count}btags_{btagger}_{pairing}_correct",
                    )
                ##################################################
                ### Mass plane for pairing methods histograms ####
                ##################################################
                for pairing in pairing_methods:
                    hists_dict[sample_name] += init_mH_2d_histograms(
                        postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}"
                    )
                    hists_dict[sample_name] += init_mH_2d_histograms(
                        postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_wrong_pairs"
                    )
                    hists_dict[sample_name] += init_mH_2d_histograms(
                        postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_lt_350_GeV"
                    )
                    hists_dict[sample_name] += init_mH_2d_histograms(
                        postfix=f"_reco_{btag_count}btags_{btagger}_{pairing}_geq_350_GeV"
                    )
                    for region in ["signal", "control"]:
                        hists_dict[sample_name] += init_mH_2d_histograms(
                            postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}"
                        )
                        hists_dict[sample_name] += init_mH_2d_histograms(
                            postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}_wrong_pairs"
                        )
                        hists_dict[sample_name] += init_mH_2d_histograms(
                            postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}_lt_350_GeV"
                        )
                        hists_dict[sample_name] += init_mH_2d_histograms(
                            postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}_geq_350_GeV"
                        )
                ##################################################
                ### X_HH histograms
                ##################################################
                X_HH_regions = {"signal": [0, 10], "control": [0, 10]}
                for pairing in pairing_methods:
                    for region, binrange in X_HH_regions.items():
                        hists_dict[sample_name] += init_HH_mass_discrim_histograms(
                            posfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}",
                            binrange=binrange,
                            bins=21,
                        )
                ############################################################
                ### Signal and Control region histograms after analysis cuts
                ############################################################
                for pairing in pairing_methods:
                    for region in ["signal", "control"]:
                        hists_dict[sample_name] += init_HH_histograms(
                            postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}",
                            binrange={
                                "pt": [20_000, 500_000],
                                "eta": [-5, 5],
                                "phi": [-3, 3],
                                "mass": [100_000, 1_100_000],
                            },
                            bins=20,
                        )
                        hists_dict[sample_name] += init_HH_histograms(
                            postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}_wrong_pairs",
                            binrange={
                                "pt": [20_000, 500_000],
                                "eta": [-5, 5],
                                "phi": [-3, 3],
                                "mass": [100_000, 1_100_000],
                            },
                            bins=20,
                        )
                        hists_dict[sample_name] += init_HH_histograms(
                            postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}_lt_350_GeV",
                            binrange={
                                "pt": [20_000, 500_000],
                                "eta": [-5, 5],
                                "phi": [-3, 3],
                                "mass": [100_000, 1_100_000],
                            },
                            bins=20,
                        )
                        hists_dict[sample_name] += init_HH_histograms(
                            postfix=f"_reco_{region}_{btag_count}btags_{btagger}_{pairing}_geq_350_GeV",
                            binrange={
                                "pt": [20_000, 500_000],
                                "eta": [-5, 5],
                                "phi": [-3, 3],
                                "mass": [100_000, 1_100_000],
                            },
                            bins=20,
                        )

        ##################################################
        ### Background estimate histograms
        ##################################################
        for pairing in pairing_methods:
            hists_dict[sample_name] += init_HH_histograms(
                hh_vars_binranges,
                postfix=f"_reco_{pairing}_bkgest_before",
            )
            hists_dict[sample_name] += init_HH_histograms(
                hh_vars_binranges,
                postfix=f"_reco_{pairing}_bkgest_after",
            )

    return hists_dict


def init_event_no_histograms(bins=100) -> list:
    """Initialize event number histograms"""

    hists = [HistogramDynamic("event_number", bins=bins)]

    return hists


def init_event_weight_histograms(bins=100, postfix=None) -> list:
    """Initialize mc event weight histograms"""

    hists = []
    for h_name in ["mc_event_weight", "total_event_weight"]:
        h_name += f"{postfix if postfix else ''}"
        hists += [HistogramDynamic(h_name, bins=bins, dtype=float)]

    return hists


def init_jet_kin_histograms(
    binrange={
        "pt": [0, 1_300_000],
        "eta": [-5, 5],
        "phi": [-3, 3],
        "mass": [0, 20_000],
    },
    bins=100,
    postfix=None,
):
    hists = []
    for kin_var in kin_labels:
        hists += [
            Histogram(
                f"jet_{kin_var}{postfix if postfix else ''}", binrange[kin_var], bins
            )
        ]

    return hists


def init_leading_jets_histograms(
    binrange={
        "pt": [0, 500_000],
        "eta": [-5, 5],
        "phi": [-3, 3],
        "mass": [0, 20_000],
    },
    bins=100,
    prefix="leading_jet",
    postfix=None,
) -> list:
    """Initialize leading jets histograms"""

    hists = []
    for i in [1, 2, 3, 4]:
        for kin_var in kin_labels:
            hists += [
                Histogram(
                    f"{prefix}_{i}_{kin_var}{postfix if postfix else ''}",
                    binrange[kin_var],
                    bins,
                )
            ]

    return hists


def init_truth_matched_mjj_histograms(binrange=[0, 200_000], bins=100) -> list:
    """Initialize truth matched mjj 1d histograms"""

    hists = []
    for reco_h in [1, 2]:
        hists += [
            Histogram(
                f"mjj{reco_h}",
                binrange=binrange,
                bins=bins,
            )
        ]
        hists += [
            Histogram(
                f"mjj{reco_h}_pairingPassedTruth",
                binrange=binrange,
                bins=bins,
            )
        ]

    return hists


def init_H_histograms(
    binrange={
        "pt": [0, 800_000],
        "eta": [-5, 5],
        "phi": [-3, 3],
        "mass": [0, 400_000],
    },
    bins=100,
    postfix=None,
) -> list:
    """Initialize H1 and H2 kinematics 1d histograms"""

    hists = []
    for h in [1, 2]:
        for kin_var in kin_labels:
            hists += [
                Histogram(
                    f"h{h}_{kin_var}{postfix if postfix else ''}",
                    binrange[kin_var],
                    bins,
                )
            ]

    return hists


def init_reco_H_truth_jet_histograms(
    binrange=[0, 7],
    bins=8,
    postfix=None,
) -> list:
    """Initialize H1 and H2 matched to the truth jet ID 1d histograms"""

    hists = []
    for reco_h in [1, 2]:
        hists += [
            Histogram(
                f"h{reco_h}_truth_jet{postfix if postfix else ''}",
                binrange,
                bins,
            )
        ]

    return hists


def init_HH_histograms(
    binrange={
        "pt": [0, 500_000],
        "eta": [-5, 5],
        "phi": [-3, 3],
        "mass": [0, 1_200_000],
    },
    bins=100,
    bins_logscale=False,
    postfix=None,
) -> list:
    """Initialize HH kinematics 1d histograms"""

    hists = []
    for var, var_range in binrange.items():
        hists += [
            Histogram(
                f"hh_{var}{postfix if postfix else ''}",
                var_range,
                bins,
                bins_logscale=bins_logscale,
            )
        ]

    return hists


def init_mH_2d_histograms(binrange=[0, 200_000], bins=50, postfix=None) -> list:
    """Initialize mH 2d histograms"""

    hists = []
    hists += [
        Histogram2d(
            f"mHH_plane{postfix if postfix else ''}",
            binrange=binrange,
            bins=bins,
        )
    ]

    return hists


def init_HH_deltaeta_histograms(binrange=[0, 5], bins=51) -> list:
    """Initialize hh deltaEta 1d histograms"""

    hists = []
    hists += [
        Histogram(
            "hh_deltaeta_baseline",
            binrange=binrange,
            bins=bins,
        )
    ]

    return hists


def init_HH_mass_discrim_histograms(binrange=[0, 10], bins=51, posfix=None) -> list:
    """Initialize hh mass discriminant 1d histograms"""

    hists = []
    hists += [
        Histogram(
            f"hh_mass_discrim{posfix if posfix else ''}",
            binrange=binrange,
            bins=bins,
        )
    ]

    return hists


def init_top_veto_histograms(binrange=[0, 7], bins=51) -> list:
    """Initialize top veto 1d histograms"""

    hists = []
    hists += [
        Histogram(
            "top_veto_baseline",
            binrange=binrange,
            bins=bins,
        )
    ]
    hists += [Histogram("top_veto_n_btags", binrange=binrange, bins=bins)]

    return hists


def init_reco_mH_truth_pairing_histograms(binrange=[0, 200_000], bins=100) -> list:
    """Initialize H truth pairing histograms"""

    hists = []
    hists += [
        Histogram(
            f"mH{reco_h}_pairingPassedTruth_children_originate_from_same_truth_parent",
            binrange=binrange,
            bins=bins,
        )
        for reco_h in [1, 2]
    ]

    return hists
