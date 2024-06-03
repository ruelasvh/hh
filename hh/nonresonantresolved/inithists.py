from argparse import Namespace
from hh.shared.utils import logger, kin_labels
from hh.shared.hist import Histogram, Histogram2d, HistogramDynamic


def init_hists(inputs: dict, args: Namespace) -> dict:
    """Initialize histograms for the different studies using the shape of inputs."""

    logger.info("Initializing hisotgrams")
    hists_dict = {}
    for sample in inputs:
        sample_name = sample["label"]
        hists_dict[sample_name] = []
        hists_dict[sample_name] += init_HH_histograms(postfix="_truth")
        hists_dict[sample_name] += init_HH_histograms(postfix="_reco_truth_matched")
        hists_dict[sample_name] += init_HH_histograms(postfix="_truth_reco_matched")
        hists_dict[sample_name] += init_HH_histograms(
            postfix="_reco_truth_matched_deltar_pairing"
        )
        hists_dict[sample_name] += init_HH_histograms(postfix="_reco_truth_matched_v2")
        hists_dict[sample_name] += init_HH_histograms(
            postfix="_reco_vs_truth_response",
            binrange={
                "pt": [-100, 100],
                "eta": [-100, 100],
                "phi": [-100, 100],
                "mass": [-100, 100],
            },
        )
        hists_dict[sample_name] += init_leading_jets_histograms(
            prefix="hh_jet", postfix="_truth_matched"
        )
        hists_dict[sample_name] += init_leading_jets_histograms(
            prefix="hh_jet", postfix="_truth_matched_4_btags"
        )
        hists_dict[sample_name] += init_leading_jets_histograms(
            prefix="hh_jet", postfix="_truth_matched_2b2j_asym"
        )
        hists_dict[sample_name] += init_leading_jets_histograms(
            prefix="hh_jet", postfix="_truth_matched_2b2j_asym_n_btags"
        )
        hists_dict[sample_name] += init_leading_jets_histograms(
            prefix="hh_jet", postfix="_truth_matched_2b2j_asym_4_btags"
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
    for reco_h in [1, 2]:
        for kin_var in kin_labels:
            hists += [
                Histogram(
                    f"h{reco_h}_{kin_var}{postfix if postfix else ''}",
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
    postfix=None,
) -> list:
    """Initialize HH kinematics 1d histograms"""

    hists = []
    for kin_var in kin_labels:
        hists += [
            Histogram(
                f"hh_{kin_var}{postfix if postfix else ''}",
                binrange[kin_var],
                bins,
            )
        ]

    return hists


def init_mH_2d_histograms(binrange=[0, 200_000], bins=50, postfix=None) -> list:
    """Initialize mH 2d histograms"""

    hists = []
    hists += [
        Histogram2d(
            f"mH_plane{postfix if postfix else ''}",
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


def init_HH_mass_discrim_histograms(binrange=[0, 10], bins=51) -> list:
    """Initialize hh mass discriminant 1d histograms"""

    hists = []
    hists += [
        Histogram(
            "hh_mass_discrim_baseline",
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
