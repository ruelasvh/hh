from argparse import Namespace
from hh.shared.utils import logger, kin_labels
from .hist import Histogram, Histogram2d, HistogramDynamic


def init_hists(inputs: dict, args: Namespace) -> dict:
    """Initialize histograms for the different studies using the shape of inputs."""

    logger.info("Initializing hisotgrams")
    hists_dict = {}
    binrange = {
        "pt": [0, 1_600_000],
        "eta": [-5, 5],
        "phi": [-3, 3],
        "mass": [0, 30_000],
    }
    for sample in inputs:
        sample_name = sample["label"]
        if any(k in sample_name for k in ["JZ", "dijets", "multijet"]):
            binrange = {**binrange, "pt": [0, 3_900_000]}
        # initialize histograms
        hists_dict[sample_name] = []
        # hists_dict[sample_name] += init_event_no_histograms()
        hists_dict[sample_name] += init_event_weight_histograms()
        # hists_dict[sample_name] += init_event_weight_histograms(
        #     postfix="_baseline_signal_region"
        # )
        # hists_dict[sample_name] += init_event_weight_histograms(
        #     postfix="_baseline_control_region"
        # )
        hists_dict[sample_name] += init_jet_kin_histograms()
        # hists_dict[sample_name] += init_jet_kin_histograms(
        #     postfix="_baseline_signal_region"
        # )
        # hists_dict[sample_name] += init_jet_kin_histograms(
        #     postfix="_baseline_control_region"
        # )
        hists_dict[sample_name] += init_leading_jets_histograms()
        # hists_dict[sample_name] += init_leading_jets_histograms(
        #     postfix="_baseline_signal_region"
        # )
        # hists_dict[sample_name] += init_leading_jets_histograms(
        #     postfix="_baseline_control_region"
        # )
        # hists_dict[sample_name] += init_reco_H_histograms(postfix="_baseline")
        # hists_dict[sample_name] += init_reco_H_histograms(
        #     postfix="_baseline_signal_region"
        # )
        # hists_dict[sample_name] += init_reco_H_histograms(
        #     postfix="_baseline_control_region"
        # )
        # hists_dict[sample_name] += init_reco_H_truth_jet_histograms(
        #     postfix="_baseline_signal_region"
        # )
        # hists_dict[sample_name] += init_reco_mH_2d_histograms(postfix="_baseline")
        # hists_dict[sample_name] += init_reco_mH_2d_histograms(
        #     postfix="_baseline_signal_region"
        # )
        # hists_dict[sample_name] += init_reco_mH_2d_histograms(
        #     postfix="_baseline_control_region"
        # )
        # hists_dict[sample_name] += init_reco_HH_histograms(postfix="_baseline")
        # hists_dict[sample_name] += init_reco_HH_histograms(
        #     postfix="_baseline_signal_region"
        # )
        # hists_dict[sample_name] += init_reco_HH_histograms(
        #     postfix="_baseline_control_region"
        # )
        # hists_dict[sample_name] += init_reco_HH_deltaeta_histograms()
        # hists_dict[sample_name] += init_reco_top_veto_histograms()
        # hists_dict[sample_name] += init_reco_HH_mass_discrim_histograms()
        # # if args.signal:
        # #     hists_dict[sample_name] += init_reco_mH_truth_pairing_histograms()
        # #     hists_dict[sample_name] += init_truth_matched_mjj_histograms()

    return hists_dict


def init_event_no_histograms(bins=100) -> list:
    """Initialize event number histograms"""

    hists = [HistogramDynamic("event_number", bins=bins)]

    return hists


def init_event_weight_histograms(bins=100, postfix=None) -> list:
    """Initialize mc event weight histograms"""

    hists = []
    for h_name in ["mc_event_weight", "pileup_weight", "total_event_weight"]:
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
    for jet_var in kin_labels.keys():
        hists += [
            Histogram(
                f"jet_{jet_var}{postfix if postfix else ''}", binrange[jet_var], bins
            )
        ]

    return hists


def init_leading_jets_histograms(
    binrange=[0, 1_300_000], bins=100, postfix=None
) -> list:
    """Initialize leading jets histograms"""

    hists = []
    for leading_jet in [1, 2, 3, 4]:
        hists += [
            Histogram(
                f"leading_jet_{leading_jet}_pt{postfix if postfix else ''}",
                binrange=binrange,
                bins=bins,
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


def init_reco_H_histograms(
    binrange={
        "pt": [0, 800_000],
        "eta": [-5, 5],
        "phi": [-3, 3],
        "mass": [0, 400_000],
    },
    bins=100,
    postfix=None,
) -> list:
    """Initialize reconstructed H1 and H2 kinematics 1d histograms"""

    hists = []
    for reco_h in [1, 2]:
        for kin_var in kin_labels.keys():
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
    """Initialize reconstructed H1 and H2 matched to the truth jet ID 1d histograms"""

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


def init_reco_HH_histograms(
    binrange={
        "pt": [0, 500_000],
        "eta": [-5, 5],
        "phi": [-3, 3],
        "mass": [0, 1_300_000],
    },
    bins=100,
    postfix=None,
) -> list:
    """Initialize reconstructed HH kinematics 1d histograms"""

    hists = []
    for kin_var in kin_labels.keys():
        hists += [
            Histogram(
                f"hh_{kin_var}{postfix if postfix else ''}",
                binrange[kin_var],
                bins,
            )
        ]

    return hists


def init_reco_mH_2d_histograms(binrange=[0, 200_000], bins=50, postfix=None) -> list:
    """Initialize reconstructed mH 2d histograms"""

    hists = []
    hists += [
        Histogram2d(
            f"mH_plane{postfix if postfix else ''}",
            binrange=binrange,
            bins=bins,
        )
    ]

    return hists


def init_reco_HH_deltaeta_histograms(binrange=[0, 5], bins=50) -> list:
    """Initialize reconstructed hh deltaEta 1d histograms"""

    hists = []
    hists += [
        Histogram(
            "hh_deltaeta_baseline",
            binrange=binrange,
            bins=bins,
        )
    ]

    return hists


def init_reco_HH_mass_discrim_histograms(binrange=[0, 10], bins=21) -> list:
    """Initialize reconstructed hh mass discriminant 1d histograms"""

    hists = []
    hists += [
        Histogram(
            "hh_mass_discrim_baseline",
            binrange=binrange,
            bins=bins,
        )
    ]

    return hists


def init_reco_top_veto_histograms(binrange=[0, 7], bins=8) -> list:
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
    """Initialize reconstructed H truth pairing histograms"""

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
