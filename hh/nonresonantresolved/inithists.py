from argparse import Namespace
from hh.shared.utils import logger, kin_labels
from .hist import Histogram, Histogram2d, HistogramDynamic


def init_hists(inputs: dict, args: Namespace) -> dict:
    """Initialize histograms for the different studies using the shape of inputs."""

    logger.info("Initializing hisotgrams")
    hists_dict = {}
    for sample in inputs:
        sample_type = sample["label"]
        hists_dict[sample_type] = []
        hists_dict[sample_type] += init_event_no_histograms()
        hists_dict[sample_type] += init_jet_kin_histograms()
        hists_dict[sample_type] += init_leading_jets_histograms()
        hists_dict[sample_type] += init_reco_H_histograms()
        hists_dict[sample_type] += init_reco_H_histograms(postfix="_signal_region")
        hists_dict[sample_type] += init_reco_H_histograms(postfix="_control_region")
        hists_dict[sample_type] += init_reco_mH_2d_histograms()
        hists_dict[sample_type] += init_reco_mH_2d_histograms(postfix="_signal_region")
        hists_dict[sample_type] += init_reco_mH_2d_histograms(postfix="_control_region")
        hists_dict[sample_type] += init_reco_HH_histograms()
        hists_dict[sample_type] += init_reco_HH_histograms(postfix="_signal_region")
        hists_dict[sample_type] += init_reco_HH_histograms(postfix="_control_region")
        hists_dict[sample_type] += init_reco_HH_deltaeta_histograms()
        hists_dict[sample_type] += init_reco_top_veto_histograms()
        hists_dict[sample_type] += init_reco_HH_mass_discrim_histograms()
        # if args.signal:
        #     hists_dict[sample_type] += init_reco_mH_truth_pairing_histograms()
        #     hists_dict[sample_type] += init_truth_matched_mjj_histograms()

    return hists_dict


def init_event_no_histograms(bins=100) -> list:
    """Initialize event number histograms"""

    hists = []
    hists += [
        HistogramDynamic(
            "event_number",
            bins=bins,
        )
    ]

    return hists


def init_jet_kin_histograms(
    binrange={
        "pt": [0, 1_300_000],
        "eta": [-5, 5],
        "phi": [-3, 3],
        "mass": [0, 20_000],
    },
    bins=100,
):
    hists = []
    for jet_var in kin_labels.keys():
        hists += [Histogram(f"jet_{jet_var}", binrange[jet_var], bins)]

    return hists


def init_leading_jets_histograms(binrange=[0, 1_300_000], bins=100) -> list:
    """Initialize leading jets histograms"""

    hists = []
    for leading_jet in [1, 2, 3, 4]:
        hists += [
            Histogram(
                f"leading_jet_{leading_jet}_pt",
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
                    f"h{reco_h}_{kin_var}_baseline{postfix if postfix else ''}",
                    binrange[kin_var],
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
                f"hh_{kin_var}_baseline{postfix if postfix else ''}",
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
            f"mH_plane_baseline{postfix if postfix else ''}",
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