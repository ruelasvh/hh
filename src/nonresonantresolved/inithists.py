import logging
from collections import defaultdict
from .hist import Histogram, Histogramddv2
from .triggers import run3_all as triggers_run3_all


def init_hists(inputs: dict, args: dict) -> dict:
    """Initialize histograms for the different studies using the shape of inputs."""

    logging.info(f"Initializing hisotgrams")

    hists_dict = defaultdict(lambda: defaultdict(int))
    for sample_type in inputs.keys():
        hists_dict[sample_type] = []
        hists_dict[sample_type] += init_leading_jets_histograms()
        hists_dict[sample_type] += init_reco_mH_histograms()
        hists_dict[sample_type] += init_reco_mH_2d_histograms()
        if args.signal:
            hists_dict[sample_type] += init_reco_mH_truth_pairing_histograms()

    return hists_dict


def init_leading_jets_histograms(binrange=[0, 1_300_00], bins=100) -> list:
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
        hists += [
            Histogram(
                f"leading_jet_{leading_jet}_pt_trigPassed_{trigger}",
                binrange=binrange,
                bins=bins,
            )
            for trigger in triggers_run3_all
        ]

    return hists


def init_reco_mH_histograms(binrange=[0, 200_000], bins=100) -> list:
    """Initialize reconstructed mH 1d histograms"""

    hists = []
    for reco_h in [1, 2]:
        hists += [
            Histogram(
                f"mH{reco_h}",
                binrange=binrange,
                bins=bins,
            )
        ]
        hists += [
            Histogram(
                f"mH{reco_h}_trigPassed_{trigger}",
                binrange=binrange,
                bins=bins,
            )
            for trigger in triggers_run3_all
        ]

    return hists


def init_reco_mH_2d_histograms(binrange=[0, 200_000], bins=50) -> list:
    """Initialize reconstructed mH 2d histograms"""

    hists = []
    hists += [
        Histogramddv2(
            "mH_plane",
            binrange=binrange,
            bins=bins,
        )
    ]
    hists += [
        Histogramddv2(
            f"mH_plane_trigPassed_allTriggersOR",
            binrange=binrange,
            bins=bins,
        )
    ]
    hists += [
        Histogramddv2(
            f"mH_plane_trigPassed_{trigger}",
            binrange=binrange,
            bins=bins,
        )
        for trigger in triggers_run3_all
    ]

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
