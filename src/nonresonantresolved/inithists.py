from collections import defaultdict
from .utils import kin_labels
from .hist import Histogram, Histogramddv2
from .triggers import run3_all as triggers_run3_all
from shared.utils import logger


def init_hists(inputs: dict, args: dict) -> dict:
    """Initialize histograms for the different studies using the shape of inputs."""

    logger.info("Initializing hisotgrams")
    hists_dict = defaultdict(lambda: defaultdict(int))
    for sample_type in inputs.keys():
        hists_dict[sample_type] = []
        hists_dict[sample_type] += init_jet_kin_histograms()
        hists_dict[sample_type] += init_leading_jets_histograms()
        hists_dict[sample_type] += init_reco_mH_histograms()
        hists_dict[sample_type] += init_reco_mH_2d_histograms()
        hists_dict[sample_type] += init_reco_hh_deltaeta_histograms()
        hists_dict[sample_type] += init_reco_top_veto_histograms()
        if args.signal:
            hists_dict[sample_type] += init_reco_mH_truth_pairing_histograms()
            hists_dict[sample_type] += init_truth_matched_mjj_histograms()

    return hists_dict


def init_jet_kin_histograms(
    binrange={"pt": [0, 1_300_000], "eta": [-5, 5], "phi": [-3, 3], "m": [0, 20_000]},
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
        hists += [
            Histogram(
                f"leading_jet_{leading_jet}_pt_trigPassed_{trigger}",
                binrange=binrange,
                bins=bins,
            )
            for trigger in triggers_run3_all
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
        hists += [
            Histogram(
                f"mjj{reco_h}_pairingPassedTruth_trigPassed_{trigger}",
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
                f"mH{reco_h}_baseline",
                binrange=binrange,
                bins=bins,
            )
        ]
        # hists += [
        #     Histogram(
        #         f"mH{reco_h}_trigPassed_{trigger}",
        #         binrange=binrange,
        #         bins=bins,
        #     )
        #     for trigger in triggers_run3_all
        # ]

    return hists


def init_reco_mH_2d_histograms(binrange=[0, 200_000], bins=50) -> list:
    """Initialize reconstructed mH 2d histograms"""

    hists = []
    hists += [
        Histogramddv2(
            "mH_plane_baseline",
            binrange=binrange,
            bins=bins,
        )
    ]
    # hists += [
    #     Histogramddv2(
    #         f"mH_plane_trigPassed_allTriggersOR",
    #         binrange=binrange,
    #         bins=bins,
    #     )
    # ]
    # hists += [
    #     Histogramddv2(
    #         f"mH_plane_trigPassed_{trigger}",
    #         binrange=binrange,
    #         bins=bins,
    #     )
    #     for trigger in triggers_run3_all
    # ]

    return hists


def init_reco_hh_deltaeta_histograms(binrange=[0, 5], bins=200) -> list:
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


def init_reco_top_veto_histograms(binrange=[0, 7.5], bins=200) -> list:
    """Initialize top veto 1d histograms"""

    hists = []
    hists += [
        Histogram(
            "top_veto_baseline",
            binrange=binrange,
            bins=bins,
        )
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
