#!/usr/bin/env python3

import uproot
import numpy as np
import awkward as ak
import vector as p4
from matplotlib.figure import Figure
import hep2plts as plts


X1000_S100_mc20a = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/samples/mc20_13TeV.hh4b_boosted_via_sh4b_boosted/grid/output/user.viruelas.HH4b-no-trig.2022_10_20.801595.Py8EG_A14NNPDF23LO_XHS_X1000_S100_4b.e8448_a899_r13167_p5057_TREE/user.viruelas.30932128._000001.output-hh4b.root"

X1000_S100_mc20d = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/samples/mc20_13TeV.hh4b_boosted_via_sh4b_boosted/grid/output/user.viruelas.HH4b-no-trig.2022_10_20.801595.Py8EG_A14NNPDF23LO_XHS_X1000_S100_4b.e8448_a899_r13144_p5057_TREE/user.viruelas.30932122._000001.output-hh4b.root"

X1000_S100_mc20e = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/samples/mc20_13TeV.hh4b_boosted_via_sh4b_boosted/grid/output/user.viruelas.HH4b-no-trig.2022_10_20.801595.Py8EG_A14NNPDF23LO_XHS_X1000_S100_4b.e8448_a899_r13145_p5057_TREE/user.viruelas.30932125._000001.output-hh4b.root"

X2000_S100_mc20a = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/samples/mc20_13TeV.hh4b_boosted_via_sh4b_boosted/grid/output/user.viruelas.HH4b-no-trig.2022_10_20.801614.Py8EG_A14NNPDF23LO_XHS_X2000_S100_4b.e8448_a899_r13167_p5057_TREE/user.viruelas.30932131._000001.output-hh4b.root"

X3000_S100_mc20a = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/samples/mc20_13TeV.hh4b_boosted_via_sh4b_boosted/grid/output/user.viruelas.HH4b-no-trig.2022_10_20.801637.Py8EG_A14NNPDF23LO_XHS_X3000_S100_4b.e8448_a899_r13167_p5057_TREE/user.viruelas.30932136._000001.output-hh4b.root"

X2000_S400_mc20e_TEST = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/run/analysis-variables.root"

invGeV = 1 / 1000

# pT_bins = np.linspace(200, 2000, 36)
# mass_bins = np.linspace(0, 1000, 51)


def draw_leadingVR_jets(data, ax, label=None, title=None, atlas_sec_tag=None):
    bins = np.arange(10)
    jet_counts, _ = np.histogram(data, bins)
    x_label = "Ghost Associated VR trk-jets"

    ax.hhist(
        jet_counts,
        bins,
        title,
        x_label=x_label,
        atlas_sec_tag=atlas_sec_tag,
    )


def draw_pT_hist(data, ax, label=None, truth=None, title=None, atlas_sec_tag=None):
    # bins = np.linspace(250, 3500, 50)
    bins = np.linspace(200, 2000, 36)
    jet_counts, _ = np.histogram(data * invGeV, bins)
    x_label = (
        "Leading large-R jet $p_{\mathrm{T}}$ [GeV]"
        + f" {f'matched to {truth}' if truth else ''}"
    )

    ax.hhist(
        # normalized
        # jet_counts / np.sum(jet_counts),
        jet_counts,
        bins,
        label,
        title,
        x_label=x_label,
        atlas_sec_tag=atlas_sec_tag,
    )


def draw_m_hist(data, ax, label=None, truth=None, title=None, atlas_sec_tag=None):
    bins = np.linspace(0, 1000, 51)
    jet_counts, _ = np.histogram(data * invGeV, bins)
    x_label = f"Leading large-R jet mass [GeV] {f'matched to {truth}' if truth else ''}"

    ax.hhist(
        # normalized
        # jet_counts / np.sum(jet_counts),
        jet_counts,
        bins,
        label,
        title,
        x_label=x_label,
        atlas_sec_tag=atlas_sec_tag,
    )


def draw_deltaR_hist(data, ax, label=None, truth=None, title=None, atlas_sec_tag=None):
    bins = np.linspace(0, 10, 101)
    jet_counts, _ = np.histogram(data, bins)

    x_label = f"$\Delta R$ between largeR jets and {truth}"

    ax.hhist(
        # normalized
        jet_counts / np.sum(jet_counts),
        bins,
        label,
        title,
        x_label=x_label,
        atlas_sec_tag=atlas_sec_tag,
    )


def match_jet_to_truth_particle(sample, cuts):
    """Match jet to truth S or H using min delatR"""

    jet_p4 = p4.zip(
        {
            "pt": sample["recojet_antikt10_NOSYS_pt"][cuts],
            "eta": sample["recojet_antikt10_NOSYS_eta"][cuts],
            "phi": sample["recojet_antikt10_NOSYS_phi"][cuts],
            "mass": sample["recojet_antikt10_NOSYS_m"][cuts],
        }
    )
    H_p4 = p4.zip(
        {
            "pt": sample["truth_H_pt"],
            "eta": sample["truth_H_eta"],
            "phi": sample["truth_H_phi"],
            "mass": sample["truth_H_m"],
        }
    )
    S_p4 = p4.zip(
        {
            "pt": sample["truth_S_pt"],
            "eta": sample["truth_S_eta"],
            "phi": sample["truth_S_phi"],
            "mass": sample["truth_S_m"],
        }
    )

    print(
        f"total number events from jets, H and S in sample (all should be equal): {len(jet_p4)}, {len(H_p4)}, {len(S_p4)}"
    )

    jet_deltaR_to_H = jet_p4.deltaR(H_p4)
    jet_deltaR_to_S = jet_p4.deltaR(S_p4)

    global_min_deltaR = 0.75

    good = jet_deltaR_to_H < global_min_deltaR
    jet_deltaR_to_H = ak.mask(jet_deltaR_to_H, good)
    which_H = ak.argmin(jet_deltaR_to_H, axis=1, keepdims=True)

    good = jet_deltaR_to_S < global_min_deltaR
    jet_deltaR_to_S = ak.mask(jet_deltaR_to_S, good)
    which_S = ak.argmin(jet_deltaR_to_S, axis=1, keepdims=True)

    print(
        f"Jets that overlap in min deltaR for S and H: {ak.sum(jet_deltaR_to_H[which_H] == jet_deltaR_to_S[which_S])}"
    )

    matched_to_H = ak.flatten(jet_deltaR_to_H[which_H] < jet_deltaR_to_S[which_S])
    matched_to_S = ~matched_to_H

    which_S = ak.mask(which_S, matched_to_S)
    which_H = ak.mask(which_H, matched_to_H)

    truth_matched_to_H_jet = jet_p4[which_H]
    truth_matched_to_S_jet = jet_p4[which_S]

    return {
        "jet_deltaR_to_H": jet_deltaR_to_H,
        "jet_deltaR_to_S": jet_deltaR_to_S,
        "truth_matched_to_H_jet": truth_matched_to_H_jet,
        "truth_matched_to_S_jet": truth_matched_to_S_jet,
        "which_S": which_S,
        "which_H": which_H,
    }


def btag_jets(truth_matched_jets, sample):
    vr_jets = sample[
        "recojet_antikt10_NOSYS_leadingVRTrackJetsBtag_DL1r_FixedCutBEff_77"
    ]
    matched_vr_jets_to_H = vr_jets[truth_matched_jets["which_H"]]
    matched_vr_jets_to_S = vr_jets[truth_matched_jets["which_S"]]
    which_largeR_jets_matched_H_btag = ak.firsts(
        ak.sum(matched_vr_jets_to_H, axis=2) >= 2
    )
    which_largeR_jets_matched_S_btag = (ak.count(matched_vr_jets_to_S, axis=2) >= 2) & (
        ak.sum(matched_vr_jets_to_S, axis=2) >= 2
    )
    # which_largeR_jets_matched_S_btag = ak.firsts(
    #     ak.sum(matched_vr_jets_to_S, axis=2) >= 2
    # )

    return {
        "truth_matched_to_H_jet_btag77": truth_matched_jets["truth_matched_to_H_jet"][
            which_largeR_jets_matched_H_btag
        ],
        "truth_matched_to_S_jet_btag77": truth_matched_jets["truth_matched_to_S_jet"][
            which_largeR_jets_matched_S_btag
        ],
    }


def draw_largeR_jets(largeR_dict):
    fig_jetPt, ax_jetPt = plts.subplots()
    fig_jetM, ax_jetM = plts.subplots()
    fig_deltaR_to_H, ax_deltaR_to_H = plts.subplots()
    fig_deltaR_to_S, ax_deltaR_to_S = plts.subplots()
    fig_truth_matched_to_H_jetM, ax_truth_matched_to_H_jetM = plts.subplots()
    fig_truth_matched_to_S_jetM, ax_truth_matched_to_S_jetM = plts.subplots()
    fig_truth_matched_to_H_jetPt, ax_truth_matched_to_H_jetPt = plts.subplots()
    fig_truth_matched_to_S_jetPt, ax_truth_matched_to_S_jetPt = plts.subplots()
    fig_VRtrack_jets, ax_VRtrack_jets = plts.subplots()

    pt_cut = 200e3
    eta_cut = 2
    m_cut = 50e3
    cuts_label = "$p_{\mathrm{T}}$ > 200 GeV, m > 50 GeV and $|\eta| < 2$"
    for sample_name, sample in largeR_dict.items():
        atlas_sec_tag = f"{sample_name}\n{cuts_label}"
        cuts = (
            (sample["recojet_antikt10_NOSYS_pt"] > pt_cut)
            & (sample["recojet_antikt10_NOSYS_m"] > m_cut)
            & (np.abs(sample["recojet_antikt10_NOSYS_eta"]) < eta_cut)
        )
        draw_pT_hist(
            np.concatenate(sample["recojet_antikt10_NOSYS_pt"][cuts]),
            ax_jetPt,
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_m_hist(
            np.concatenate(sample["recojet_antikt10_NOSYS_m"][cuts]),
            ax_jetM,
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_leadingVR_jets(
            np.concatenate(sample["recojet_antikt10_NOSYS_goodVRTrackJets"][cuts]),
            ax_VRtrack_jets,
            sample_name,
            atlas_sec_tag=atlas_sec_tag,
        )

        truth_matched_jets = match_jet_to_truth_particle(sample=sample, cuts=cuts)
        draw_deltaR_hist(
            ak.ravel(truth_matched_jets["jet_deltaR_to_H"]),
            ax_deltaR_to_H,
            truth="H",
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_deltaR_hist(
            ak.ravel(truth_matched_jets["jet_deltaR_to_S"]),
            ax_deltaR_to_S,
            truth="S",
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_m_hist(
            ak.ravel(truth_matched_jets["truth_matched_to_H_jet"].mass),
            ax_truth_matched_to_H_jetM,
            label="No btag",
            truth="H",
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_pT_hist(
            ak.ravel(truth_matched_jets["truth_matched_to_H_jet"].pt),
            ax_truth_matched_to_H_jetPt,
            label="No btag",
            truth="H",
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_m_hist(
            ak.ravel(truth_matched_jets["truth_matched_to_S_jet"].mass),
            ax_truth_matched_to_S_jetM,
            label="No btag",
            truth="S",
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_pT_hist(
            ak.ravel(truth_matched_jets["truth_matched_to_S_jet"].pt),
            ax_truth_matched_to_S_jetPt,
            label="No btag",
            truth="S",
            atlas_sec_tag=atlas_sec_tag,
        )

        truth_matched_jets_btag = btag_jets(truth_matched_jets, sample)
        draw_m_hist(
            ak.ravel(truth_matched_jets_btag["truth_matched_to_H_jet_btag77"].mass),
            ax_truth_matched_to_H_jetM,
            label="btag77",
            truth="H",
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_pT_hist(
            ak.ravel(truth_matched_jets_btag["truth_matched_to_H_jet_btag77"].pt),
            ax_truth_matched_to_H_jetPt,
            label="btag77",
            truth="H",
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_m_hist(
            ak.ravel(truth_matched_jets_btag["truth_matched_to_S_jet_btag77"].mass),
            ax_truth_matched_to_S_jetM,
            label="btag77",
            truth="S",
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_pT_hist(
            ak.ravel(truth_matched_jets_btag["truth_matched_to_S_jet_btag77"].pt),
            ax_truth_matched_to_S_jetPt,
            label="btag77",
            truth="S",
            atlas_sec_tag=atlas_sec_tag,
        )

    fig_jetPt.savefig("large-R-jets-pT.png", bbox_inches="tight")
    fig_jetM.savefig("large-R-jets-mass.png", bbox_inches="tight")
    fig_deltaR_to_H.savefig("large-R-jets-deltaR-to-H.png", bbox_inches="tight")
    fig_deltaR_to_S.savefig("large-R-jets-deltaR-to-S.png", bbox_inches="tight")
    fig_truth_matched_to_H_jetM.savefig(
        "large-R-jets-mass-truth-matched-to-H.png", bbox_inches="tight"
    )
    fig_truth_matched_to_S_jetM.savefig(
        "large-R-jets-mass-truth-matched-to-S.png", bbox_inches="tight"
    )
    fig_truth_matched_to_H_jetPt.savefig(
        "large-R-jets-pT-truth-matched-to-H.png", bbox_inches="tight"
    )
    fig_truth_matched_to_S_jetPt.savefig(
        "large-R-jets-pT-truth-matched-to-S.png", bbox_inches="tight"
    )
    fig_VRtrack_jets.savefig("good_VR_track_jets.png", bbox_inches="tight")


# def run():
#     for chunk in uproot.iterate(
#         [
#             f"{X2000_S100_mc20a}:AnalysisMiniTree",
#             f"{X1000_S100_mc20d}:AnalysisMiniTree",
#             f"{X1000_S100_mc20e}:AnalysisMiniTree",
#         ],
#         ["recojet_antikt10_NOSYS_pt"],
#     ):
#         print(chunk["recojet_antikt10_NOSYS_pt"])


def run():
    largeR_dict = {}
    samples = {
        "X2000_S400_mc20e_TEST": X2000_S400_mc20e_TEST,
        # "X1000_S100_mc20a": X1000_S100_mc20a,
        # "X2000_S100_mc20a": X2000_S100_mc20a,
        # "X3000_S100_mc20a": X3000_S100_mc20a,
    }
    for sname, fname in samples.items():
        with uproot.open(f"{fname}:AnalysisMiniTree") as f:
            largeR_dict[sname] = f.arrays(
                filter_name="/(recojet_antikt10_NOSYS|truth_[HS])_[pt|eta|phi|m|goodVRTrackJets|leadingVRTrackJetsBtag_DL1r_FixedCutBEff_77]/",
            )
    draw_largeR_jets(largeR_dict)


if __name__ == "__main__":
    run()
