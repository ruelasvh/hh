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


def draw_leadingVR_jets(leadingVRJets_dict):
    fig = Figure((4, 3))
    ax = fig.add_subplot()

    nJets = np.hstack(leadingVRJets_dict["recojet_antikt10_NOSYS_goodVRTrackJets"])

    bins = np.arange(8)
    frq, edges = np.histogram(nJets, bins)

    ax.bar(
        edges[:-1],
        frq,
        width=np.diff(edges),
        align="edge",
        label="VR trk-jets",
    )

    ax.legend(frameon=False)
    ax.set_xlabel("Ghost Associated VR trk-jets")
    ax.set_xticks(np.arange(len(bins)))
    ax.set_ylabel("Events")

    fig.canvas.print_figure("good_VR_track_jets.png", bbox_inches="tight")


# select_good_largeRjets():
#     get_2btags

# plot_leading_largeR_jet_masses():

# return leading_largeR_mass


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
        "jet_deltaR_to_H": ak.ravel(jet_deltaR_to_H),
        "jet_deltaR_to_S": ak.ravel(jet_deltaR_to_S),
        "truth_matched_to_H_jetM": ak.ravel(truth_matched_to_H_jet.mass),
        "truth_matched_to_S_jetM": ak.ravel(truth_matched_to_S_jet.mass),
    }


def draw_pT_hist(data, ax, label=None, title=None):
    # //normalise
    # if (normalise && h->Integral() > 0)
    #     h->Scale(1.0 /h->Integral(1, h->GetNbinsX()));
    # bins = np.linspace(250, 3500, 50)
    bins = np.linspace(200, 2000, 36)
    jet_counts, _ = np.histogram(data * invGeV, bins)
    x_label = "Leading large-R jet $p_{\mathrm{T}}$ [GeV]"

    ax.hhist(
        # normalized
        jet_counts / np.sum(jet_counts),
        bins,
        label,
        title,
        x_label=x_label,
    )


def draw_m_hist(data, ax, label=None, truth=None, title=None):
    bins = np.linspace(0, 1000, 51)
    jet_counts, _ = np.histogram(data * invGeV, bins)
    x_label = f"leading large-R jet mass [GeV] {f'matched to {truth}' if truth else ''}"

    ax.hhist(
        # normalized
        jet_counts / np.sum(jet_counts),
        bins,
        label,
        title,
        x_label=x_label,
    )

    ax.axvline(x=400, color="r")
    ax.axvline(x=125, color="g")


def draw_deltaR_hist(data, ax, label=None, truth=None, title=None):
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
    )


def draw_largeR_jets(largeR_dict):
    fig_jetPt, ax_jetPt = plts.subplots()
    fig_jetM, ax_jetM = plts.subplots()
    fig_deltaR_to_H, ax_deltaR_to_H = plts.subplots()
    fig_deltaR_to_S, ax_deltaR_to_S = plts.subplots()
    fig_truth_matched_to_H_jetM, ax_truth_matched_to_H_jetM = plts.subplots()
    fig_truth_matched_to_S_jetM, ax_truth_matched_to_S_jetM = plts.subplots()

    pt_cut = 200e3
    eta_cut = 2
    m_cut = 50e3
    title = "With $p_{\mathrm{T}}$ > 200 GeV, m > 50 GeV and $|\eta| < 2$ selection"
    for sample_name, sample in largeR_dict.items():
        cuts = (
            (sample["recojet_antikt10_NOSYS_pt"] > pt_cut)
            & (sample["recojet_antikt10_NOSYS_m"] > m_cut)
            & (np.abs(sample["recojet_antikt10_NOSYS_eta"]) < eta_cut)
        )
        draw_pT_hist(
            np.concatenate(sample["recojet_antikt10_NOSYS_pt"][cuts]),
            ax_jetPt,
            sample_name,
            title=title,
        )
        draw_m_hist(
            np.concatenate(sample["recojet_antikt10_NOSYS_m"][cuts]),
            ax_jetM,
            sample_name,
            title=title,
        )
        truth_matched_jets = match_jet_to_truth_particle(sample=sample, cuts=cuts)
        draw_deltaR_hist(
            truth_matched_jets["jet_deltaR_to_H"],
            ax_deltaR_to_H,
            sample_name,
            "H",
            title=title,
        )
        draw_deltaR_hist(
            truth_matched_jets["jet_deltaR_to_S"],
            ax_deltaR_to_S,
            sample_name,
            "S",
            title=title,
        )
        draw_m_hist(
            (truth_matched_jets["truth_matched_to_H_jetM"]),
            ax_truth_matched_to_H_jetM,
            sample_name,
            "H",
            title=title,
        )
        draw_m_hist(
            truth_matched_jets["truth_matched_to_S_jetM"],
            ax_truth_matched_to_S_jetM,
            sample_name,
            "S",
            title=title,
        )

    # fig_jetPt.savefig("large-R-jets-pT.png", bbox_inches="tight")
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
                filter_name="/(recojet_antikt10_NOSYS|truth_[HS])_[pt|eta|phi|m]/"
            )
    draw_largeR_jets(largeR_dict)


if __name__ == "__main__":
    run()
