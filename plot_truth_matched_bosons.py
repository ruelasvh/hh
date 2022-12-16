#!/usr/bin/env python3

import uproot
import numpy as np
import awkward as ak
import vector as p4
import hep2plts as plts

SAMPLES_PATH = "/lustre/fs22/group/atlas/ruelasv/samples/mc20_13TeV.gHH4b/output"

mG3000_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514741.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M3000.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352670._000001.output-hh4b.root"
mG2500_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514739.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M2500.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352658._000001.output-hh4b.root"
mG2000_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514737.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M2000.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352641._000001.output-hh4b.root"
mG1500_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514734.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M1500.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352621._000001.output-hh4b.root"
mG1000_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514729.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M1000.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352577._000001.output-hh4b.root"
mG900_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514728.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M900.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352567._000001.output-hh4b.root"
mG800_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514727.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M800.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352557._000001.output-hh4b.root"
mG700_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514726.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M700.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352546._000001.output-hh4b.root"
mG600_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514725.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M600.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352535._000001.output-hh4b.root"

mG300_mc20d = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/run/analysis-variables.root"

invGeV = 1 / 1000


def draw_jets_count(data, ax, label=None, title=None, atlas_sec_tag=None):
    bins = np.arange(10)
    jet_counts, _ = np.histogram(data, bins)
    x_label = "Number of jets/event"
    y_label = "Frequency"

    ax.hhist(
        jet_counts,
        bins,
        label,
        title,
        x_label=x_label,
        y_label=y_label,
        atlas_sec_tag=atlas_sec_tag,
    )


def draw_pT_hist(data, ax, label=None, truth=None, title=None, **kwargs):
    bins = np.linspace(100, 2000, 36)
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
        log=True,
        **kwargs,
    )


def draw_m_hist(
    data,
    ax,
    label=None,
    title=None,
    xlabel=None,
    bins=None,
    **kwargs,
):
    bins = np.linspace(0, 600, 51) if bins is None else bins
    jet_counts, _ = np.histogram(data * invGeV, bins)
    x_label = xlabel if xlabel else r"$m_j$ [GeV]"
    ax.hhist(
        # normalized
        # jet_counts / np.sum(jet_counts),
        jet_counts,
        bins,
        label,
        title,
        x_label=x_label,
        log=True,
        **kwargs,
    )


def draw_deltaR_hist(data, ax, label=None, truth=None, title=None, atlas_sec_tag=None):
    bins = np.linspace(0, 1, 41)
    jet_counts, _ = np.histogram(data, bins)

    x_label = f"$\Delta R (j,{truth})$"

    ax.hhist(
        # normalized
        # jet_counts / np.sum(jet_counts),
        jet_counts,
        bins,
        label,
        title,
        x_label=x_label,
        atlas_sec_tag=atlas_sec_tag + f"\n$\Delta R < 0.75$",
        log=True,
    )


def match_jet_to_truth_particle(sample, cuts):
    """Match jet to truth H1 or H2 using min delatR"""

    jet_pt = sample["recojet_antikt10_NOSYS_pt"][cuts]
    jet_eta = sample["recojet_antikt10_NOSYS_eta"][cuts]
    jet_phi = sample["recojet_antikt10_NOSYS_phi"][cuts]
    jet_m = sample["recojet_antikt10_NOSYS_m"][cuts]
    n_jets = ak.count(jet_pt, axis=-1)
    valid = n_jets > 1

    jet_p4 = p4.zip(
        {
            "pt": jet_pt[valid],
            "eta": jet_eta[valid],
            "phi": jet_phi[valid],
            "mass": jet_m[valid],
        }
    )
    H1_p4 = p4.zip(
        {
            "pt": sample["truth_H1_pt"][valid],
            "eta": sample["truth_H1_eta"][valid],
            "phi": sample["truth_H1_phi"][valid],
            "mass": sample["truth_H1_m"][valid],
        }
    )
    H2_p4 = p4.zip(
        {
            "pt": sample["truth_H2_pt"][valid],
            "eta": sample["truth_H2_eta"][valid],
            "phi": sample["truth_H2_phi"][valid],
            "mass": sample["truth_H2_m"][valid],
        }
    )

    print(
        f"total number events from jets, H1 and H2 in sample (all should be equal): {len(jet_p4)}, {len(H1_p4)}, {len(H2_p4)}"
    )

    jet_dR_to_H1 = jet_p4.deltaR(H1_p4)
    jet_dR_to_H2 = jet_p4.deltaR(H2_p4)

    global_min_dR = 0.75

    good = jet_dR_to_H1 < global_min_dR
    jet_dR_to_H1 = ak.mask(jet_dR_to_H1, good)
    which_H1 = ak.argmin(jet_dR_to_H1, axis=1, keepdims=True)

    good = jet_dR_to_H2 < global_min_dR
    jet_dR_to_H2 = ak.mask(jet_dR_to_H2, good)
    which_H2 = ak.argmin(jet_dR_to_H2, axis=1, keepdims=True)

    # print(which_H1)
    # print(which_H2)

    print(ak.sum(which_H1 == which_H2))
    # print(jet_dR_to_H1[ak.count(which_H1[which_H1 == which_H2], axis=-1) > 0])
    # print(jet_dR_to_H2[ak.count(which_H2[which_H1 == which_H2], axis=-1) > 0])

    truth_matched_to_H1_jet = jet_p4[which_H1]
    truth_matched_to_H2_jet = jet_p4[which_H2]

    return {
        "jet_dR_to_H1": jet_dR_to_H1,
        "jet_dR_to_H2": jet_dR_to_H2,
        "truth_matched_to_H1_jet": truth_matched_to_H1_jet,
        "truth_matched_to_H2_jet": truth_matched_to_H2_jet,
        "which_H1": which_H1,
        "which_H2": which_H2,
        "valid": valid,
    }


def btag_jets(truth_matched_jets, sample):
    vr_jets = sample[
        "recojet_antikt10_NOSYS_leadingVRTrackJetsBtag_DL1r_FixedCutBEff_77"
    ]
    # pick events that have truth matched large-R jets
    matched_vr_jets_to_H1 = vr_jets[truth_matched_jets["valid"]][
        truth_matched_jets["which_H1"]
    ]
    matched_vr_jets_to_H2 = vr_jets[truth_matched_jets["valid"]][
        truth_matched_jets["which_H2"]
    ]
    # pick events with at least 2 btagged vr jets
    # which_largeR_jets_matched_H1_btag = ak.firsts(
    #     ak.sum(matched_vr_jets_to_H1, axis=-1) >= 2
    # )
    # which_largeR_jets_matched_H2_btag = ak.firsts(
    #     ak.sum(matched_vr_jets_to_H2, axis=-1) >= 2
    # )
    # # which_largeR_jets_matched_H2_btag = (
    # #     ak.count(matched_vr_jets_to_H2, axis=2) >= 2
    # # ) & (ak.sum(matched_vr_jets_to_H2, axis=2) >= 2)
    vr_jet_btags_per_H1 = ak.sum(matched_vr_jets_to_H1, axis=-1)
    vr_jet_btags_per_H2 = ak.sum(matched_vr_jets_to_H2, axis=-1)
    which_largeR_jets_matched_H1_btag = vr_jet_btags_per_H1 >= 2
    which_largeR_jets_matched_H2_btag = vr_jet_btags_per_H2 >= 2

    return {
        "truth_matched_to_H1_jet_btag77": truth_matched_jets["truth_matched_to_H1_jet"][
            which_largeR_jets_matched_H1_btag
        ],
        "truth_matched_to_H2_jet_btag77": truth_matched_jets["truth_matched_to_H2_jet"][
            which_largeR_jets_matched_H2_btag
        ],
    }


def compute_mass(p4_1, p4_2):
    reconstructed_system = p4_1 + p4_2
    return reconstructed_system.mass


def draw_largeR_jets(largeR_dict):
    fig_jetPt, ax_jetPt = plts.subplots()
    fig_jetM, ax_jetM = plts.subplots()
    fig_jets_counts, ax_jets_counts = plts.subplots()
    fig_deltaR_to_H1, ax_deltaR_to_H1 = plts.subplots()
    fig_deltaR_to_H2, ax_deltaR_to_H2 = plts.subplots()
    fig_truth_matched_to_H1_jetM, ax_truth_matched_to_H1_jetM = plts.subplots()
    fig_truth_matched_to_H2_jetM, ax_truth_matched_to_H2_jetM = plts.subplots()
    fig_truth_matched_to_H1_jetPt, ax_truth_matched_to_H1_jetPt = plts.subplots()
    fig_truth_matched_to_H2_jetPt, ax_truth_matched_to_H2_jetPt = plts.subplots()
    fig_truth_matched_to_H1H2_jetM, ax_truth_matched_to_H1H2_jetM = plts.subplots()

    pt_cut = 250
    m_cut = 50
    eta_cut = 2
    cuts_label = f"$p_T$ > {pt_cut} GeV, m > {m_cut} GeV and $|\eta| < {eta_cut}$"
    pt_cut = pt_cut / invGeV
    m_cut = m_cut / invGeV
    for sample_name, sample in largeR_dict.items():
        # atlas_sec_tag = f"DSID 514722, mc20d\n{cuts_label}"
        atlas_sec_tag = f"{cuts_label}"
        njets = ak.count(sample["recojet_antikt10_NOSYS_pt"], axis=-1)
        cuts = (
            (sample["recojet_antikt10_NOSYS_pt"] > pt_cut)
            & (sample["recojet_antikt10_NOSYS_m"] > m_cut)
            & (np.abs(sample["recojet_antikt10_NOSYS_eta"]) < eta_cut)
        )
        draw_jets_count(
            ak.ravel(njets),
            ax_jets_counts,
            f"Large-R jets {sample_name}",
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_pT_hist(
            np.concatenate(sample["recojet_antikt10_NOSYS_pt"][cuts]),
            ax_jetPt,
            sample_name,
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_m_hist(
            np.concatenate(sample["recojet_antikt10_NOSYS_m"][cuts]),
            ax_jetM,
            sample_name,
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_jets_count(
            np.concatenate(sample["recojet_antikt10_NOSYS_goodVRTrackJets"][cuts]),
            ax_jets_counts,
            f"VR track jets {sample_name}",
            atlas_sec_tag=atlas_sec_tag,
        )

        color = next(ax_truth_matched_to_H1_jetM._get_lines.prop_cycler)["color"]
        truth_matched_jets = match_jet_to_truth_particle(sample=sample, cuts=cuts)
        draw_deltaR_hist(
            ak.ravel(truth_matched_jets["jet_dR_to_H1"]),
            ax_deltaR_to_H1,
            sample_name,
            truth="H1",
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_deltaR_hist(
            ak.ravel(truth_matched_jets["jet_dR_to_H2"]),
            ax_deltaR_to_H2,
            sample_name,
            truth="H2",
            atlas_sec_tag=atlas_sec_tag,
        )
        draw_m_hist(
            ak.ravel(truth_matched_jets["truth_matched_to_H1_jet"].mass),
            ax_truth_matched_to_H1_jetM,
            label=f"{sample_name} (-- btag77)",
            xlabel=r"$m_{j_1}$ [GeV]",
            atlas_sec_tag=atlas_sec_tag,
            color=color,
        )
        draw_pT_hist(
            ak.ravel(truth_matched_jets["truth_matched_to_H1_jet"].pt),
            ax_truth_matched_to_H1_jetPt,
            label=f"{sample_name} (-- btag77)",
            truth="H1",
            atlas_sec_tag=atlas_sec_tag,
            color=color,
        )
        draw_m_hist(
            ak.ravel(truth_matched_jets["truth_matched_to_H2_jet"].mass),
            ax_truth_matched_to_H2_jetM,
            label=f"{sample_name} (-- btag77)",
            xlabel=r"$m_{j_2}$ [GeV]",
            atlas_sec_tag=atlas_sec_tag,
            color=color,
        )
        draw_pT_hist(
            ak.ravel(truth_matched_jets["truth_matched_to_H2_jet"].pt),
            ax_truth_matched_to_H2_jetPt,
            label=f"{sample_name} (-- btag77)",
            truth="H2",
            atlas_sec_tag=atlas_sec_tag,
            color=color,
        )
        draw_m_hist(
            ak.ravel(
                compute_mass(
                    truth_matched_jets["truth_matched_to_H1_jet"],
                    truth_matched_jets["truth_matched_to_H2_jet"],
                )
            ),
            ax_truth_matched_to_H1H2_jetM,
            label=f"{sample_name}",
            xlabel=r"$m_{j_1j_2}$ [GeV]",
            atlas_sec_tag=atlas_sec_tag,
            bins=np.linspace(0, 3500, 51),
        )

        truth_matched_jets_btag = btag_jets(truth_matched_jets, sample)
        draw_m_hist(
            ak.ravel(truth_matched_jets_btag["truth_matched_to_H1_jet_btag77"].mass),
            ax_truth_matched_to_H1_jetM,
            xlabel=r"$m_{j_1}$ [GeV]",
            # label=f"{sample_name} btag77",
            atlas_sec_tag=atlas_sec_tag,
            color=color,
            linestyle="dashed",
        )
        draw_pT_hist(
            ak.ravel(truth_matched_jets_btag["truth_matched_to_H1_jet_btag77"].pt),
            ax_truth_matched_to_H1_jetPt,
            # label=f"{sample_name} btag77",
            truth="H1",
            atlas_sec_tag=atlas_sec_tag,
            color=color,
            linestyle="dashed",
        )
        draw_m_hist(
            ak.ravel(truth_matched_jets_btag["truth_matched_to_H2_jet_btag77"].mass),
            ax_truth_matched_to_H2_jetM,
            xlabel=r"$m_{j_2}$ [GeV]",
            # label=f"{sample_name} btag77",
            atlas_sec_tag=atlas_sec_tag,
            color=color,
            linestyle="dashed",
        )
        draw_pT_hist(
            ak.ravel(truth_matched_jets_btag["truth_matched_to_H2_jet_btag77"].pt),
            ax_truth_matched_to_H2_jetPt,
            # label=f"{sample_name} btag77",
            truth="H2",
            atlas_sec_tag=atlas_sec_tag,
            color=color,
            linestyle="dashed",
        )

    # ax_jetM.set_yscale("log")
    fig_jetPt.savefig("large-R-jets-pT.png", bbox_inches="tight")
    fig_jetM.savefig("large-R-jets-mass.png", bbox_inches="tight")
    fig_jets_counts.savefig("jets_counts.png", bbox_inches="tight")
    fig_deltaR_to_H1.savefig("large-R-jets-deltaR-to-H1.png", bbox_inches="tight")
    fig_deltaR_to_H2.savefig("large-R-jets-deltaR-to-H2.png", bbox_inches="tight")
    fig_truth_matched_to_H1_jetM.savefig(
        "large-R-jets-mass-truth-matched-to-H1.png", bbox_inches="tight"
    )
    fig_truth_matched_to_H2_jetM.savefig(
        "large-R-jets-mass-truth-matched-to-H2.png", bbox_inches="tight"
    )
    fig_truth_matched_to_H1_jetPt.savefig(
        "large-R-jets-pT-truth-matched-to-H1.png", bbox_inches="tight"
    )
    fig_truth_matched_to_H2_jetPt.savefig(
        "large-R-jets-pT-truth-matched-to-H2.png", bbox_inches="tight"
    )
    fig_truth_matched_to_H1H2_jetM.savefig("H1H2-mass.png", bbox_inches="tight")


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
        # "mG300_mc20d": mG300_mc20d,
        # "mG600_mc20e": mG600_mc20e,
        "mG700_mc20e": mG700_mc20e,
        # "mG800_mc20e": mG800_mc20e,
        # "mG900_mc20e": mG900_mc20e,
        "mG1000_mc20e": mG1000_mc20e,
        "mG1500_mc20e": mG1500_mc20e,
        "mG2000_mc20e": mG2000_mc20e,
        "mG2500_mc20e": mG2500_mc20e,
        "mG3000_mc20e": mG3000_mc20e,
    }
    for sname, fname in samples.items():
        with uproot.open(f"{fname}:AnalysisMiniTree") as f:
            largeR_dict[sname] = f.arrays(
                filter_name="/(recojet_antikt10_NOSYS|truth_H[12])_[pt|eta|phi|m|goodVRTrackJets|leadingVRTrackJetsBtag_DL1r_FixedCutBEff_77]/",
            )
    draw_largeR_jets(largeR_dict)


if __name__ == "__main__":
    run()
