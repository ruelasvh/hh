import concurrent.futures
import uproot
import numpy as np
import awkward as ak
import vector as p4
import matplotlib.pyplot as plt
import mplhep as hep
from histograms import IntHistogram, IntHistogramdd, EffHistogram

plt.style.use(hep.style.ATLAS)

np.seterr(divide="ignore", invalid="ignore")

executor = concurrent.futures.ThreadPoolExecutor(8 * 32)  # 32 threads

invGeV = 1 / 1_000


# Define samples
mc21_ggF_k01_small = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/run/analysis-variables-run3-k01"
mc21_ggF_k10_small = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/run/analysis-variables-run3-k10"
mc21_ggF_k10 = "/lustre/fs22/group/atlas/ruelasv/samples/mc21_13p6TeV.hh4b.ggF/output/user.viruelas.HH4b.ggF.2022_12_15.601480.PhPy8EG_HH4b_cHHH10d0.e8472_s3873_r13829_p5440_TREE/"
mc21_ggF_k01 = "/lustre/fs22/group/atlas/ruelasv/samples/mc21_13p6TeV.hh4b.ggF/output/user.viruelas.HH4b.ggF.2022_12_15.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5440_TREE/"

# Define triggers and trigger sets
run3_all_short = [
    "Asymm 2b2j DL1d@77%",  # run3
    "Asymm 3b1j DL1d@82%",  # run3
    "2b1j DL1d@70%",  # run2 reoptimized
    "Symm 2b2j DL1d@60%",  # run2 reoptimized
    "Asymm 2b2j+L1mu DL1d@77%",  # run3
]
run3_all = [
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bdl1d77_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25",  # 1 asymm
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_3j20c_020jvt_bdl1d82_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25",  # 2 asymm
    "HLT_j150_2j55_0eta290_020jvt_bdl1d70_pf_ftf_preselj80XX2j45b90_L1J85_3J30",  # 3 symm
    "HLT_2j35c_020jvt_bdl1d60_2j35c_020jvt_pf_ftf_presel2j25XX2j25b85_L14J15p0ETA25",  # 4 symm,
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bdl1d77_pf_ftf_presel2c20XX2c20b85_L1MU8F_2J15_J20",  # 5 asymm
]
run3_main_stream = run3_all[1:4]
run3_delayed_stream = run3_all[0:4]
run3_asymm_L1_jet = run3_all[0:2]
run3_asymm_L1_all = run3_all[0:2] + run3_all[4:5]
run2 = run3_all[2:4]

trig_sets = {
    "Run 2": run2,
    "Main physics stream": run3_main_stream,
    "Main + delayed streams": run3_delayed_stream,
    "Asymm L1 jet": run3_asymm_L1_jet,
    "Asymm L1 all": run3_asymm_L1_all,
    "OR of all": run3_all,
}

# Define histograms
truth_diHiggs_mass_hist = IntHistogram(
    "truth_diHiggs_mass", [200_000, 1_300_000], bins=100
)
leading_jets_pt_hist = IntHistogramdd("leading_jets_pt", [20_000, 2_000_000], bins=100)
trig_sets_hists = {}
for trig_set_key in trig_sets.keys():
    trig_sets_hists[trig_set_key] = EffHistogram(
        f"truth_diHiggs_passed_trig_{trig_set_key}", [200_000, 1_300_000], bins=40
    )
leading_jets_passed_trig_hists = {}
for ith_leading_jet in np.arange(0, 4):
    leading_jets_passed_trig_hists[ith_leading_jet] = {}
    bin_range = [20_000, 1_300_000]
    if ith_leading_jet == 2:
        bin_range = [20_000, 900_000]
    if ith_leading_jet == 3:
        bin_range = [20_000, 700_000]
    for trig_short in run3_all_short:
        leading_jets_passed_trig_hists[ith_leading_jet][trig_short] = EffHistogram(
            f"jet{ith_leading_jet}_pt_passed_{trig_short}",
            bin_range,
            bins=30,
        )

btaggers = ["DL1dv01_77", "GN120220509_77"]
leading_b_jets_pt_hists = {}
for btagger in btaggers:
    leading_b_jets_pt_hists[btagger] = IntHistogramdd(
        f"leading_b_jets_pt_{btagger}", [20_000, 2_000_000], bins=100
    )
trig_sets_btag_hists = {}
for btagger in btaggers:
    trig_sets_btag_hists[btagger] = {}
    for trig_set in trig_sets.keys():
        trig_sets_btag_hists[btagger][trig_set] = EffHistogram(
            f"truth_diHiggs_passed_trig_{trig_set}_btag_{btagger}",
            [200_000, 1_300_000],
            bins=40,
        )
leading_jets_passed_trig_btag_hists = {}
for btagger in btaggers:
    leading_jets_passed_trig_btag_hists[btagger] = {}
    for ith_leading_jet in np.arange(0, 4):
        leading_jets_passed_trig_btag_hists[btagger][ith_leading_jet] = {}
        bin_range = [20_000, 1_300_000]
        if ith_leading_jet == 2:
            bin_range = [20_000, 900_000]
        if ith_leading_jet == 3:
            bin_range = [20_000, 700_000]
        for trig_short in run3_all_short:
            leading_jets_passed_trig_btag_hists[btagger][ith_leading_jet][
                trig_short
            ] = EffHistogram(
                f"b_jet{ith_leading_jet}_pt_passed_{trig_short}",
                bin_range,
                bins=30,
            )


def get_truth_diHiggs(data, cuts=None):
    H1_p4 = p4.zip(
        {
            "pt": data["truth_H1_pt"][cuts],
            "eta": data["truth_H1_eta"][cuts],
            "phi": data["truth_H1_phi"][cuts],
            "mass": data["truth_H1_m"][cuts],
        }
    )
    H2_p4 = p4.zip(
        {
            "pt": data["truth_H2_pt"][cuts],
            "eta": data["truth_H2_eta"][cuts],
            "phi": data["truth_H2_phi"][cuts],
            "mass": data["truth_H2_m"][cuts],
        }
    )
    return H1_p4 + H2_p4


def get_valid_events_mask(events, pt_cut=20_000, eta_cut=2.5):
    jets_pt = events["recojet_antikt4_NOSYS_pt"]
    jets_eta = events["recojet_antikt4_NOSYS_eta"]
    passed_kin_cut = (jets_pt > pt_cut) & (np.abs(jets_eta) < eta_cut)
    passed_4_jets = ak.num(jets_pt[passed_kin_cut]) > 3
    valid_events = passed_4_jets
    return valid_events


def compute_eff(passed, total, percentage=True):
    eff = np.empty(total.shape)
    eff[:] = np.nan
    np.divide(passed, total, out=eff, where=total != 0)
    return eff * 100 if percentage else eff


# def draw_jeti_pt_vs_trig_eff_btag(data, tagger="DL1dv01", tagger_eff="77", cuts=None):
#     jets_pt_pre_trig = data["recojet_antikt4_NOSYS_pt"][cuts]
#     mass_res = 10
#     mass_bins = np.arange(20, 250, mass_res)
#     for i in np.arange(0, 4):
#         fig, (ax_top, ax_bottom) = plt.subplots(
#             2, height_ratios=(20, 10), sharex=True, constrained_layout=True
#         )
#         for trig, trig_short in zip(run3_all, run3_all_short):
#             trig_selection = data[f"trigPassed_{trig}"]
#             jets_pt_post_trig = data["recojet_antikt4_NOSYS_pt"][trig_selection]
#             jets_pt_post_trig = jets_pt_post_trig[cuts[trig_selection]]
#             btags = data[f"recojet_antikt4_NOSYS_{tagger}_FixedCutBEff_{tagger_eff}"][
#                 trig_selection
#             ]
#             btags = btags[cuts[trig_selection]]
#             btags_3j = ak.sum(btags, axis=-1) > 2
#             jets_pt_post_trig_btagged = jets_pt_post_trig[btags_3j]
#             h_tot, _ = np.histogram(jets_pt_pre_trig[:, i] * invGeV, mass_bins)
#             h_pass, _ = np.histogram(jets_pt_post_trig[:, i] * invGeV, mass_bins)
#             h_pass_btag, _ = np.histogram(
#                 jets_pt_post_trig_btagged[:, i] * invGeV, mass_bins
#             )
#             eff = (h_pass_btag / h_tot) * 100
#             hep.histplot(
#                 eff,
#                 mass_bins,
#                 ax=ax_top,
#                 histtype="errorbar",
#                 xerr=mass_res / 2,
#                 yerr=False,
#                 label=trig_short,
#             )
#             # ratio = h_pass_btag / h_pass
#             ratio = h_pass / h_pass_btag
#             hep.histplot(
#                 ratio,
#                 mass_bins,
#                 ax=ax_bottom,
#                 histtype="errorbar",
#                 xerr=False,
#                 yerr=False,
#                 label=trig_short,
#             )

#         ax_top.set_ylabel("Efficiency [%]")
#         ax_bottom.set_ylim(0.5, 2.1)
#         ax_bottom.axhline(y=1.0, color="black")
#         ax_bottom.set_xlabel(f"jet_{i+1} " + r"$p_{T}$ [GeV]")
#         ax_bottom.set_ylabel(f"Ratio to b-tagged")
#         fig.savefig(
#             f"trig_eff_vs_jet{i+1}_pt_{tagger}_{tagger_eff}.png", bbox_inches="tight"
#         )


def fill_jet_pt_hists(events, cuts):
    jets_pt = events["recojet_antikt4_NOSYS_pt"][cuts]
    leading_jets_pt_hist.fill(*[jets_pt[:, i] for i in np.arange(0, 4)])


def fill_b_jet_pt_hists(events, cuts=None, tagger="DL1dv01", tagger_eff="77"):
    jets_pt = events["recojet_antikt4_NOSYS_pt"]
    btags = events[f"recojet_antikt4_NOSYS_{tagger}_FixedCutBEff_{tagger_eff}"]
    btags_4 = ak.sum(btags, axis=-1) > 3
    valid_post_btags_4 = cuts & btags_4
    leading_b_jets_pt_hists[f"{tagger}_{tagger_eff}"].fill(
        *[jets_pt[valid_post_btags_4][:, i] for i in np.arange(0, 4)]
    )


def fill_truth_diHiggs_hists(events, cuts):
    diHiggs = get_truth_diHiggs(events)
    diHiggs_mass = ak.ravel(diHiggs.mass)
    truth_diHiggs_mass_hist.fill(diHiggs_mass)
    for trig_set_key, trig_set_values in trig_sets.items():
        trig_set_selections = []
        for i_trig, trig in enumerate(trig_set_values):
            trig_selection = events[f"trigPassed_{trig}"]
            if i_trig == 0:
                trig_set_selections = trig_selection
            else:
                trig_set_selections = trig_set_selections | trig_selection
        valid_pre_trig = cuts
        valid_post_trig = cuts & trig_set_selections
        trig_sets_hists[trig_set_key].fill(
            diHiggs_mass[valid_post_trig], diHiggs_mass[valid_pre_trig]
        )


def fill_truth_diHiggs_btags_hists(
    events, cuts=None, tagger="DL1dv01", tagger_eff="77"
):
    diHiggs = get_truth_diHiggs(events)
    diHiggs_mass = ak.ravel(diHiggs.mass)
    truth_diHiggs_mass_hist.fill(diHiggs_mass)
    for trig_set_key, trig_set_values in trig_sets.items():
        trig_selections = []
        for i_trig, trig in enumerate(trig_set_values):
            trig_selection = events[f"trigPassed_{trig}"]
            if i_trig == 0:
                trig_selections = trig_selection
            else:
                trig_selections = trig_selections | trig_selection
        btags = events[f"recojet_antikt4_NOSYS_{tagger}_FixedCutBEff_{tagger_eff}"]
        btags_4 = ak.sum(btags, axis=-1) > 3
        valid_pre_trig = cuts & btags_4
        valid_post_trig = valid_pre_trig & trig_selections
        trig_sets_btag_hists[f"{tagger}_{tagger_eff}"][trig_set_key].fill(
            diHiggs_mass[valid_post_trig], diHiggs_mass[valid_pre_trig]
        )


def fill_leading_jet_pt_passed_trig_hists(data, cuts):
    jets_pt = data["recojet_antikt4_NOSYS_pt"]
    for ith_leading_jet in np.arange(0, 4):
        for trig, trig_short in zip(run3_all, run3_all_short):
            trig_selection = data[f"trigPassed_{trig}"]
            valid_pre_trig = cuts
            valid_post_trig = cuts & trig_selection
            leading_jets_passed_trig_hists[ith_leading_jet][trig_short].fill(
                jets_pt[valid_post_trig][:, ith_leading_jet],
                jets_pt[valid_pre_trig][:, ith_leading_jet],
            )


def fill_leading_b_jet_pt_vs_trig_hists(
    data, cuts=None, tagger="DL1dv01", tagger_eff="77"
):
    jets_pt = data["recojet_antikt4_NOSYS_pt"]
    for ith_leading_jet in np.arange(0, 4):
        for trig, trig_short in zip(run3_all, run3_all_short):
            trig_selection = data[f"trigPassed_{trig}"]
            btags = data[f"recojet_antikt4_NOSYS_{tagger}_FixedCutBEff_{tagger_eff}"]
            btags_4 = ak.sum(btags, axis=-1) > 3
            valid_no_trig = cuts & btags_4
            valid_trig = trig_selection & cuts & btags_4
            leading_jets_passed_trig_btag_hists[f"{tagger}_{tagger_eff}"][
                ith_leading_jet
            ][trig_short].fill(
                jets_pt[valid_trig][:, ith_leading_jet],
                jets_pt[valid_no_trig][:, ith_leading_jet],
            )


def draw_leading_b_jet_pt_vs_trig_eff_hists(tagger="DL1dv01", tagger_eff="77"):
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    axs = axs.flat
    for i in np.arange(0, 4):
        ith_leading_jet = i
        for trig_short in run3_all_short:
            mass_bins = (
                leading_jets_passed_trig_btag_hists[f"{tagger}_{tagger_eff}"][
                    ith_leading_jet
                ][trig_short].edges
                * invGeV
            )
            eff, err = leading_jets_passed_trig_btag_hists[f"{tagger}_{tagger_eff}"][
                ith_leading_jet
            ][trig_short].values
            hep.histplot(
                eff * 100,
                mass_bins,
                ax=axs[i],
                histtype="errorbar",
                xerr=True,
                yerr=err * 100,
                label=trig_short,
            )
        axs[i].set_xlim(mass_bins[0], mass_bins[-1])
        axs[i].set_ylim(0, 105)
        axs[i].set_xlabel(f"jet_{i+1} " + r"$p_{T}$ [GeV]")
        axs[i].set_ylabel("Efficiency [%]")
        handles, labels = axs[i].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.98, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    fig.savefig(f"trig_eff_vs_jet_pt_{tagger}_{tagger_eff}.png", bbox_inches="tight")
    plt.close()


def draw_leading_jet_pt_vs_trig_eff_hists():
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    axs = axs.flat
    for i in np.arange(0, 4):
        ith_leading_jet = i
        for trig_short in run3_all_short:
            mass_bins = (
                leading_jets_passed_trig_hists[ith_leading_jet][trig_short].edges
                * invGeV
            )
            eff, err = leading_jets_passed_trig_hists[ith_leading_jet][
                trig_short
            ].values
            hep.histplot(
                eff * 100,
                mass_bins,
                ax=axs[i],
                histtype="errorbar",
                xerr=True,
                yerr=err * 100,
                label=trig_short,
            )
        axs[i].set_ylim(0, 105)
        axs[i].set_xlim(mass_bins[0], mass_bins[-1])
        axs[i].set_xlabel(f"jet_{i+1} " + r"$p_{T}$ [GeV]")
        axs[i].set_ylabel("Efficiency [%]")
        handles, labels = axs[i].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.98, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    fig.savefig("trig_eff_vs_jet_pt.png", bbox_inches="tight")
    plt.close()


def draw_truth_diHiggs_hists():
    fig, ax = plt.subplots()
    hep.histplot(
        truth_diHiggs_mass_hist.values / np.sum(truth_diHiggs_mass_hist.values),
        truth_diHiggs_mass_hist.edges * invGeV,
        ax=ax,
        label=r"$\kappa_{\lambda}=?$",
    )
    ax.legend()
    ax.set_ylabel("Events")
    ax.set_xlabel(r"$m_{hh}$ [GeV]")
    fig.savefig("truth_diHiggs_mass.png", bbox_inches="tight")
    plt.close()


def draw_truth_diHiggs_trig_eff_hists():
    fig, ax = plt.subplots()
    num_trig_sets = len(trig_sets_hists.keys())
    cm = plt.get_cmap("gist_ncar")
    for i_trig_set, trig_set_key in enumerate(trig_sets_hists.keys()):
        mass_ranges = trig_sets_hists[trig_set_key].edges * invGeV
        eff, err = trig_sets_hists[trig_set_key].values
        hep.histplot(
            eff * 100,
            mass_ranges,
            ax=ax,
            histtype="errorbar",
            xerr=True,
            yerr=err * 100,
            label=trig_set_key,
            color=cm(1.0 * i_trig_set / num_trig_sets),
            # markersize=5.0,
            # elinewidth=0.8,
        )
    ax.legend(loc="lower center")
    ax.set_ylim(0, 105)
    ax.set_xlabel(r"$m_{HH}$ [GeV]")
    ax.set_ylabel("Trigger Efficiency [%]")

    fig.savefig("trig_eff_vs_truth_diHiggs_mass.png", bbox_inches="tight")
    plt.close()


def draw_truth_diHiggs_trig_eff_btag_hists(tagger="DL1dv01", tagger_eff="77"):
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, height_ratios=(20, 10), sharex=True, constrained_layout=True
    )
    trig_sets_btagger_hists = trig_sets_btag_hists[f"{tagger}_{tagger_eff}"]
    num_trig_sets = len(trig_sets_btagger_hists.keys())
    cm = plt.get_cmap("gist_ncar")
    for i_trig_set, trig_set_key in enumerate(trig_sets_btagger_hists.keys()):
        bins = trig_sets_btagger_hists[trig_set_key].edges * invGeV
        eff, err = trig_sets_btagger_hists[trig_set_key].values
        hep.histplot(
            eff * 100,
            bins,
            ax=ax_top,
            histtype="errorbar",
            xerr=True,
            yerr=err * 100,
            label=trig_set_key,
            color=cm(1.0 * i_trig_set / num_trig_sets),
            # markersize=5.0,
            # elinewidth=0.8,
        )
        run2_key = "Run 2"
        if trig_set_key != run2_key:
            ratio = eff / trig_sets_btagger_hists[run2_key].values[0]
            hep.histplot(
                ratio,
                bins,
                ax=ax_bottom,
                histtype="errorbar",
                xerr=False,
                yerr=False,
                label=trig_set_key,
                color=cm(1.0 * i_trig_set / num_trig_sets),
            )
    ax_top.set_ylim(0, 105)
    ax_top.legend(loc="lower center")
    ax_top.set_ylabel("Trigger Efficiency [%]")
    ax_bottom.set_xlabel(r"$m_{HH}$ [GeV]")
    ax_bottom.set_ylabel("Ratio to Run2")
    ax_bottom.set_ylim(0.5, 2.1)
    ax_bottom.axhline(y=1.0, color="black")

    fig.savefig(
        f"trig_eff_vs_truth_diHiggs_mass_4btags_{tagger}_{tagger_eff}.png",
        bbox_inches="tight",
    )
    plt.close()


def draw_jet_pt_hists():
    fig, ax = plt.subplots()
    bins = leading_jets_pt_hist.edges
    labels = ["Leading", "Subleading", "Third Leading", "Fourth Leading"]
    for ith_leading_jet_pt, label in zip(leading_jets_pt_hist.values, labels):
        hep.histplot(
            ith_leading_jet_pt,
            bins * invGeV,
            label=label,
            ax=ax,
        )

    ax.legend(loc="upper right")
    ax.set_xlabel(r"$p_{T}$ [GeV]")
    ax.set_ylabel("Jets")
    ax.set_yscale("log")
    fig.savefig("leading_jets_pt.png", bbox_inches="tight")
    plt.close()


def draw_b_jet_pt_hists(tagger="DL1dv01", tagger_eff="77"):
    fig, ax = plt.subplots()
    bins = leading_b_jets_pt_hists[f"{tagger}_{tagger_eff}"].edges
    labels = ["Leading", "Subleading", "Third Leading", "Fourth Leading"]
    for ith_leading_jet_pt, label in zip(
        leading_b_jets_pt_hists[f"{tagger}_{tagger_eff}"].values, labels
    ):
        hep.histplot(
            ith_leading_jet_pt,
            bins * invGeV,
            label=label,
            ax=ax,
        )

    ax.legend(loc="upper right")
    ax.set_xlabel(r"$p_{T}$ [GeV]")
    ax.set_ylabel("Jets")
    ax.set_yscale("log")
    fig.savefig(f"leading_b_jets_pt{tagger}_{tagger_eff}.png", bbox_inches="tight")
    plt.close()


def fill_hists(events):
    run3_cuts = get_valid_events_mask(events, pt_cut=20_000)
    fill_truth_diHiggs_hists(events, run3_cuts)
    fill_jet_pt_hists(events, run3_cuts)
    fill_leading_jet_pt_passed_trig_hists(events, run3_cuts)
    fill_b_jet_pt_hists(events, cuts=run3_cuts, tagger="DL1dv01", tagger_eff="77")
    fill_b_jet_pt_hists(events, cuts=run3_cuts, tagger="GN120220509", tagger_eff="77")
    fill_leading_b_jet_pt_vs_trig_hists(
        events, cuts=run3_cuts, tagger="DL1dv01", tagger_eff="77"
    )
    fill_leading_b_jet_pt_vs_trig_hists(
        events, cuts=run3_cuts, tagger="GN120220509", tagger_eff="77"
    )
    fill_truth_diHiggs_btags_hists(
        events, cuts=run3_cuts, tagger="DL1dv01", tagger_eff="77"
    )
    fill_truth_diHiggs_btags_hists(
        events, cuts=run3_cuts, tagger="GN120220509", tagger_eff="77"
    )
    # run2_cuts = get_valid_events_mask(events, pt_cut=40_000)
    # fill_truth_diHiggs_btags_hists(
    #     events, cuts=run2_cuts, tagger="DL1dv01", tagger_eff="77"
    # )
    # fill_truth_diHiggs_btags_hists(
    #     events, cuts=run2_cuts, tagger="GN120220509", tagger_eff="77"
    # )


def draw_hists():
    draw_truth_diHiggs_hists()
    draw_truth_diHiggs_trig_eff_hists()
    draw_jet_pt_hists()
    draw_leading_jet_pt_vs_trig_eff_hists()
    draw_b_jet_pt_hists(tagger="DL1dv01", tagger_eff="77")
    draw_b_jet_pt_hists(tagger="GN120220509", tagger_eff="77")
    draw_leading_b_jet_pt_vs_trig_eff_hists(tagger="DL1dv01", tagger_eff="77")
    draw_leading_b_jet_pt_vs_trig_eff_hists(tagger="GN120220509", tagger_eff="77")
    draw_truth_diHiggs_trig_eff_btag_hists(tagger="DL1dv01", tagger_eff="77")
    draw_truth_diHiggs_trig_eff_btag_hists(tagger="GN120220509", tagger_eff="77")


def run():
    # samples = {"mc21_ggF_k10_small": mc21_ggF_k10_small}
    # samples = {"mc21_ggF_k01_small": mc21_ggF_k01_small}
    # samples = {"mc21_ggF_k01": mc21_ggF_k01}
    samples = {"mc21_ggF_k10": mc21_ggF_k10}
    vars = ["pt", "eta", "phi", "m"]
    for sample_name, sample_path in samples.items():
        for events, report in uproot.iterate(
            [f"{sample_path}*.root:AnalysisMiniTree"],
            [
                *[f"recojet_antikt4_NOSYS_{var}" for var in vars],
                *[f"truth_H1_{var}" for var in vars],
                *[f"truth_H2_{var}" for var in vars],
                *[f"trigPassed_{trig}" for trig in run3_all],
                "recojet_antikt4_NOSYS_DL1dv01_FixedCutBEff_77",
                "recojet_antikt4_NOSYS_GN120220509_FixedCutBEff_77",
            ],
            step_size="1 GB",
            report=True,
            # decompression_executor=executor,
            # interpretation_executor=executor,
        ):
            print(report)
            fill_hists(events)
    draw_hists()


# def run():
#     vars = ["pt", "eta", "phi", "m"]
#     for events, report in uproot.iterate(
#         [
#             f"{mc21_ggF_k01_small}*.root:AnalysisMiniTree",
#             f"{mc21_ggF_k10_small}*.root:AnalysisMiniTree",
#         ],
#         [
#             *[f"recojet_antikt4_NOSYS_{var}" for var in vars],
#             *[f"truth_H1_{var}" for var in vars],
#             *[f"truth_H2_{var}" for var in vars],
#             *[f"trigPassed_{trig}" for trig in run3_all],
#             "recojet_antikt4_NOSYS_DL1dv01_FixedCutBEff_77",
#             "recojet_antikt4_NOSYS_GN120220509_FixedCutBEff_77",
#         ],
#         step_size="1 GB",
#         report=True,
#     ):
#         print(dir(report))
#         fill_hists(events)
#     draw_hists()


if __name__ == "__main__":
    run()
