import uproot
import numpy as np
import awkward as ak
import vector as p4
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ATLAS)

ggF_mc_21 = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/run/analysis-variables-run3.root"


run3_all_short = [
    "Asymm 2b2j DL1d@77%",
    "Asymm 3b1j DL1d@82%",
    "2b1j DL1d@70%",
    "Symm 2b2j DL1d@60%",
    "Asymm 2b2j+L1mu DL1d@77%",
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
    "run2": run2,
    "run3_main_stream": run3_main_stream,
    "run3_delayed_stream": run3_delayed_stream,
    "run3_asymm_L1_jet": run3_asymm_L1_jet,
    "run3_asymm_L1_all": run3_asymm_L1_all,
    "run3_all": run3_all,
}

invGeV = 1 / 1_000


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


def get_valid_jets_pt(data):
    jets_pt = data["recojet_antikt4_NOSYS_pt"]
    jets_eta = data["recojet_antikt4_NOSYS_eta"]
    jets_pt_valid = jets_pt[(jets_pt > 20_000) & (np.abs(jets_eta) < 2.5)]
    jets_pt_valid = jets_pt_valid[ak.num(jets_pt_valid) > 3]
    return jets_pt_valid


def get_valid_jets_mask(data):
    jets_pt = data["recojet_antikt4_NOSYS_pt"]
    jets_eta = data["recojet_antikt4_NOSYS_eta"]
    jets_pt_valid = jets_pt[(jets_pt > 25_000) & (np.abs(jets_eta) < 2.5)]
    jets_pt_valid_mask = ak.num(jets_pt_valid) > 3
    return jets_pt_valid_mask


def draw_truth_diHiggs_mass(mass):
    fig, ax = plt.subplots()
    bins = np.arange(200, 1_000, 10)
    h, bins = np.histogram(mass * invGeV, bins)
    hep.histplot(h / np.sum(h), bins, ax=ax, label=r"$\kappa_{\lambda}=?$")
    ax.legend()
    ax.set_ylabel("Events")
    ax.set_xlabel(r"$m_{hh}$ [GeV]")
    fig.savefig("truth_diHiggs_mass.png", bbox_inches="tight")


def draw_jet_pt(data, jet_cuts):
    fig, ax = plt.subplots()
    jets_pt = data["recojet_antikt4_NOSYS_pt"][jet_cuts]
    leading = jets_pt[:, 0]
    subleading = jets_pt[:, 1]
    third_leading = jets_pt[:, 2]
    fourth_leading = jets_pt[:, 3]
    mass_res = 10
    mass_bins = np.arange(20, 1000, mass_res)
    h_leading, _ = np.histogram(leading * invGeV, mass_bins)
    h_subleading, _ = np.histogram(subleading * invGeV, mass_bins)
    h_third_leading, _ = np.histogram(third_leading * invGeV, mass_bins)
    h_fourth_leading, _ = np.histogram(fourth_leading * invGeV, mass_bins)
    hep.histplot(h_leading, mass_bins, label="Leading", ax=ax)
    hep.histplot(h_subleading, mass_bins, label="Subleading", ax=ax)
    hep.histplot(
        h_third_leading,
        mass_bins,
        label="Third Leading",
        ax=ax,
    )
    hep.histplot(
        h_fourth_leading,
        mass_bins,
        label="Third Leading",
        ax=ax,
    )

    ax.legend(loc="upper right")
    ax.set_xlabel(r"$p_{T}$ [GeV]")
    ax.set_ylabel("Jets")
    ax.set_yscale("log")
    fig.savefig("leading_jets_pt.png", bbox_inches="tight")


def draw_trig_eff(diHiggs, data):
    fig, ax = plt.subplots()
    diHiggs_mass = ak.ravel(diHiggs.mass)
    mass_res = 50
    mass_bins = np.arange(250, 1300, mass_res)
    h_tot, _ = np.histogram(diHiggs_mass * invGeV, mass_bins)
    num_trig_sets = len(trig_sets.keys())
    cm = plt.get_cmap("gist_rainbow")
    for i_trig_set, trig_set_key in enumerate(trig_sets.keys()):
        trig_selections = []
        for i_trig, trig in enumerate(trig_sets[trig_set_key]):
            trig_selection = data[f"trigPassed_{trig}"]
            if i_trig == 0:
                trig_selections = trig_selection
            else:
                trig_selections = trig_selections | trig_selection
        passed_trig = diHiggs_mass[trig_selections]
        h_pass, _ = np.histogram(passed_trig * invGeV, mass_bins)
        eff = (h_pass / h_tot) * 100
        hep.histplot(
            eff,
            mass_bins,
            ax=ax,
            histtype="errorbar",
            xerr=mass_res / 2,
            yerr=np.sqrt(eff),
            label=trig_set_key,
            color=cm(1.0 * i_trig_set / num_trig_sets),
        )
        # plt.errorbar(
        #     mass_bins[:-1],
        #     eff,
        #     yerr=np.sqrt(eff),
        #     xerr=np.diff(mass_bins) / 2,
        #     fmt=".",
        #     markersize=10.0,
        #     elinewidth=1,
        # )
        ax.legend(loc="lower right")
        ax.set_xlabel(r"$m_{HH}$ [GeV]")
        ax.set_ylabel("Trigger efficiency")

    fig.savefig("trig_eff_vs_truth_diHiggs_mass.png", bbox_inches="tight")


def draw_jet_pt_vs_trig_eff(data, jet_cuts):
    jets_pt_pre_trig = data["recojet_antikt4_NOSYS_pt"][jet_cuts]
    mass_res = 10
    mass_bins = np.arange(20, 250, mass_res)
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    axs = axs.flat
    for i in np.arange(0, 4):
        for trig, trig_short in zip(run3_all, run3_all_short):
            trig_selection = data[f"trigPassed_{trig}"]
            jets_pt_post_trig = data["recojet_antikt4_NOSYS_pt"][trig_selection]
            jets_pt_post_trig = jets_pt_post_trig[jet_cuts[trig_selection]]
            h_tot, _ = np.histogram(jets_pt_pre_trig[:, i] * invGeV, mass_bins)
            h_pass, _ = np.histogram(jets_pt_post_trig[:, i] * invGeV, mass_bins)
            eff = (h_pass / h_tot) * 100
            hep.histplot(
                eff,
                mass_bins,
                ax=axs[i],
                histtype="errorbar",
                xerr=True,
                yerr=np.sqrt(eff),
                label=trig_short,
            )
        axs[i].set_ylim(-5, 115)
        axs[i].set_xlim(mass_bins[0], mass_bins[-1])
        axs[i].set_xlabel(f"jet_{i+1} " + r"$p_{T}$ [GeV]")
        axs[i].set_ylabel("Efficiency")
        handles, labels = axs[i].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.98, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    fig.savefig("trig_eff_vs_jet_pt.png", bbox_inches="tight")


def draw_jet_pt_vs_trig_eff_btag(data, tagger="DL1dv01", tagger_eff="77", cuts=None):
    jets_pt_pre_trig = data["recojet_antikt4_NOSYS_pt"][cuts]
    mass_res = 10
    mass_bins = np.arange(20, 250, mass_res)
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    axs = axs.flat
    for i in np.arange(0, 4):
        for trig, trig_short in zip(run3_all, run3_all_short):
            trig_selection = data[f"trigPassed_{trig}"]
            jets_pt_post_trig = data["recojet_antikt4_NOSYS_pt"][trig_selection]
            jets_pt_post_trig = jets_pt_post_trig[cuts[trig_selection]]
            btags = data[f"recojet_antikt4_NOSYS_{tagger}_FixedCutBEff_{tagger_eff}"][
                trig_selection
            ]
            btags = btags[cuts[trig_selection]]
            btags_3j = ak.sum(btags, axis=-1) > 2
            jets_pt_post_trig_btagged = jets_pt_post_trig[btags_3j]
            h_tot, _ = np.histogram(jets_pt_pre_trig[:, i] * invGeV, mass_bins)
            h_pass, _ = np.histogram(
                jets_pt_post_trig_btagged[:, i] * invGeV, mass_bins
            )
            eff = (h_pass / h_tot) * 100
            hep.histplot(
                eff,
                mass_bins,
                ax=axs[i],
                histtype="errorbar",
                xerr=mass_res / 2,
                yerr=np.sqrt(eff),
                label=trig_short,
            )
        axs[i].set_ylim(-5, 85)
        axs[i].set_xlim(mass_bins[0], mass_bins[-1])
        axs[i].set_xlabel(f"jet_{i+1} " + r"$p_{T}$ [GeV]")
        axs[i].set_ylabel("Efficiency")
        handles, labels = axs[i].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.98, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    fig.savefig(f"trig_eff_vs_jet_pt_{tagger}_{tagger_eff}.png", bbox_inches="tight")


def draw_jeti_pt_vs_trig_eff_btag(data, tagger="DL1dv01", tagger_eff="77", cuts=None):
    jets_pt_pre_trig = data["recojet_antikt4_NOSYS_pt"][cuts]
    mass_res = 10
    mass_bins = np.arange(20, 250, mass_res)
    for i in np.arange(0, 4):
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, height_ratios=(20, 10), sharex=True, constrained_layout=True
        )
        for trig, trig_short in zip(run3_all, run3_all_short):
            trig_selection = data[f"trigPassed_{trig}"]
            jets_pt_post_trig = data["recojet_antikt4_NOSYS_pt"][trig_selection]
            jets_pt_post_trig = jets_pt_post_trig[cuts[trig_selection]]
            btags = data[f"recojet_antikt4_NOSYS_{tagger}_FixedCutBEff_{tagger_eff}"][
                trig_selection
            ]
            btags = btags[cuts[trig_selection]]
            btags_3j = ak.sum(btags, axis=-1) > 2
            jets_pt_post_trig_btagged = jets_pt_post_trig[btags_3j]
            h_tot, _ = np.histogram(jets_pt_pre_trig[:, i] * invGeV, mass_bins)
            h_pass, _ = np.histogram(jets_pt_post_trig[:, i] * invGeV, mass_bins)
            h_pass_btag, _ = np.histogram(
                jets_pt_post_trig_btagged[:, i] * invGeV, mass_bins
            )
            eff = (h_pass_btag / h_tot) * 100
            hep.histplot(
                eff,
                mass_bins,
                ax=ax_top,
                histtype="errorbar",
                xerr=mass_res / 2,
                yerr=np.sqrt(eff),
                label=trig_short,
            )
            # ratio = h_pass_btag / h_pass
            ratio = h_pass / h_pass_btag
            hep.histplot(
                ratio,
                mass_bins,
                ax=ax_bottom,
                histtype="errorbar",
                xerr=False,
                yerr=False,
                label=trig_short,
            )

        ax_top.set_ylabel("Efficiency")
        ax_bottom.set_ylim(0, 5)
        ax_bottom.axhline(y=1.0, color="black")
        ax_bottom.set_xlabel(f"jet_{i+1} " + r"$p_{T}$ [GeV]")
        ax_bottom.set_ylabel(f"Ratio to b-tagged")
        fig.savefig(
            f"trig_eff_vs_jet{i+1}_pt_{tagger}_{tagger_eff}.png", bbox_inches="tight"
        )


def draw_jet_pt_vs_trig_eff_all(axs, data, jet_cuts):
    jets_pt_pre_trig = data["recojet_antikt4_NOSYS_pt"][jet_cuts]
    mass_res = 50
    mass_bins = np.arange(20, 1_000, mass_res)
    trig_selections = []
    for i_trig, trig in enumerate(run3_all):
        trig_selection = data[f"trigPassed_{trig}"]
        if i_trig == 0:
            trig_selections = trig_selection
        else:
            trig_selections = trig_selections | trig_selection

    jets_pt_post_trig = data["recojet_antikt4_NOSYS_pt"][trig_selections]
    jets_pt_post_trig = jets_pt_post_trig[jet_cuts[trig_selections]]

    for i in np.arange(0, 4):
        h_tot, _ = np.histogram(jets_pt_pre_trig[:, i] * invGeV, mass_bins)
        h_pass, _ = np.histogram(jets_pt_post_trig[:, i] * invGeV, mass_bins)
        eff = (h_pass / h_tot) * 100
        hep.histplot(
            eff,
            mass_bins,
            ax=axs[i],
            histtype="errorbar",
            xerr=mass_res / 2,
            yerr=np.sqrt(eff),
        )
        axs[i].set_xlabel(f"jet_{i+1} " + r"$p_{T}$ [GeV]")
        axs[i].set_ylabel("Efficiency")


def draw_plots(data_dict):
    for sample_name, data in data_dict.items():
        diHiggs = get_truth_diHiggs(data)
        draw_truth_diHiggs_mass(diHiggs.mass)
        draw_trig_eff(diHiggs, data)
        jet_cuts = get_valid_jets_mask(data)
        draw_jet_pt(data, jet_cuts)
        draw_jet_pt_vs_trig_eff(data, jet_cuts)
        draw_jet_pt_vs_trig_eff_btag(
            data, tagger="DL1dv01", tagger_eff="77", cuts=jet_cuts
        )
        draw_jet_pt_vs_trig_eff_btag(
            data, tagger="GN120220509", tagger_eff="77", cuts=jet_cuts
        )
        draw_jeti_pt_vs_trig_eff_btag(
            data, tagger="DL1dv01", tagger_eff="77", cuts=jet_cuts
        )
        draw_jeti_pt_vs_trig_eff_btag(
            data, tagger="GN120220509", tagger_eff="77", cuts=jet_cuts
        )


def run():
    jets_dict = {}
    samples = {"ggF_mc_21": ggF_mc_21}
    for sample_name, sample_path in samples.items():
        with uproot.open(f"{sample_path}:AnalysisMiniTree") as tree:
            jets_dict[sample_name] = tree.arrays(
                filter_name="/(recojet_antikt4_NOSYS|truth_H[12])_[pt|eta|phi|m|DL1d|GN1]|trigPassed_HLT_[j80c|2j35c|j150]/i"
            )
    draw_plots(jets_dict)


if __name__ == "__main__":
    run()
