import uproot
import numpy as np
import awkward as ak
import vector as p4
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ATLAS)

ggF_mc_21 = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/run/analysis-variables.root"

trig_list = [
    "trigPassed_HLT_j150_2j55_0eta290_020jvt_bdl1r70_pf_ftf_preselj80XX2j45b90_L1J85_3J30"
]


invGeV = 1 / 1000


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


def draw_truth_diHiggs_mass(ax, mass):
    h, bins = np.histogram(mass * invGeV, 100)
    hep.histplot(h, bins, ax=ax)


def draw_trig_eff(ax, diHiggs, data):
    diHiggs_mass = ak.ravel(diHiggs.mass)
    mass_res = 50
    mass_bins = np.arange(200, 1400, mass_res)
    h_tot, _ = np.histogram(diHiggs_mass * invGeV, mass_bins)
    for trig in trig_list:
        passed_trig = diHiggs_mass[data[trig]]
        print("passed trig: ", passed_trig)
        h_pass, _ = np.histogram(passed_trig * invGeV, mass_bins)
        eff = (h_pass / h_tot) * 100
        hep.histplot(
            eff,
            mass_bins,
            ax=ax,
            histtype="errorbar",
            xerr=mass_res / 2,
            yerr=np.sqrt(eff),
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
        ax.set_ylim(0)
        ax.set_xlabel(r"$m_{HH}$ [GeV]")
        ax.set_ylabel("Trigger efficiency")


def draw_plots(data_dict):
    fig_diHiggs_mass, ax_diHiggs_mass = plt.subplots()
    fig_trig_eff, ax_trig_eff = plt.subplots()

    pt_cut = 40
    eta_cut = 2.5
    cuts_label = f"$p_T$ > {pt_cut} GeV and $|\eta| < {eta_cut}$"
    pt_cut = pt_cut / invGeV
    for sample_name, data in data_dict.items():
        cuts = (data["recojet_antikt4_NOSYS_pt"] > pt_cut) & (
            np.abs(data["recojet_antikt4_NOSYS_eta"]) < eta_cut
        )
        diHiggs = get_truth_diHiggs(data)
        draw_truth_diHiggs_mass(ax_diHiggs_mass, diHiggs.mass)
        draw_trig_eff(ax_trig_eff, diHiggs, data)

    fig_diHiggs_mass.savefig("truth_diHiggs_mass.png", bbox_inches="tight")
    fig_trig_eff.savefig("trig_eff_vs_truth_diHiggs_massv2.png", bbox_inches="tight")


def run():
    jets_dict = {}
    samples = {"ggF_mc_21": ggF_mc_21}
    for sample_name, sample_path in samples.items():
        with uproot.open(f"{sample_path}:AnalysisMiniTree") as tree:
            jets_dict[sample_name] = tree.arrays(
                filter_name="/(recojet_antikt4_NOSYS|truth_H[12])_[pt|eta|phi|m]|trigPassed_HLT_[j80c|2j35c|j150]/i"
            )
    draw_plots(jets_dict)


if __name__ == "__main__":
    run()
