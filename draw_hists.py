import matplotlib.pyplot as plt
import mplhep as hep
import triggers
import numpy as np

plt.style.use(hep.style.ATLAS)

invGeV = 1 / 1_000


def draw_var_vs_trig_eff_hists(hists, sample_name, var_label, y_lims=None):
    n_plots = len(hists.items())
    fig, axs = plt.subplots(2, n_plots // 2, constrained_layout=True)
    axs = axs.flat
    for ith_var, passed_trig_hists in hists.items():
        ax = axs[ith_var]
        for trig, hist in zip(triggers.run3_all_short, passed_trig_hists.values()):
            passed, total = hist._hist
            total_eff = np.sum(passed) / np.sum(total)
            total_eff = round(total_eff * 100)
            bins = hist.edges * invGeV
            eff, err = hist.values
            hep.histplot(
                eff,
                bins,
                ax=ax,
                histtype="errorbar",
                xerr=True,
                yerr=err,
                label=f"{trig}: " + r"$\epsilon = $" + f"{(total_eff)}%",
            )
            hep.histplot(
                (total / np.sum(total)),
                bins,
                histtype="fill",
                stack=True,
                ax=ax,
                color="silver",
            )
        if y_lims:
            y_lims(ith_var, ax)
        ax.set_yscale("log")
        ax.set_ylim(ax.get_ylim()[0], 1)
        ax.set_xlabel(f"{var_label}{ith_var+1} [GeV]")
        ax.set_ylabel("Trigger Efficiency")
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.98, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    fig.savefig(f"{var_label}_vs_trig_eff_{sample_name}.png", bbox_inches="tight")
    plt.close()


def set_leading_jets_y_lims(ith_var, ax):
    if ith_var == 0:
        ax.set_xlim(0, 250)
    if ith_var == 1:
        ax.set_xlim(0, 250)
    if ith_var == 2:
        ax.set_xlim(0, 170)
    if ith_var == 3:
        ax.set_xlim(0, 150)


def draw_hists(outputs):
    for sample_name, hists_dict in outputs.items():
        if "leading_jets_passed_trig_hists" in hists_dict:
            draw_var_vs_trig_eff_hists(
                hists_dict["leading_jets_passed_trig_hists"],
                sample_name,
                var_label=r"jet",
                y_lims=set_leading_jets_y_lims,
            )

    for sample_name, hists_dict in outputs.items():
        if "mH_passed_trig_hists" in hists_dict:
            draw_var_vs_trig_eff_hists(
                hists_dict["mH_passed_trig_hists"],
                sample_name,
                var_label="mH",
            )
