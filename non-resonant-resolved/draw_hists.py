import matplotlib.pyplot as plt
import mplhep as hep
import triggers
import numpy as np

plt.style.use(hep.style.ATLAS)

invGeV = 1 / 1_000


def draw_var_vs_eff_hists(hists, sample_name, var_label, eff_label, y_lims=None):
    n_plots = len(hists.keys())
    fig, axs = plt.subplots(2, n_plots // 2, constrained_layout=True)
    axs = axs.flat
    for ith_var, passed_trig_hists in hists.items():
        ax = axs[ith_var]
        for key, hist in passed_trig_hists.items():
            label = eff_label
            if "trig" in eff_label:
                trig_index = triggers.run3_all.index(key)
                label = triggers.run3_all_short[trig_index]
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
                label=f"{label}: " + r"$\epsilon = $" + f"{(total_eff)}%",
            )
            hep.histplot(
                (total / np.amax(total)),
                bins,
                histtype="fill",
                stack=True,
                ax=ax,
                color="silver",
            )
        if y_lims:
            y_lims(ith_var, ax)
        ax.set_ylim(ax.get_ylim()[0], 1.1)
        ax.set_xlabel(f"{var_label}{ith_var+1} [GeV]")
        ax.set_ylabel(f"{eff_label.capitalize()} Efficiency")
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.98, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    fig.savefig(
        f"plots/{var_label}_vs_{eff_label}_eff_{sample_name}.png", bbox_inches="tight"
    )
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


def draw_signal_region(ax, m_h1, m_h2):
    xhh = np.sqrt(
        np.power((m_h1 - 124_000) / (0.1 * m_h1), 2)
        + np.pwer((m_h2 - 117_000) / (0.1 * m_h2), 2)
    )
    xhh = xhh[xhh < 1.6]
    ax.contour(xhh, [1.6])


def setAxLabels(ax, var_label):
    ax.set_ylabel(f"{var_label}2 [GeV]", fontsize=10)
    ax.set_xlabel(f"{var_label}1 [GeV]", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.set_aspect("equal", adjustable="box")


getSampleLabel = (
    lambda l: r"$\kappa_{\lambda} = 1$" if l == "k01" else r"$\kappa_{\lambda} = 10$"
)


def draw_2d_var_hist(hist, var_label="", eff_label="", sample_name="", extra_label=""):
    fig, ax = plt.subplots(constrained_layout=True)
    if eff_label == "inclusive":
        _, total = hist._hist
        vals = total / np.amax(total)
    else:
        vals, _ = hist.values
    bins = hist.edges * invGeV
    pc, cb, _ = hep.hist2dplot(
        vals,
        bins,
        bins,
        ax=ax,
        cbarpad=0.15,
        cbarsize="5%",
    )
    ax.set_ylabel(f"{var_label}2 [GeV]")
    ax.set_xlabel(f"{var_label}1 [GeV]")
    ax.set_aspect("equal", adjustable="box")
    hep.atlas.label(loc=1)
    fig.savefig(
        f"plots/{var_label}_plane_vs_{eff_label}_trig_eff_{sample_name}_single.png",
        bbox_inches="tight",
    )
    plt.close()


def draw_2d_var_vs_trig_eff_hists(
    hists, sample_name, var_label, eff_label="", y_lims=None
):

    fig, axs = plt.subplots(2, 3, tight_layout=True)
    axs = axs.flat
    for ith_var, passed_trig_hists in hists.items():
        for ith_trig, trig_name, hist in zip(
            np.arange(0, len(triggers.run3_all_short)),
            triggers.run3_all_short,
            passed_trig_hists.values(),
        ):
            ax = axs[ith_trig]
            ax.set_title(f"{getSampleLabel(sample_name)}, {trig_name}", fontsize=8)
            setAxLabels(ax, var_label)
            eff, _ = hist.values
            bins = hist.edges * invGeV
            _, total = hist._hist
            pc, cb, _ = hep.hist2dplot(
                eff,
                bins,
                bins,
                ax=ax,
                cbarpad=0.15,
                cbarsize="5%",
            )
            cb.ax.tick_params(labelsize=10)
        if eff_label == "inclusive":
            vals = total / np.amax(total)
            extra_label = "invariant mass plane"
        else:
            hist = list(passed_trig_hists.values())[-1]
            vals, _ = hist.values
            bins = hist.edges * invGeV
            extra_label = "OR of all triggers"
        draw_2d_var_hist(hist, var_label, eff_label, sample_name)
        pc, cb, _ = hep.hist2dplot(
            vals,
            bins,
            bins,
            ax=axs[-1],
            cbarpad=0.15,
            cbarsize="5%",
        )
        axs[-1].set_title(
            f"{getSampleLabel(sample_name)}, {extra_label}",
            fontsize=10,
        )
        setAxLabels(axs[-1], var_label)
        cb.ax.tick_params(labelsize=10)
    fig.savefig(
        f"plots/{var_label}_plane_vs_{eff_label}_trig_eff_{sample_name}.png",
        bbox_inches="tight",
    )
    plt.close()


def draw_hists(outputs):
    for sample_name, hists_dict in outputs.items():
        if "leading_jets_passed_trig_hists" in hists_dict:
            draw_var_vs_eff_hists(
                hists_dict["leading_jets_passed_trig_hists"],
                sample_name,
                var_label=r"jet",
                eff_label="trigger",
                y_lims=set_leading_jets_y_lims,
            )

    for sample_name, hists_dict in outputs.items():
        if "mH_passed_trig_hists" in hists_dict:
            draw_var_vs_eff_hists(
                hists_dict["mH_passed_trig_hists"],
                sample_name,
                var_label="mH",
                eff_label="trigger",
            )

    for sample_name, hists_dict in outputs.items():
        if "mH_plane_passed_trig_hists" in hists_dict:
            draw_2d_var_vs_trig_eff_hists(
                hists_dict["mH_plane_passed_trig_hists"],
                sample_name,
                var_label="mH",
                eff_label="inclusive",
            )

    for sample_name, hists_dict in outputs.items():
        if "mH_plane_passed_exclusive_trig_hists" in hists_dict:
            draw_2d_var_vs_trig_eff_hists(
                hists_dict["mH_plane_passed_exclusive_trig_hists"],
                sample_name,
                var_label="mH",
                eff_label="exclusive",
            )

    for sample_name, hists_dict in outputs.items():
        if "mH_passed_pairing_hists" in hists_dict:
            draw_var_vs_eff_hists(
                hists_dict["mH_passed_pairing_hists"],
                sample_name,
                var_label="mH",
                eff_label="pairing",
            )
