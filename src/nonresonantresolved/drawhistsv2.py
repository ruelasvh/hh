import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import re
from .utils import find_hist, find_all_hists, inv_GeV, nth, kin_labels
from .selection import X_HH, R_CR
from shared.utils import logger

plt.style.use(hep.style.ATLAS)


def find_hists(
    iteratable,
    pred=None,
):
    """Returns the found values in the iterable given pred.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.

    """
    return list(filter(pred, iteratable))


def draw_hists(hists_group) -> None:
    """Draw all the histrograms"""

    draw_1d_hists(
        hists_group={key: value for key, value in hists_group.items() if "ggF" in key},
        hist_prefix="hh_deltaeta_baseline",
        xlabel="$\Delta\eta_{HH}$",
        ylabel="Events",
        label=", no $\mathrm{X}_{\mathrm{Wt}}$ cut",
        xcut=1.5,
    )
    draw_1d_hists(
        hists_group,
        hist_prefix="top_veto_baseline",
        xlabel="$\mathrm{X}_{\mathrm{Wt}}$",
        ylabel="Events",
        xcut=1.5,
    )
    draw_1d_hists(
        hists_group={key: value for key, value in hists_group.items() if "ggF" in key},
        hist_prefix="hh_mass_discrim_baseline",
        xlabel="$\mathrm{X}_{\mathrm{HH}}$",
        ylabel="Events",
        xcut=1.6,
    )
    for sample_type, sample_hists in hists_group.items():
        draw_jet_kin_hists(
            sample_hists=sample_hists,
            sample_name=sample_type + "_baseline",
            yscale="log",
        )
        draw_mH_1D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type + "_baseline",
            hist_prefix="mH[12]_baseline$",
            xlim=(90, 150),
        )
        draw_mH_plane_2D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type + "_baseline",
            hist_prefix="mH_plane_baseline$",
        )
        draw_mH_1D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type + "_baseline_signal_region",
            hist_prefix="mH[12]_baseline_signal_region$",
        )
        draw_mH_plane_2D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type + "_baseline_signal_region",
            hist_prefix="mH_plane_baseline_signal_region$",
        )
        draw_mH_1D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type + "_baseline_control_region",
            hist_prefix="mH[12]_baseline_control_region$",
        )
        draw_mH_plane_2D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type + "_baseline_control_region",
            hist_prefix="mH_plane_baseline_control_region$",
        )


def draw_1d_hists(
    hists_group,
    hist_prefix,
    xlabel=None,
    ylabel="Frequency",
    label=None,
    yscale="linear",
    xcut=None,
):
    """Draw 1D histograms in one figure. The number of histograms in the figure is
    determined by the number of samples in the hists_group dictionary. hist_prefix
    is used to select the histograms to be drawn."""

    fig, ax = plt.subplots()
    for sample_type, sample_hists in hists_group.items():
        hist_name = find_hist(sample_hists, lambda h: hist_prefix in h)
        hist = sample_hists[hist_name]
        hep.histplot(
            hist["values"],
            hist["edges"],
            ax=ax,
            label=sample_type,
            linewidth=2.0,
        )
    if xcut:
        ax.axvline(x=xcut, ymax=0.6, color="purple")
    ax.legend()
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin=ymin, ymax=ymax * 1.5)
    hep.atlas.label(loc=1, ax=ax, label=label, rlabel="")
    fig.savefig(f"plots/{hist_prefix}.png", bbox_inches="tight")
    plt.close()


def draw_mH_1D_hists(sample_hists, sample_name, hist_prefix, xlim=None):
    fig, axs = plt.subplots(2, constrained_layout=True)
    axs = axs.flat
    hists = find_hists(sample_hists, lambda h: re.match(hist_prefix, h))
    for i in range(len(hists)):
        ax = axs[i]
        logger.debug(hists[i])
        hist = sample_hists[hists[i]]
        hep.histplot(
            hist["values"][:],
            hist["edges"][:] * inv_GeV,
            histtype="fill",
            stack=True,
            ax=ax,
            color="silver",
            label=sample_name,
        )
        if xlim:
            ax.set_xlim(*xlim)
        ax.legend()
        ax.set_xlabel("$m_{H" + str(i + 1) + "}$ [GeV]")
        ax.set_ylabel("Frequency")
        hep.atlas.label(loc=1, ax=ax, rlabel="")
    fig.savefig(f"plots/mH_{sample_name}.png", bbox_inches="tight")
    plt.close()


def draw_mH_plane_2D_hists(sample_hists, sample_name, hist_prefix):
    fig, ax = plt.subplots()
    hist_name = find_hist(sample_hists, lambda h: re.match(hist_prefix, h))
    hist = sample_hists[hist_name]
    binsGeV = hist["edges"][:] * inv_GeV
    hep.hist2dplot(
        hist["values"],
        binsGeV,
        binsGeV,
        ax=ax,
        cbarpad=0.15,
        cbarsize="5%",
    )
    ax.set_ylabel(r"$m_{H2}$ [GeV]")
    ax.set_xlabel(r"$m_{H1}$ [GeV]")
    ax.set_ylim(50, 200)
    ax.set_xlim(50, 200)
    X, Y = np.meshgrid(binsGeV, binsGeV)
    X_HH_discrim = X_HH(X, Y)
    ax.contour(
        X,
        Y,
        X_HH_discrim,
        levels=[1.55, 1.6],
        colors=["red", "black"],
        linestyles=["solid", "dashed"],
    )
    R_CR_discrim = R_CR(X, Y)
    ax.contour(
        X,
        Y,
        R_CR_discrim,
        levels=[45],
        colors=["black"],
        linestyles=["dashed"],
    )
    hep.atlas.label(loc=0, ax=ax, label=sample_name, com=13.6)
    fig.savefig(
        f"plots/mH_plane_{sample_name}.png",
        bbox_inches="tight",
    )
    plt.close()


def draw_jet_kin_hists(sample_hists, sample_name, yscale="linear"):
    jet_vars = ["pt"]
    for jet_var in jet_vars:
        fig, ax = plt.subplots()
        hist_name = find_hist(sample_hists, lambda h: f"jet_{jet_var}" in h)
        logger.debug(hist_name)
        hist = sample_hists[hist_name]
        hep.histplot(
            hist["values"][:],
            hist["edges"][:] * inv_GeV,
            ax=ax,
            label=sample_name,
        )
        ax.set_yscale("log")
        ax.set_ylabel("Frequency")
        ax.set_yscale(yscale)
        ax.legend()
        ax.set_xlabel(f"jet {kin_labels[jet_var]} [GeV]")
        fig.savefig(f"plots/jet_{jet_var}_{sample_name}.png", bbox_inches="tight")
        plt.close()
