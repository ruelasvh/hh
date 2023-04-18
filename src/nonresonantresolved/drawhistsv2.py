import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import re
from .utils import find_hist, find_all_hists, inv_GeV, nth, kin_labels
from .selection import X_HH, R_CR
from .error import get_efficiency_with_uncertainties
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
        hists_group,
        hist_prefix="hh_deltaeta_baseline",
        xlabel="$\Delta\eta_{HH}$",
        label=", no $\mathrm{X}_{\mathrm{Wt}}$ cut",
    )
    draw_1d_hists(
        hists_group,
        hist_prefix="top_veto_baseline",
        xlabel="$\mathrm{X}_{\mathrm{Wt}}$",
    )
    draw_1d_hists(
        hists_group,
        hist_prefix="hh_mass_discrim_baseline",
        xlabel="$\mathrm{X}_{\mathrm{HH}}$",
    )
    for sample_type, sample_hists in hists_group.items():
        draw_mH_1D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type + "_baseline",
        )
        draw_mH_plane(
            sample_hists=sample_hists,
            sample_name=sample_type + "_baseline",
        )


def draw_1d_hists(
    hists_group, hist_prefix, xlabel=None, ylabel="Frequency", label=None
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
        )
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    hep.atlas.label(loc=1, ax=ax, label=label, rlabel="")
    fig.savefig(f"plots/{hist_prefix}.png", bbox_inches="tight")
    plt.close()


def draw_mH_1D_hists(sample_hists, sample_name):
    fig, axs = plt.subplots(2, constrained_layout=True)
    axs = axs.flat
    hists = find_hists(sample_hists, lambda h: re.match("mH[12]_baseline", h))
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
            label="",
        )
        ax.set_xlabel("$m_{H" + str(i + 1) + "}$ [GeV]")
        ax.set_ylabel("Frequency")
        hep.atlas.label(loc=1, ax=ax, label=sample_name, rlabel="")
    fig.savefig(f"plots/mH_{sample_name}.png", bbox_inches="tight")
    plt.close()


def draw_mH_plane(sample_hists, sample_name):
    fig, ax = plt.subplots()
    hist_name = find_hist(sample_hists, lambda h: "mH_plane_baseline" in h)
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
    ax.contour(X, Y, R_CR_discrim, levels=[45], colors=["black"], linestyles=["dashed"])
    hep.atlas.label(loc=0, ax=ax, label=sample_name, com=13.6)
    fig.savefig(
        f"plots/mH_plane_{sample_name}.png",
        bbox_inches="tight",
    )
    plt.close()
