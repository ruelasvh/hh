import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from .utils import find_hist, find_all_hists, inv_GeV, nth, kin_labels
from .selection import X_HH, R_CR
from .error import get_efficiency_with_uncertainties
from shared.utils import logger

plt.style.use(hep.style.ATLAS)


def draw_hists(hists: list, sample_name: str, args: dict) -> None:
    """Draw all the histrograms"""

    logger.info(f"Drawing hitograms for sample type: {sample_name}")
    draw_jet_kin_hists(hists, sample_name)
    draw_leading_jet_hists(hists, sample_name)
    draw_mH_plane(
        hist=find_hist(hists, lambda h: "mH_plane_baseline" in h.name),
        sample_name=sample_name + "_baseline",
    )
    draw_mH_plane(
        hist=find_hist(hists, lambda h: "mH_plane_framework" in h.name),
        sample_name=sample_name + "_framework",
    )
    draw_mH_1D_hists(
        find_all_hists(hists, "mH[12]_baseline"), sample_name + "_baseline"
    )
    draw_mH_1D_hists(
        find_all_hists(hists, "mH[12]_framework"), sample_name + "_framework"
    )
    draw_hh_deltaeta_hists(
        hists=find_all_hists(hists, "hh_deltaeta_baseline"),
        sample_name=sample_name + "_baseline",
    )
    if args.signal:
        draw_var_vs_eff_hists(hists, sample_name, "H")
        # TODO: Fix efficiencies going above 1 in fillhists.py
        # draw_var_vs_eff_hists(hists, sample_name, "jj")


def draw_jet_kin_hists(hists, sample_name):
    jet_vars = ["pt"]
    for jet_var in jet_vars:
        fig, ax = plt.subplots()
        hist = find_hist(hists, lambda h: f"jet_{jet_var}" in h.name)
        logger.debug(hist.name)
        binsGeV = hist.edges * inv_GeV
        hep.histplot(
            hist.values,
            binsGeV,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_ylabel("Frequency")
        ax.set_xlabel(f"jet {kin_labels[jet_var]} [GeV]")
        fig.savefig(f"plots/jet_{jet_var}_{sample_name}.png", bbox_inches="tight")
        plt.close()


def draw_leading_jet_hists(hists, sample_name):
    jet_vars = ["pt"]
    for jet_var in jet_vars:
        fig, ax = plt.subplots()
        for ith_jet in range(1, 5):
            hist = find_hist(
                hists, lambda h: f"leading_jet_{ith_jet}_{jet_var}" in h.name
            )
            logger.debug(hist.name)
            binsGeV = hist.edges * inv_GeV
            hep.histplot(
                hist.values,
                binsGeV,
                ax=ax,
                label=f"{nth[ith_jet]} leading jet",
            )
        ax.legend()
        ax.set_ylabel("Frequency")
        ax.set_xlabel(f"jet {kin_labels[jet_var]} [GeV]")
        fig.savefig(
            f"plots/leading_jet_{jet_var}_{sample_name}.png", bbox_inches="tight"
        )
        plt.close()


def draw_mH_plane(hist, sample_name):
    fig, ax = plt.subplots()
    binsGeV = hist.edges * inv_GeV
    hep.hist2dplot(
        hist.values,
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


def draw_mH_1D_hists(hists, sample_name):
    fig, axs = plt.subplots(2, constrained_layout=True)
    axs = axs.flat
    for i in range(len(hists)):
        ax = axs[i]
        logger.debug(hists[i].name)
        hep.histplot(
            hists[i].values,
            hists[i].edges * inv_GeV,
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


def draw_hh_deltaeta_hists(hists, sample_name):
    fig, ax = plt.subplots()
    logger.debug(hists[0].name)
    hep.histplot(
        hists[0].values,
        hists[0].edges,
        ax=ax,
        label="",
    )
    ax.set_xlabel("$\Delta\eta_{hh}$")
    ax.set_ylabel("Frequency")
    hep.atlas.label(loc=1, ax=ax, label=sample_name, rlabel="")
    fig.savefig(f"plots/hh_deltaeta_{sample_name}.png", bbox_inches="tight")
    plt.close()


def draw_var_vs_eff_hists(hists, sample_name, var_label):
    fig, axs = plt.subplots(2, constrained_layout=True)
    axs = axs.flat
    for ith_var in [1, 2]:
        hist_passed = find_hist(
            hists, lambda h: f"m{var_label}{ith_var}_pairingPassedTruth" in h.name
        )
        hist_total = find_hist(hists, lambda h: f"m{var_label}{ith_var}" in h.name)
        ax = axs[ith_var - 1]
        bins = hist_total.edges * inv_GeV
        total = hist_total.values
        logger.debug(hist_total.name)
        logger.debug(total)
        passed = hist_passed.values
        logger.debug(hist_passed.name)
        logger.debug(passed)
        eff, err = get_efficiency_with_uncertainties(passed, total)
        total_eff = np.sum(passed) / np.sum(total)
        total_eff = round(total_eff * 100)
        # eff plot
        hep.histplot(
            eff,
            bins,
            ax=ax,
            histtype="errorbar",
            xerr=True,
            yerr=err,
            label="$\epsilon = " + f"{(total_eff)}$%",
        )
        # kinematic plot
        hep.histplot(
            (total / np.amax(total)),
            bins,
            histtype="fill",
            stack=True,
            ax=ax,
            color="silver",
            label="",
        )
        ax.set_ylim(ax.get_ylim()[0], 1.4)
        ax.set_xlabel("$m_{" + var_label + str(ith_var) + "}$ [GeV]")
        ax.set_ylabel("Pairing Efficiency")
        ax.legend(loc="upper right")
        hep.atlas.label(loc=1, ax=ax, label=sample_name, rlabel="")
    fig.savefig(
        f"plots/m{var_label}_vs_pairing_eff_{sample_name}.png", bbox_inches="tight"
    )
    plt.close()
