import numpy as np
import matplotlib.pyplot as plt
import mplhep as hplt
import re
from pathlib import Path
from .utils import (
    logger,
    find_hist,
    find_hists,
    inv_GeV,
    kin_labels,
    get_com_lumi_label,
)
from .selection import X_HH, R_CR

np.seterr(divide="ignore", invalid="ignore")

plt.style.use(hplt.style.ATLAS)


def draw_1d_hists(
    hists_group,
    hist_prefix,
    xlabel=None,
    ylabel="Frequency",
    third_exp_label="",
    luminosity=None,
    yscale="linear",
    ynorm_binwidth=False,
    xcut=None,
    density=False,
    ggFk01_factor=None,
    ggFk10_factor=None,
    data2b_factor=None,
    postfix=None,
    output_dir=Path("plots"),
):
    """Draw 1D histograms in one figure. The number of histograms in the figure is
    determined by the number of samples in the hists_group dictionary. hist_prefix
    is used to select the histograms to be drawn."""

    fig, ax = plt.subplots()
    for sample_type, sample_hists in hists_group.items():
        hist_name = find_hist(sample_hists, lambda h: hist_prefix in h)
        is_data = "data" in sample_type
        hist = sample_hists[hist_name]
        hist_values = (
            hist["values"][:] * luminosity
            if luminosity and not is_data
            else hist["values"][:]
        )
        hist_edges = hist["edges"][:]
        hist_edges = (
            hist_edges * inv_GeV
            if xlabel is not None and "GeV" in xlabel
            else hist_edges
        )
        bin_width = hist_edges[1] - hist_edges[0] if ynorm_binwidth else 1.0
        scale_factor = 1.0
        if ggFk01_factor and "ggF" in sample_type and "k01" in sample_type:
            scale_factor = ggFk01_factor
        if ggFk10_factor and "ggF" in sample_type and "k10" in sample_type:
            scale_factor = ggFk10_factor
        if data2b_factor and "data" in sample_type and "2b" in sample_type:
            scale_factor = data2b_factor
        hist_values = hist_values * scale_factor
        bin_norm = 1.0 / hist_values.sum() if density else bin_width
        hplt.histplot(
            hist_values * bin_norm,
            hist_edges - bin_width * 0.5,
            ax=ax,
            label=(str(scale_factor) + r"$\times$" if scale_factor > 1 else "")
            + sample_type,
            linewidth=2.0,
            density=density,
        )
        ax.set_ylabel(ylabel + " / %.2g" % bin_width if ynorm_binwidth else ylabel)
    ax.legend()
    if xcut:
        ax.axvline(x=xcut, ymax=0.6, color="purple")
        ax.text(
            xcut,
            0.55,
            f"{xcut}",
            color="purple",
            rotation=90,
            va="baseline",
            ha="right",
            transform=ax.get_xaxis_transform(),
        )
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_yscale(yscale)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin=ymin, ymax=ymax * 1.5)
    hplt.atlas.label(
        label="Work In Progress",
        rlabel=get_com_lumi_label(luminosity) + third_exp_label,
        loc=4,
        ax=ax,
        pad=0.01,
    )
    filename = output_dir / f"{hist_prefix}{'_' + postfix if postfix else ''}.png"
    fig.savefig(filename, bbox_inches="tight")
    plt.close()


def draw_mH_1D_hists_v2(
    hists_group,
    hist_prefix,
    luminosity=None,
    xlims=None,
    ylims=None,
    third_exp_label="",
    ylabel="Frequency",
    ynorm_binwidth=False,
    yscale="linear",
    region=None,
    ggFk01_factor=None,
    ggFk10_factor=None,
    output_dir=Path("plots"),
):
    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(8, 10))
    axs = axs.flat
    for sample_type, sample_hists in hists_group.items():
        is_data = "data" in sample_type
        if region == "SR" and is_data:
            continue
        hists = find_hists(sample_hists, lambda h: re.match(hist_prefix, h))
        for i in range(len(hists)):
            ax = axs[i]
            hist = sample_hists[hists[i]]
            hist_values = (
                hist["values"][:] * luminosity
                if luminosity and not is_data
                else hist["values"][:]
            )
            hist_edges = hist["edges"][:]
            bin_width = hist_edges[1] - hist_edges[0] if ynorm_binwidth else 1.0
            scale_factor = 1
            if ggFk01_factor and "ggF" in sample_type and "k01" in sample_type:
                scale_factor = ggFk01_factor
            if ggFk10_factor and "ggF" in sample_type and "k10" in sample_type:
                scale_factor = ggFk10_factor
            hplt.histplot(
                hist_values * scale_factor * 1.0 / bin_width,
                hist_edges * inv_GeV,
                histtype="step",
                stack=False,
                ax=ax,
                label=sample_type,
                # yerr=True,
            )
            ax.set_yscale(yscale)
            if xlims:
                ax.set_xlim(*xlims)
            if ylims is None:
                ylims = ax.get_ylim()
            ax.set_ylim(ylims[0], ylims[1] * 1.1)
            ax.legend()
            ax.set_xlabel("$m_{H" + str(i + 1) + "}$ [GeV]")
            ax.set_ylabel(ylabel + " / %.2g" % bin_width if ynorm_binwidth else ylabel)
            hplt.atlas.label(
                rlabel=get_com_lumi_label(luminosity) + third_exp_label,
                loc=4,
                ax=ax,
                pad=0.01,
            )
    plot_postfix = "_" + region if region else ""
    fig.savefig(f"{output_dir}/mH1_mH2{plot_postfix}.png", bbox_inches="tight")
    plt.close()


def draw_mH_1D_hists(
    sample_hists,
    sample_name,
    hist_prefix,
    luminosity=None,
    region=None,
    xlim=None,
    third_exp_label="",
    output_dir=Path("plots"),
):
    fig, axs = plt.subplots(2, constrained_layout=True)
    axs = axs.flat
    hists = find_hists(sample_hists, lambda h: re.match(hist_prefix, h))
    is_data = "data" in sample_name
    for i in range(len(hists)):
        ax = axs[i]
        logger.debug(hists[i])
        hist = sample_hists[hists[i]]
        hist_values = (
            hist["values"][:] * luminosity
            if luminosity and not is_data
            else hist["values"][:]
        )
        hist_bins_GeV = hist["edges"][:] * inv_GeV
        hplt.histplot(
            hist_values,
            hist_bins_GeV,
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
        hplt.atlas.label(
            rlabel=get_com_lumi_label(luminosity) + third_exp_label, loc=4, ax=ax
        )
    plot_postfix = sample_name + ("_" + region if region else "")
    fig.savefig(f"{output_dir}/mH_{plot_postfix}.png", bbox_inches="tight")
    plt.close()


def draw_mH_plane_2D_hists(
    sample_hists,
    sample_name,
    hist_prefix,
    region=None,
    luminosity=None,
    output_dir=Path("plots"),
):
    fig, ax = plt.subplots()
    hist_name = find_hist(sample_hists, lambda h: re.match(hist_prefix, h))
    hist = sample_hists[hist_name]
    bins_GeV = hist["edges"][:] * inv_GeV
    is_data = "data" in sample_name
    hist_values = (
        hist["values"][:] * luminosity
        if luminosity and not is_data
        else hist["values"][:]
    )
    logger.info(
        f"{region} {sample_name} {hist_name} counts: {np.sum(hist['values'][:])}"
    )
    hplt.hist2dplot(
        hist_values,
        bins_GeV,
        bins_GeV,
        ax=ax,
        cbarpad=0.15,
        cbarsize="5%",
    )
    ax.set_ylabel(r"$m_{H2}$ [GeV]")
    ax.set_xlabel(r"$m_{H1}$ [GeV]")
    ax.set_ylim(50, 200)
    ax.set_xlim(50, 200)
    X, Y = np.meshgrid(bins_GeV, bins_GeV)
    signal_region = X_HH(X, Y)
    ax.contour(
        X,
        Y,
        signal_region,
        levels=[1.55, 1.6],
        colors=["red", "black"],
        linestyles=["solid", "dashed"],
    )
    # line connecting SR and CR points x_ur, y_ur
    ax.plot(
        (139.35, 161.84),
        (132.35, 154.84),
        color="black",
        linestyle="dashed",
    )
    # line connecting SR and CR points x_dr, y_dr
    ax.plot(
        (137.24, 155.42),
        (103.76, 85.58),
        color="black",
        linestyle="dashed",
    )
    # line connecting SR and CR points x_ul, y_ul
    ax.plot(
        (110.51, 92.93),
        (130.05, 148.07),
        color="black",
        linestyle="dashed",
    )
    # line connecting SR and CR points x_dl, y_dl
    ax.plot(
        (111.77, 98.21),
        (104.77, 91.21),
        color="black",
        linestyle="dashed",
    )
    # ax.axline((124, 117), (125 + 45, 117 + 45), color="black", linestyle="dashed")
    # ax.axline((124, 117), slope=1, color="black", linestyle="dashed")
    # ax.axline((124, 117), slope=-1, color="black", linestyle="dashed")
    if is_data:
        ax.contourf(
            X,
            Y,
            signal_region,
            levels=[0, 1.54],
            colors=["black", "white"],
            extend="min",
        )
    congrol_region = R_CR(X, Y)
    ax.contour(
        X,
        Y,
        congrol_region,
        levels=[45],
        colors=["black"],
        linestyles=["dashed"],
    )
    hplt.atlas.label(loc=0, ax=ax, label=sample_name, com=13.6, lumi=luminosity)
    plot_postfix = sample_name + ("_" + region if region else "")
    fig.savefig(
        f"{output_dir}/mH_plane_{plot_postfix}.png",
        bbox_inches="tight",
    )
    plt.close()


def draw_kin_hists(
    sample_hists,
    sample_name,
    object="jet",
    luminosity=None,
    ylabel="Events",
    yscale="linear",
    ynorm_binwidth=False,
    third_exp_label="",
    output_dir=Path("plots"),
):
    kin_vars = kin_labels.keys()
    is_data = "data" in sample_name
    for kin_var in kin_vars:
        fig, ax = plt.subplots()
        hist_name = find_hist(sample_hists, lambda h: f"{object}_{kin_var}" in h)
        logger.debug(hist_name)
        hist = sample_hists[hist_name]
        hist_values = (
            hist["values"][:] * luminosity
            if luminosity and not is_data
            else hist["values"][:]
        )
        hist_edges = (
            hist["edges"][:] * inv_GeV
            if kin_var in ["pt", "mass"]
            else hist["edges"][:]
        )
        bin_width = hist_edges[1] - hist_edges[0] if ynorm_binwidth else 1.0
        # hist_bin_centers = (hist_edges[1:] + hist_edges[:-1]) / 2
        hplt.histplot(
            hist_values * 1.0 / bin_width,
            hist_edges,
            # hist_bin_centers,
            ax=ax,
            label=sample_name,
        )
        hplt.atlas.label(
            rlabel=get_com_lumi_label(luminosity) + third_exp_label,
            loc=4,
            ax=ax,
            pad=0.01,
        )
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        ax.legend()
        unit = "[GeV]" if kin_var in ["pt", "mass"] else ""
        ax.set_xlabel(f"{object} {kin_labels[kin_var]} {unit}".rstrip())
        fig.savefig(
            f"{output_dir}/{object}_{kin_var}_{sample_name}.png", bbox_inches="tight"
        )
        plt.close()
