import numpy as np
import matplotlib.pyplot as plt
import mplhep as hplt
import re
from pathlib import Path
from shared.utils import (
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


def draw_hists(
    hists_group: dict,
    luminosity: float,
    btag: str,
    plots_postfix: str,
    output_dir: Path,
) -> None:
    """Draw all the histrograms"""

    # check if output_dir exists, if not create it
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    draw_1d_hists(
        hists_group={
            key: value
            for key, value in hists_group.items()
            if "ggF" in key or "data" in key
        },
        hist_prefix="hh_deltaeta",
        xlabel="$\Delta\eta_{HH}$",
        ylabel="Events",
        luminosity=luminosity,
        ynorm_binwidth=True,
        xcut=1.5,
        third_exp_label="\n" + btag + ", no $\mathrm{X}_{\mathrm{Wt}}$ cut",
        ggFk01_factor=500,
        ggFk10_factor=50,
        output_dir=output_dir,
    )
    draw_1d_hists(
        hists_group={
            key: value
            for key, value in hists_group.items()
            if "ggF_k01" in key or "ttbar" in key
        },
        hist_prefix="top_veto",
        xlabel="$\mathrm{X}_{\mathrm{Wt}}$",
        ylabel="Events",
        third_exp_label=f"\n{btag}",
        ynorm_binwidth=True,
        luminosity=luminosity,
        ggFk01_factor=50,
        xcut=1.5,
        output_dir=output_dir,
    )
    draw_1d_hists(
        hists_group={
            key: value
            for key, value in hists_group.items()
            if "ggF_k01" in key or "ttbar" in key
        },
        hist_prefix="top_veto_n_btags",
        xlabel="$\mathrm{X}_{\mathrm{Wt}}$ btags",
        ylabel="Events",
        third_exp_label=f"\n{btag}",
        luminosity=luminosity,
        output_dir=output_dir,
    )
    draw_1d_hists(
        hists_group,
        hist_prefix="hh_mass_discrim",
        xlabel="$\mathrm{X}_{\mathrm{HH}}$",
        ylabel="Events",
        ynorm_binwidth=True,
        luminosity=luminosity,
        third_exp_label=f"\n{btag}",
        xcut=1.6,
        output_dir=output_dir,
    )
    draw_mH_1D_hists_v2(
        hists_group,
        hist_prefix="mH[12]_baseline_signal_region$",
        region="SR",
        luminosity=luminosity,
        ylabel="Events",
        xlims=(60, 200),
        third_exp_label=f"\n{btag} Signal Region",
        output_dir=output_dir,
    )
    draw_mH_1D_hists_v2(
        hists_group,
        hist_prefix="mH[12]_baseline_control_region$",
        region="CR",
        luminosity=luminosity,
        ylabel="Events",
        xlims=(60, 200),
        third_exp_label=f"\n{btag} Control Region",
        output_dir=output_dir,
    )
    for sample_type, sample_hists in hists_group.items():
        draw_jet_kin_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            luminosity=luminosity,
            yscale="log",
            output_dir=output_dir,
        )
        draw_mH_1D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            hist_prefix="mH[12]_baseline$",
            xlim=(90, 150),
            output_dir=output_dir,
        )
        draw_mH_plane_2D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            hist_prefix="mH_plane_baseline$",
            output_dir=output_dir,
        )
        draw_mH_1D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            region="SR",
            hist_prefix="mH[12]_baseline_signal_region$",
            output_dir=output_dir,
        )
        draw_mH_plane_2D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            region="SR",
            hist_prefix="mH_plane_baseline_signal_region$",
            output_dir=output_dir,
        )
        draw_mH_1D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            region="CR",
            hist_prefix="mH[12]_baseline_control_region$",
            output_dir=output_dir,
        )
        draw_mH_plane_2D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            region="CR",
            hist_prefix="mH_plane_baseline_control_region$",
            output_dir=output_dir,
        )


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
    ggFk01_factor=None,
    ggFk10_factor=None,
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
        hist_values = hist["values"][:]
        hist_edges = hist["edges"][:]
        bin_width = hist_edges[1] - hist_edges[0] if ynorm_binwidth else 1.0
        scale_factor = 1
        if ggFk01_factor and "ggF_k01" in sample_type:
            scale_factor = ggFk01_factor
        if ggFk10_factor and "ggF_k10" in sample_type:
            scale_factor = ggFk10_factor
        hplt.histplot(
            hist_values
            * bin_width
            * scale_factor
            * (luminosity if luminosity and not is_data else 1.0),
            hist_edges,
            ax=ax,
            label=(str(scale_factor) + r"$\times$" if scale_factor != 1 else "")
            + sample_type,
            linewidth=2.0,
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
        rlabel=get_com_lumi_label(luminosity) + third_exp_label,
        loc=4,
        ax=ax,
        pad=0.01,
    )
    fig.savefig(f"{output_dir}/{hist_prefix}.png", bbox_inches="tight")
    plt.close()


def draw_mH_1D_hists_v2(
    hists_group,
    hist_prefix,
    luminosity=None,
    xlims=None,
    third_exp_label="",
    ylabel="Frequency",
    ynorm_binwidth=False,
    region=None,
    multijet_factor=None,
    ttbar_factor=None,
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
            hist_values = hist["values"][:]
            hist_edges = hist["edges"][:]
            bin_width = hist_edges[1] - hist_edges[0] if ynorm_binwidth else 1.0
            scale_factor = 1
            if ggFk01_factor and "ggF_k01" in sample_type:
                scale_factor = ggFk01_factor
            if ggFk10_factor and "ggF_k10" in sample_type:
                scale_factor = ggFk10_factor
            hplt.histplot(
                hist_values
                * bin_width
                * scale_factor
                * (luminosity if luminosity and not is_data else 1.0),
                hist_edges * inv_GeV,
                # histtype="step"
                # if "ggF_k01" in sample_type or "ggF_k10" in sample_type
                # else "fill",
                # stack=False
                # if "ggF_k01" in sample_type or "ggF_k10" in sample_type
                # else True,
                histtype="step",
                stack=False,
                ax=ax,
                label=sample_type,
            )
            if xlims:
                ax.set_xlim(*xlims)
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.1)
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
    xlim=None,
    region=None,
    output_dir=Path("plots"),
):
    fig, axs = plt.subplots(2, constrained_layout=True)
    axs = axs.flat
    hists = find_hists(sample_hists, lambda h: re.match(hist_prefix, h))
    for i in range(len(hists)):
        ax = axs[i]
        logger.debug(hists[i])
        hist = sample_hists[hists[i]]
        hplt.histplot(
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
        hplt.atlas.label(loc=1, ax=ax, rlabel="")
    plot_postfix = sample_name + ("_" + region if region else "")
    fig.savefig(f"{output_dir}/mH_{plot_postfix}.png", bbox_inches="tight")
    plt.close()


def draw_mH_plane_2D_hists(
    sample_hists,
    sample_name,
    hist_prefix,
    region=None,
    output_dir=Path("plots"),
):
    fig, ax = plt.subplots()
    hist_name = find_hist(sample_hists, lambda h: re.match(hist_prefix, h))
    hist = sample_hists[hist_name]
    binsGeV = hist["edges"][:] * inv_GeV
    print(sample_name)
    print(hist_name)
    print(np.sum(hist["values"][:]))
    hplt.hist2dplot(
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
    hplt.atlas.label(loc=0, ax=ax, label=sample_name, com=13.6)
    plot_postfix = sample_name + ("_" + region if region else "")
    fig.savefig(
        f"{output_dir}/mH_plane_{plot_postfix}.png",
        bbox_inches="tight",
    )
    plt.close()


def draw_jet_kin_hists(
    sample_hists,
    sample_name,
    luminosity=None,
    yscale="linear",
    third_exp_label="",
    output_dir=Path("plots"),
):
    jet_vars = ["pt"]
    is_data = "data" in sample_name
    for jet_var in jet_vars:
        fig, ax = plt.subplots()
        hist_name = find_hist(sample_hists, lambda h: f"jet_{jet_var}" in h)
        logger.debug(hist_name)
        hist = sample_hists[hist_name]
        hplt.histplot(
            hist["values"][:] * (luminosity if luminosity and not is_data else 1.0),
            hist["edges"][:] * inv_GeV,
            ax=ax,
            label=sample_name,
        )
        hplt.atlas.label(
            rlabel=get_com_lumi_label(luminosity) + third_exp_label,
            loc=4,
            ax=ax,
            pad=0.01,
        )
        ax.set_yscale("log")
        ax.set_ylabel("Frequency")
        ax.set_yscale(yscale)
        ax.legend()
        ax.set_xlabel(f"jet {kin_labels[jet_var]} [GeV]")
        fig.savefig(
            f"{output_dir}/jet_{jet_var}_{sample_name}.png", bbox_inches="tight"
        )
        plt.close()
