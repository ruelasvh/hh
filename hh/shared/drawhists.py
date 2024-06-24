import re
import numpy as np
import mplhep as hplt
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from hh.shared.selection import X_HH, R_CR
from hh.shared.utils import (
    find_hist,
    find_hists,
    inv_GeV,
    kin_labels,
    get_com_lumi_label,
    jz_leading_jet_pt,
    get_legend_label,
)

np.seterr(divide="ignore", invalid="ignore")

plt.style.use(hplt.style.ATLAS)


def draw_1d_hists(
    hists_group,
    hist_prefix,
    energy,
    xlabel: str = None,
    ylabel: str = "Frequency",
    third_exp_label="",
    legend_labels: dict = None,
    luminosity=None,
    yscale="linear",
    xmin=None,
    xmax=None,
    ynorm_binwidth=False,
    xcut=None,
    density=False,
    draw_errors=False,
    ggFk01_factor=None,
    ggFk10_factor=None,
    ggFk05_factor=None,
    data2b_factor=None,
    output_dir=Path("plots"),
):
    """Draw 1D histograms in one figure. The number of histograms in the figure is
    determined by the number of samples in the hists_group dictionary. hist_prefix
    is used to select the histograms to be drawn."""

    # assert that legend_labels is either a list or "auto" with list size equal to the number of samples
    if isinstance(legend_labels, dict):
        assert len(legend_labels) == len(hists_group), (
            "legend_labels map must have the same size as the number of samples in hists_group."
            f"Expected {len(hists_group)} labels, got {len(legend_labels)}."
        )

    fig, ax = plt.subplots()
    for sample_type, sample_hists in hists_group.items():
        hist_name = find_hist(sample_hists, lambda h: re.match(hist_prefix, h))
        is_data = "data" in sample_type
        hist = sample_hists[hist_name]
        hist_values = (
            hist["values"] * luminosity
            if luminosity and not is_data
            else hist["values"]
        )
        hist_errors = hist["errors"] if draw_errors else np.zeros_like(hist_values)
        hist_errors = (
            hist_errors * luminosity if luminosity and not is_data else hist_errors
        )
        hist_edges = hist["edges"]
        hist_edges = (
            hist_edges * inv_GeV
            if xlabel is not None and "GeV" in xlabel
            else hist_edges
        )
        bin_width = hist_edges[1] - hist_edges[0] if ynorm_binwidth else 1.0
        scale_factor = 1.0
        if ggFk01_factor and "ggF" in sample_type and "k01" in sample_type:
            scale_factor = ggFk01_factor
        if ggFk05_factor and "ggF" in sample_type and "k05" in sample_type:
            scale_factor = ggFk05_factor
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
            yerr=hist_errors * bin_norm if draw_errors else None,
            label=get_legend_label(
                sample_type,
                legend_labels,
                postfix=(
                    r" ($\times$" + f"{scale_factor})" if scale_factor > 1 else None
                ),
            ),
            linewidth=2.0,
            density=density,
        )
        ax.set_ylabel(ylabel + " / %.2g" % bin_width if ynorm_binwidth else ylabel)
    ax.legend(loc="upper right")
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
    _, ymax = ax.get_ylim()
    ax.set_ylim(ymin=0.1 if yscale == "log" else 0.0, ymax=ymax * 1.5)
    if xmin is not None:
        ax.set_xlim(xmin=xmin)
    if xmax is not None:
        ax.set_xlim(xmax=xmax)
    hplt.atlas.label(
        label="Work In Progress",
        data=True,  # prevents adding Simulation label, sim labels are added in legend
        rlabel=get_com_lumi_label(luminosity, energy) + third_exp_label,
        loc=4,
        ax=ax,
        pad=0.01,
    )
    plot_name = hist_prefix.replace("$", "")
    plot_name += f"_{sample_type}" if len(hists_group) == 1 else ""
    plt.tight_layout()
    fig.savefig(f"{output_dir}/{plot_name}.png", bbox_inches="tight")
    plt.close(fig)


def draw_1d_hists_v2(
    hists_group,
    hist_prefixes,
    energy,
    xlabel=None,
    ylabel="Frequency",
    baseline_hist=None,
    legend_labels: dict = None,
    legend_options: dict = {"loc": "upper right"},
    third_exp_label="",
    luminosity=None,
    yscale="linear",
    xmin=None,
    xmax=None,
    ynorm_binwidth=False,
    density=False,
    draw_errors=False,
    draw_ratio=False,
    ymin_ratio=-0.5,
    ymax_ratio=1.5,
    ylabel_ratio="Ratio",
    scale_factors=None,
    plot_name="truth_vs_reco",
    output_dir=Path("plots"),
):
    """Draw 1D histograms in one figure. The number of histograms in the figure is
    determined by the number of samples in the hists_group dictionary. hist_prefix
    is used to select the histograms to be drawn."""

    # assert that legend_labels is either a list or "auto" with list size equal to the number of samples
    if isinstance(legend_labels, dict):
        assert len(legend_labels) == len(hist_prefixes), (
            "legend_labels map must have the same size as the number of samples in hists_group."
            f"Expected {len(hist_prefixes)} labels, got {len(legend_labels)}."
        )

    if draw_ratio:
        fig = plt.figure()
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])
        ax, ax_ratio = gs.subplots(sharex=True)
    else:
        fig, ax = plt.subplots()

    for sample_type, sample_hists in hists_group.items():
        base_hist_values = None
        for i, hist_prefix in enumerate(hist_prefixes):
            hist_name = find_hist(sample_hists, lambda h: re.match(hist_prefix, h))
            is_data = "data" in sample_type
            hist = sample_hists[hist_name]
            hist_values = (
                hist["values"] * luminosity
                if luminosity and not is_data
                else hist["values"]
            )
            hist_errors = hist["errors"] if draw_errors else np.zeros_like(hist_values)
            hist_errors = (
                hist_errors * luminosity if luminosity and not is_data else hist_errors
            )
            hist_edges = hist["edges"]
            hist_edges = (
                hist_edges * inv_GeV
                if xlabel is not None and "GeV" in xlabel
                else hist_edges
            )
            bin_width = hist_edges[1] - hist_edges[0] if ynorm_binwidth else 1.0
            scale_factor = 1.0
            if scale_factors:
                scale_factor = scale_factors[i]
            hist_values = hist_values * scale_factor
            bin_norm = 1.0 / hist_values.sum() if density else bin_width
            hist_values = hist_values * bin_norm
            label = get_legend_label(
                hist_prefix,
                legend_labels,
                postfix=(
                    r" ($\times$" + f"{scale_factor})" if scale_factor > 1 else None
                ),
            )
            hplt.histplot(
                hist_values,
                hist_edges - bin_width * 0.5,
                yerr=hist_errors * bin_norm if draw_errors else None,
                label=label,
                linewidth=2.0,
                density=density,
                color=f"C{i}",
                ax=ax,
            )
            ax.set_ylabel(ylabel + " / %.2g" % bin_width if ynorm_binwidth else ylabel)
            if draw_ratio:
                if i == 0:
                    base_hist_values = hist_values
                else:
                    hplt.histplot(
                        hist_values / base_hist_values,
                        hist_edges - bin_width * 0.5,
                        color=f"C{i}",
                        ax=ax_ratio,
                    )
                    ax_ratio.set_ylabel(ylabel_ratio, loc="center")
                    ax_ratio.set_ylim(ymin_ratio, ymax_ratio)
                    ax_ratio.axhline(1, color="black", linestyle="--", linewidth=0.4)
                    # # Get the list of ytick labels
                    # labels = ax_ratio.get_yticklabels()
                    # # Hide the first and last label
                    # labels[0].set_visible(False)
                    # labels[-1].set_visible(False)
        if baseline_hist is not None:
            hplt.histplot(
                baseline_hist,
                hist_edges - bin_width * 0.5,
                yerr=hist_errors * bin_norm if draw_errors else None,
                label=r"Flat $m\mathrm{HH}$ distribution",
                linewidth=2.0,
                density=density,
                color=f"C{i+1}",
                ax=ax,
            )
    ax.legend(**legend_options)
    if xlabel:
        ax.set_xlabel(xlabel) if not draw_ratio else ax_ratio.set_xlabel(xlabel)
    ax.set_yscale(yscale)
    _, ymax = ax.get_ylim()
    ax.set_ylim(ymin=0.1 if yscale == "log" else 0.0, ymax=ymax * 1.5)
    if xmin is not None:
        ax.set_xlim(xmin=xmin)
    if xmax is not None:
        ax.set_xlim(xmax=xmax)
    hplt.atlas.label(
        label="Work In Progress",
        data=True,  # prevents adding Simulation label, sim labels are added in legend
        rlabel=get_com_lumi_label(luminosity, energy) + third_exp_label,
        loc=4,
        ax=ax,
        pad=0.01,
    )
    plot_name += f"_{sample_type}" if len(hists_group) == 1 else ""
    plt.tight_layout()
    fig.savefig(f"{output_dir}/{plot_name}.png", bbox_inches="tight")
    plt.close(fig)


def draw_efficiency(
    hists_group,
    hist_prefixes,
    energy,
    xlabel=None,
    ylabel="Efficiency",
    legend_labels: dict = None,
    legend_options: dict = {"loc": "upper right"},
    third_exp_label="",
    luminosity=None,
    xmin=None,
    xmax=None,
    ymin=0,
    ymax=1,
    draw_errors=False,
    output_dir=Path("plots"),
    plot_name="efficiency",
):
    """Draw 1D histograms in one figure. The number of histograms in the figure is
    determined by the number of samples in the hists_group dictionary. hist_prefix
    is used to select the histograms to be drawn."""

    # assert that legend_labels is either a list or "auto" with list size equal to the number of samples
    if isinstance(legend_labels, dict):
        assert len(legend_labels) == len(hist_prefixes), (
            "legend_labels map must have the same size as the number of samples in hists_group."
            f"Expected {len(hist_prefixes)} labels, got {len(legend_labels)}."
        )

    fig, ax = plt.subplots()

    for sample_type, sample_hists in hists_group.items():
        for i, hists in enumerate(hist_prefixes):
            is_data = "data" in sample_type
            hist_total_name = find_hist(sample_hists, lambda h: re.match(hists[0], h))
            hist_total = sample_hists[hist_total_name]
            hist_total_values = hist_total["values"]
            hist_total_values = (
                hist_total_values * luminosity
                if luminosity and not is_data
                else hist_total_values
            )
            hist_total_edges = hist_total["edges"]
            hist_total_edges = (
                hist_total_edges * inv_GeV
                if xlabel is not None and "GeV" in xlabel
                else hist_total_edges
            )
            label = get_legend_label(hist_total_name, legend_labels)
            hist_pass_name = find_hist(sample_hists, lambda h: re.match(hists[1], h))
            hist_pass = sample_hists[hist_pass_name]
            hist_pass_values = hist_pass["values"]
            hist_pass_values = (
                hist_pass_values * luminosity
                if luminosity and not is_data
                else hist_pass_values
            )
            eff = hist_pass_values / hist_total_values
            hplt.histplot(
                eff,
                hist_total_edges,
                label=label,
                color=f"C{i}",
                ax=ax,
            )

    ax.legend(**legend_options)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin=ymin, ymax=ymax * 1.5)
    if xmin is not None:
        ax.set_xlim(xmin=xmin)
    if xmax is not None:
        ax.set_xlim(xmax=xmax)
    hplt.atlas.label(
        label="Work In Progress",
        data=True,  # prevents adding Simulation label, sim labels are added in legend
        rlabel=get_com_lumi_label(luminosity, energy) + third_exp_label,
        loc=4,
        ax=ax,
        pad=0.01,
    )
    plot_name += f"_{sample_type}" if len(hists_group) == 1 else ""
    plt.tight_layout()
    fig.savefig(f"{output_dir}/{plot_name}.png", bbox_inches="tight")
    plt.close(fig)


def draw_mH_1D_hists_v2(
    hists_group,
    hist_prefix,
    energy,
    luminosity=None,
    xlims=None,
    ylims=None,
    third_exp_label="",
    ylabel="Frequency",
    ynorm_binwidth=False,
    yscale="linear",
    ggFk01_factor=None,
    ggFk10_factor=None,
    output_dir=Path("plots"),
):
    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(8, 10))
    axs = axs.flat
    for sample_type, sample_hists in hists_group.items():
        is_data = "data" in sample_type
        hists = find_hists(sample_hists, lambda h: re.match(hist_prefix, h))
        for i in range(len(hists)):
            if "signal" in hists[i] and is_data:
                continue
            ax = axs[i]
            hist = sample_hists[hists[i]]
            hist_values = (
                hist["values"] * luminosity
                if luminosity and not is_data
                else hist["values"]
            )
            hist_edges = hist["edges"]
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
                rlabel=get_com_lumi_label(luminosity, energy) + third_exp_label,
                loc=4,
                ax=ax,
                pad=0.01,
            )
    plot_name = hist_prefix.replace("$", "")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/{plot_name}.png", bbox_inches="tight")
    plt.close(fig)


def draw_mH_1D_hists(
    sample_hists,
    sample_name,
    hist_prefix,
    energy,
    luminosity=None,
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
        hist = sample_hists[hists[i]]
        hist_values = (
            hist["values"] * luminosity
            if luminosity and not is_data
            else hist["values"]
        )
        hplt.histplot(
            hist_values,
            hist["edges"] * inv_GeV,
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
            rlabel=get_com_lumi_label(luminosity, energy) + third_exp_label,
            loc=4,
            ax=ax,
        )
    plot_name = f"{sample_name}_{hist_prefix.replace('$', '')}"
    plt.tight_layout()
    fig.savefig(f"{output_dir}/mH_{plot_name}.png", bbox_inches="tight")
    plt.close(fig)


def draw_mH_plane_2D_hists(
    sample_hists,
    sample_name,
    hist_prefix,
    energy,
    luminosity=None,
    log_z=False,
    label_z="Events",
    output_dir=Path("plots"),
):
    fig, ax = plt.subplots()
    hist_name = find_hist(sample_hists, lambda h: re.match(hist_prefix, h))
    hist = sample_hists[hist_name]
    is_data = "data" in sample_name
    bins_GeV = hist["edges"] * inv_GeV
    hist_values = (
        hist["values"] * luminosity if luminosity and not is_data else hist["values"]
    )
    # remove outliers from hist_values
    hist_values[hist_values > 2000] = 0
    pcm, cbar, _ = hplt.hist2dplot(
        hist_values,
        bins_GeV,
        bins_GeV,
        ax=ax,
        cbarpad=0.20,
        cbarsize="5%",
        flow=None,
        cbarextend=True,
        # cmap="PuBu_r",
        cmap="RdBu_r",
    )
    cbar.set_label(label_z)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: str(x)))
    if log_z:
        pcm.set_norm(
            colors.SymLogNorm(
                linthresh=0.015,
                # linthresh=1,
                # linscale=1,
                vmin=hist_values.min(),
                vmax=hist_values.max(),
                base=10,
            )
        )
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1E"))
        cbar.set_label(label_z + " (Symmetric Log Scale)")
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
        colors=["tab:red", "black"],
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
    hplt.atlas.label(
        loc=0, ax=ax, label=sample_name.replace("_", " "), com=energy, lumi=luminosity
    )
    plot_name = f"{sample_name}_{hist_name}"
    plt.tight_layout()
    fig.savefig(f"{output_dir}/{plot_name}.png", bbox_inches="tight")
    plt.close(fig)


def draw_kin_hists(
    sample_hists,
    sample_name,
    energy,
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
        hist_name = find_hist(
            sample_hists, lambda h: f"{object.lower()}_{kin_var}" in h
        )
        hist = sample_hists[hist_name]
        hist_values = (
            hist["values"] * luminosity
            if luminosity and not is_data
            else hist["values"]
        )
        hist_edges = (
            hist["edges"] * inv_GeV if kin_var in ["pt", "mass"] else hist["edges"]
        )
        bin_width = hist_edges[1] - hist_edges[0] if ynorm_binwidth else 1.0
        hplt.histplot(
            hist_values * 1.0 / bin_width,
            hist_edges,
            ax=ax,
            label=sample_name.replace("_", " "),
        )
        hplt.atlas.label(
            rlabel=get_com_lumi_label(luminosity, energy) + third_exp_label,
            loc=4,
            ax=ax,
            pad=0.01,
        )
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        ax.legend()
        unit = "[GeV]" if kin_var in ["pt", "mass"] else ""
        ax.set_xlabel(f"{object} {kin_labels[kin_var]} {unit}".rstrip())
        plot_name = f"{sample_name}_{hist_name}"
        plt.tight_layout()
        fig.savefig(f"{output_dir}/{plot_name}.png", bbox_inches="tight")
        plt.close(fig)


def num_events_vs_sample(
    hists_group,
    hist_prefix,
    energy,
    xlabel="Samples",
    ylabel="Frequency",
    third_exp_label="",
    luminosity=None,
    density=False,
    output_dir=Path("plots"),
):
    fig, ax = plt.subplots()
    counts = []
    for sample_type, sample_hists in hists_group.items():
        hist_name = find_hist(sample_hists, lambda h: re.match(hist_prefix, h))
        count = np.sum(sample_hists[hist_name]["values"])
        counts.append(count)

    for sample_type, count in zip(hists_group.keys(), counts):
        jz = re.search(r"JZ[0-9]", sample_type)
        if jz:
            jz = jz.group()
            jz = (
                "Leading jet $p_{\mathrm{T}}$ "
                + f"{jz_leading_jet_pt['min'][int(jz[-1])]}-{jz_leading_jet_pt['max'][int(jz[-1])]} GeV"
            )
        normalized_count = count / sum(counts)
        ax.bar(
            sample_type,
            normalized_count if density else count,
            label=f"{jz if jz else sample_type}: {int(count * luminosity if luminosity else count)}",
        )

    ax.legend(loc="upper right")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    hplt.atlas.label(label=third_exp_label, loc=0, ax=ax, com=energy, lumi=luminosity)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin=ymin, ymax=ymax * len(hists_group) * 0.3)
    plt.tight_layout()
    fig.savefig(
        f"{output_dir}/{ylabel.lower().replace(' ', '_')}_{xlabel.lower().replace(' ', '_')}.png",
        bbox_inches="tight",
    )
    plt.close(fig)
