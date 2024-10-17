import re
import numpy as np
import mplhep as hplt
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from hh.shared.selection import X_HH, R_CR
from hh.shared.utils import (
    find_hist,
    find_hists,
    GeV,
    kin_labels,
    get_com_lumi_label,
    jz_leading_jet_pt,
    get_legend_label,
    register_bottom_offset,
)
from hh.shared.error import get_efficiency_with_uncertainties

np.seterr(divide="ignore", invalid="ignore")

plt.style.use(hplt.style.ATLAS)


class HistPlottable:
    def __init__(
        self,
        counts,
        bins,
        errors=None,
        scale_factor=None,
        normalize=False,
        binwidth_norm=True,
        ylabel="Entries",
        xlabel=None,
        legend_label=None,
    ):

        self.counts = counts
        self.counts = self.counts * scale_factor if scale_factor else self.counts
        self.counts = self.counts
        self.bins = bins
        self.errors = errors
        if self.errors is not None:
            self.errors = self.errors * scale_factor if scale_factor else self.errors
            self.errors = self.errors
        self.scale_factor = scale_factor
        self.normalize = normalize
        self.binwidth_norm = binwidth_norm
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.legend_label = legend_label

    def plot(self, ax, **kwargs):
        self.bins = (
            self.bins * GeV
            if self.xlabel is not None and "GeV" in self.xlabel
            else self.bins
        )
        if self.normalize:
            self.counts = self.counts / np.sum(self.counts)
            self.binwidth_norm = None
            ax.set_ylabel(f"{self.ylabel} / N")
        elif self.binwidth_norm:
            bin_widths = np.diff(self.bins)
            # check that they are the same widths, else throw an error and say bins need to be uniform
            assert np.allclose(bin_widths, bin_widths[0]), (
                "Bins need to be uniform to use bin width normalization. "
                f"Got bin widths: {bin_widths}"
            )
            self.binwidth_norm = bin_widths[0]
            ax.set_ylabel(
                rf"{self.ylabel} / {self.binwidth_norm:.2g} "
                + "".join(
                    [
                        unit
                        for unit in ["MeV", "GeV", "%"]
                        if self.xlabel is not None and unit in self.xlabel
                    ]
                )
            )
        if self.legend_label:
            self.legend_label = get_legend_label(
                self.legend_label,
                None,
                postfix=(
                    rf" ($\times{self.scale_factor}$)"
                    if self.scale_factor is not None
                    else None
                ),
            )
        hplt.histplot(
            self.counts,
            self.bins,
            yerr=self.errors if self.errors is not None else None,
            label=self.legend_label,
            ax=ax,
            **kwargs,
        )


def draw_1d_hists(
    hists_group,
    hist_prefix,
    energy,
    xlabel: str = None,
    ylabel: str = "Entries",
    third_exp_label="",
    legend_labels: dict = None,
    luminosity=None,
    yscale="linear",
    xmin=None,
    xmax=None,
    normalize=False,
    binwidth_norm=True,
    xcut=None,
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
        hist_values = hist["values"]
        hist_edges = hist["edges"]
        hist_errors = hist["errors"] if draw_errors else None
        scale_factor = None
        if ggFk01_factor and "ggF" in sample_type and "k01" in sample_type:
            scale_factor = ggFk01_factor
        if ggFk05_factor and "ggF" in sample_type and "k05" in sample_type:
            scale_factor = ggFk05_factor
        if ggFk10_factor and "ggF" in sample_type and "k10" in sample_type:
            scale_factor = ggFk10_factor
        if data2b_factor and "data" in sample_type and "2b" in sample_type:
            scale_factor = data2b_factor
        bin_width = hist_edges[1] - hist_edges[0]
        bin_centers = hist_edges + (bin_width * 0.5)
        hist_main = HistPlottable(
            hist_values,
            bin_centers,
            errors=hist_errors,
            scale_factor=scale_factor,
            normalize=normalize,
            binwidth_norm=binwidth_norm,
            ylabel=ylabel,
            xlabel=xlabel,
            legend_label=legend_labels[sample_type] if legend_labels else None,
        )
        hist_main.plot(ax=ax, linewidth=2.0)
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
        rlabel=get_com_lumi_label(energy, luminosity) + third_exp_label,
        loc=4,
        ax=ax,
        pad=0.01,
    )
    plot_name = hist_prefix.replace("$", "")
    plot_name += f"_{sample_type}" if len(hists_group) == 1 else ""
    plt.tight_layout()
    fig.savefig(f"{output_dir}/{plot_name}.png", bbox_inches="tight")
    plt.close(fig)


def draw_signal_vs_background(
    var_name,
    signal,
    background,
    xlabel,
    luminosity,
    energy,
    ylabel: str = "Entries",
    legend_labels: dict = None,
    legend_options: dict = {"loc": "upper right"},
    third_exp_label: str = "",
    show_counts: bool = False,
    plot_name: str = "signal_vs_background",
    output_dir: Path = Path("plots"),
    **kwargs,
):
    sig = {}
    for sig_hists in signal.values():
        hist_name = find_hist(sig_hists, lambda h: var_name == h)
        sig = sig_hists[hist_name]
    bkg = {}
    for bkg_hists in background.values():
        hist_name = find_hist(bkg_hists, lambda h: var_name == h)
        hist = bkg_hists[hist_name]
        if "edges" not in bkg:
            bkg["edges"] = hist["edges"]
        if "values" not in bkg:
            bkg["values"] = hist["values"]
        else:
            bkg["values"] += hist["values"]
        if "errors" not in bkg:
            bkg["errors"] = hist["errors"]
        else:
            bkg["errors"] = np.sqrt(bkg["errors"] ** 2 + hist["errors"] ** 2)
    bkg = {"Background MC": bkg}
    sig = {k: sig for k in signal}
    samples = {**sig, **bkg}
    fig, ax = plt.subplots()
    for sample_name in samples:
        hist = samples[sample_name]
        if hist:
            hist_main = HistPlottable(
                hist["values"],
                hist["edges"],
                errors=hist["errors"],
                legend_label=legend_labels.get(sample_name, sample_name),
                scale_factor=100 if "ggF" in sample_name else None,
                binwidth_norm=False,
                xlabel=xlabel,
            )
            hist_main.plot(ax=ax, linewidth=2.0)
    hplt.atlas.label(
        label="Work In Progress",
        data=True,  # prevents adding Simulation label, sim labels are added in legend
        rlabel=get_com_lumi_label(energy, luminosity) + third_exp_label,
        loc=4,
        ax=ax,
        pad=0.01,
    )
    if show_counts:
        countsstr = "\n".join(
            [
                f"{legend_labels.get(s, s)}: {(np.sum(h['values'])):.2f}"
                for s, h in samples.items()
                if h
            ]
        )
        ax.text(
            0.5,
            0.5,
            countsstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin=xmin, xmax=1100)
    ax.set_ylim(ymin=0, ymax=ax.get_ylim()[1] * 1.5)
    ax.legend(**legend_options)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/{plot_name}.png", bbox_inches="tight")
    plt.close(fig)


def draw_1d_hists_v2(
    hists_group,
    hist_prefixes,
    energy,
    xlabel=None,
    ylabel="Entries",
    baseline=None,
    legend_labels: dict = None,
    legend_options: dict = {"loc": "upper right"},
    third_exp_label="",
    luminosity=None,
    yscale="linear",
    xmin=None,
    xmax=None,
    binwidth_norm=True,
    normalize=False,
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
            hist_values = hist["values"]
            hist_edges = hist["edges"]
            hist_errors = hist["errors"] if draw_errors else None
            hist_main = HistPlottable(
                hist_values,
                hist_edges,
                errors=hist_errors,
                scale_factor=scale_factors[i] if scale_factors else None,
                normalize=normalize,
                binwidth_norm=binwidth_norm,
                legend_label=legend_labels[hist_prefix] if legend_labels else None,
                xlabel=xlabel,
                ylabel=ylabel,
            )
            hist_main.plot(ax=ax, linewidth=2.0, color=f"C{i}")
            if draw_ratio:
                if i == 0:
                    base_hist_values = hist_values
                else:
                    hist_ratio = HistPlottable(
                        hist_values / base_hist_values,
                        hist_edges,
                        xlabel=xlabel,
                        ylabel=ylabel,
                    )
                    hist_ratio.plot(ax=ax_ratio, color=f"C{i}")
        if baseline is not None:
            infvar = np.array([np.inf])
            edges_underflow_overflow = np.concatenate([-infvar, hist_edges, infvar])
            baseline_hist, _ = np.histogramdd(baseline, bins=[edges_underflow_overflow])
            hist_baseline = HistPlottable(
                baseline_hist,
                hist_edges,
                errors=hist_errors,
                legend_label=r"Flat $m_\mathrm{HH}$ distribution",
                normalize=normalize,
                binwidth_norm=binwidth_norm,
                xlabel=xlabel,
                ylabel=ylabel,
            )
            hist_baseline.plot(ax=ax, linewidth=2.0, color=f"C{i+1}")
    ax.legend(**legend_options)
    ax.set_yscale(yscale)
    _, ymax = ax.get_ylim()
    ax.set_ylim(ymin=0.1 if yscale == "log" else 0.0, ymax=ymax * 1.5)
    if xmin is not None:
        ax.set_xlim(xmin=xmin)
    if xmax is not None:
        ax.set_xlim(xmax=xmax)
    if draw_ratio:
        ax_ratio.set_ylabel(ylabel_ratio, loc="center")
        ax_ratio.set_ylim(ymin_ratio, ymax_ratio)
        ax_ratio.axhline(1, color="black", linestyle="--", linewidth=0.4)
        ax_ratio.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)
    hplt.atlas.label(
        label="Work In Progress",
        data=True,  # prevents adding Simulation label, sim labels are added in legend
        rlabel=get_com_lumi_label(energy, luminosity) + third_exp_label,
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
    draw_errors=True,
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
            hist_total_name = find_hist(sample_hists, lambda h: re.match(hists[0], h))
            hist_total = sample_hists[hist_total_name]
            hist_pass_name = find_hist(sample_hists, lambda h: re.match(hists[1], h))
            hist_pass = sample_hists[hist_pass_name]
            eff, errors = get_efficiency_with_uncertainties(
                hist_pass["values"][1:-1], hist_total["values"][1:-1]
            )
            breakpoint()
            hist_main = HistPlottable(
                eff,
                hist_total["edges"],
                errors=errors if draw_errors else None,
                legend_label=legend_labels[hist_total_name] if legend_labels else None,
                xlabel=xlabel,
                ylabel=ylabel,
            )
            error_attrs = (
                {"histtype": "errorbar", "solid_capstyle": "projecting", "capsize": 3}
                if draw_errors
                else {}
            )
            hist_main.plot(
                ax=ax,
                color=f"C{i}",
                **error_attrs,
            )

    ax.legend(**legend_options)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # scale y-axis so that the legend does not overlap with the plot
    ax.set_ylim(ymin=ymin, ymax=ymax * (1 + 0.1 * len(hist_prefixes) + 0.1))
    if xmin is not None:
        ax.set_xlim(xmin=xmin)
    if xmax is not None:
        ax.set_xlim(xmax=xmax)
    hplt.atlas.label(
        label="Work In Progress",
        data=True,  # prevents adding Simulation label, sim labels are added in legend
        rlabel=get_com_lumi_label(energy, luminosity) + third_exp_label,
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
    ylabel="Entries",
    density=False,
    binwidth_norm=False,
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
            bin_width = hist_edges[1] - hist_edges[0]
            scale_factor = 1
            if ggFk01_factor and "ggF" in sample_type and "k01" in sample_type:
                scale_factor = ggFk01_factor
            if ggFk10_factor and "ggF" in sample_type and "k10" in sample_type:
                scale_factor = ggFk10_factor
            hist_values = hist_values * scale_factor
            if density and not binwidth_norm:
                hist_values = hist_values / np.sum(hist_values)
            hplt.histplot(
                hist_values[1:-1],
                hist_edges * GeV,
                histtype="step",
                stack=False,
                ax=ax,
                label=sample_type,
                binwnorm=binwidth_norm,
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
            ax.set_ylabel(ylabel + " / %.2g" % bin_width if binwidth_norm else ylabel)
            hplt.atlas.label(
                rlabel=get_com_lumi_label(energy, luminosity) + third_exp_label,
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
            hist["edges"] * GeV,
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
        ax.set_ylabel("Entries")
        hplt.atlas.label(
            rlabel=get_com_lumi_label(energy, luminosity) + third_exp_label,
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
    label_z="Entries",
    xrange=(50, 200),
    yrange=(50, 200),
    third_exp_label="",
    output_dir=Path("plots"),
):
    fig, ax = plt.subplots()
    hist_name = find_hist(sample_hists, lambda h: re.match(hist_prefix, h))
    hist = sample_hists[hist_name]
    is_data = "data" in sample_name
    bins_GeV = hist["edges"] * GeV
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
        cbarpad=0.15,
        cbarsize="5%",
        flow=None,
        cbarextend=True,
        cmap="RdBu_r",
    )
    cbar.set_label(label_z)
    # move top bar "multiplier" so doesn't overlap with the plot labels
    register_bottom_offset(cbar.ax.yaxis)
    if log_z:
        cbar.ax.set_yscale(
            "symlog",
            base=10,
            linthresh=100 if hist_values.max() > 100 else 0.015,
            linscale=0.05 if hist_values.max() > 100 else 1,
            subs=[2, 3, 4, 5, 6, 7, 8, 9],
        )
    ax.set_ylabel(r"$m_{H2}$ [GeV]")
    ax.set_xlabel(r"$m_{H1}$ [GeV]")
    ax.set_ylim(*yrange)
    ax.set_xlim(*xrange)
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
    plot_label = sample_name.replace("_", " ")
    plot_label = f"{plot_label}, {third_exp_label}" if third_exp_label else plot_label
    hplt.atlas.label(
        loc=0,
        ax=ax,
        label=plot_label,
        com=energy,
        lumi=luminosity,
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
    ylabel="Entries",
    yscale="linear",
    density=False,
    binwidth_norm=False,
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
        hist_edges = hist["edges"] * GeV if kin_var in ["pt", "mass"] else hist["edges"]
        if density and not binwidth_norm:
            hist_values = hist_values / np.sum(hist_values)
        hplt.histplot(
            hist_values[1:-1],
            hist_edges,
            ax=ax,
            label=sample_name.replace("_", " "),
            binwnorm=binwidth_norm,
        )
        hplt.atlas.label(
            rlabel=get_com_lumi_label(energy, luminosity) + third_exp_label,
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
    ylabel="Entries",
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
