import re
import numpy as np
import mplhep as hplt
from pathlib import Path
import matplotlib.pyplot as plt
from hh.shared.selection import X_HH, R_CR
from hh.shared.utils import (
    find_hist,
    find_hists,
    GeV,
    get_com_lumi_label,
    jz_leading_jet_pt,
    get_legend_label,
    register_bottom_offset,
)
from hh.shared.labels import kin_labels

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


def draw_dijet_slices_hists(
    hists_group,
    hist_name,
    energy,
    xlabel: str = None,
    ylabel: str = "Entries",
    third_exp_label="",
    legend_labels: list = None,
    luminosity=None,
    yscale="log",
    normalize=False,
    binwidth_norm=True,
    output_dir=Path("plots"),
):
    """Draw dijet slices histograms in one figure."""

    # assert that legend_labels is either a list or "auto" with list size equal to the number of samples
    if isinstance(legend_labels, dict):
        assert len(legend_labels) == len(hists_group), (
            "legend_labels map must have the same size as the number of samples in hists_group."
            f"Expected {len(hists_group)} labels, got {len(legend_labels)}."
        )

    fig, ax = plt.subplots()
    sorted_keys = sorted(hists_group.keys())
    hists = []
    labels = []
    bins = []
    for sample_type in sorted_keys:
        labels.append(sample_type)
        hist = hists_group[sample_type][hist_name]
        hist_values = hist["values"][1:-1]
        hists.append(hist_values)
        bins = hist["edges"]
    hist_main = HistPlottable(
        hists,
        bins,
        normalize=normalize,
        binwidth_norm=binwidth_norm,
        ylabel=ylabel,
        xlabel=xlabel,
        legend_label=legend_labels if legend_labels else labels,
    )
    hist_main.plot(ax=ax, histtype="fill", stack=True)
    ax.legend(loc="upper right")
    ax.set_yscale(yscale)
    # _, ymax = ax.get_ylim()
    # ax.set_ylim(ymin=0.1 if yscale == "log" else 0.0, ymax=ymax * 1.5)
    if xlabel:
        ax.set_xlabel(xlabel)
    hplt.atlas.label(
        label="Work In Progress",
        data=True,  # prevents adding Simulation label, sim labels are added in legend
        rlabel=get_com_lumi_label(energy, luminosity) + third_exp_label,
        loc=4,
        ax=ax,
        pad=0.01,
    )
    plot_name = f"multijet_slices_{hist_name}"
    plt.tight_layout()
    fig.savefig(f"{output_dir}/{plot_name}.png", bbox_inches="tight")
    plt.close(fig)


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
    styles={},
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
        hist = find_hist(sample_hists, lambda h_name: hist_prefix == h_name)
        hist_values = hist["values"][1:-1]
        hist_edges = hist["edges"]
        hist_errors = hist["errors"][1:-1] if draw_errors else None
        scale_factor = None
        if ggFk01_factor and "ggF" in sample_type and "k01" in sample_type:
            scale_factor = ggFk01_factor
        if ggFk05_factor and "ggF" in sample_type and "k05" in sample_type:
            scale_factor = ggFk05_factor
        if ggFk10_factor and "ggF" in sample_type and "k10" in sample_type:
            scale_factor = ggFk10_factor
        if data2b_factor and "data" in sample_type and "2b" in sample_type:
            scale_factor = data2b_factor
        # bin_width = hist_edges[1] - hist_edges[0]
        # bin_centers = hist_edges + (bin_width * 0.5)
        hist_main = HistPlottable(
            hist_values,
            # bin_centers,
            hist_edges,
            errors=hist_errors,
            scale_factor=scale_factor,
            normalize=normalize,
            binwidth_norm=binwidth_norm,
            ylabel=ylabel,
            xlabel=xlabel,
            legend_label=legend_labels[sample_type] if legend_labels else None,
        )
        hist_main.plot(ax=ax, **styles)
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
                hist["values"][1:-1],
                hist["edges"],
                errors=hist["errors"][1:-1],
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
    hist_names,
    energy,
    xlabel=None,
    ylabel="Entries",
    baseline=None,
    legend_labels: list = None,
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
    if isinstance(legend_labels, list):
        assert len(legend_labels) == len(hist_names), (
            "legend_labels list must have the same size as the number of samples in hists_group."
            f"Expected {len(hist_names)} labels, got {len(legend_labels)}."
        )

    if draw_ratio:
        fig = plt.figure()
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])
        ax, ax_ratio = gs.subplots(sharex=True)
    else:
        fig, ax = plt.subplots()

    for sample_type, sample_hists in hists_group.items():
        base_hist_values = None
        for i, hist_name in enumerate(hist_names):
            hist = sample_hists[hist_name]
            hist_values = hist["values"][1:-1]
            hist_edges = hist["edges"]
            hist_errors = hist["errors"][1:-1] if draw_errors else None
            hist_main = HistPlottable(
                hist_values,
                hist_edges,
                errors=hist_errors,
                scale_factor=scale_factors[i] if scale_factors else None,
                normalize=normalize,
                binwidth_norm=binwidth_norm,
                legend_label=legend_labels[i] if legend_labels else None,
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
                baseline_hist[1:-1],
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


def eff_err(
    arr: np.ndarray,
    n_counts: int,
    suppress_zero_divison_error: bool = False,
    norm: bool = False,
) -> np.ndarray:
    """Calculate statistical efficiency uncertainty.

    Parameters
    ----------
    arr : numpy.array
        Efficiency values
    n_counts : int
        Number of used statistics to calculate efficiency
    suppress_zero_divison_error : bool
        Not raising Error for zero division
    norm : bool, optional
        If True, normed (relative) error is being calculated, by default False

    Returns
    -------
    numpy.array
        Efficiency uncertainties

    Raises
    ------
    ValueError
        If n_counts <=0

    Notes
    -----
    This method uses binomial errors as described in section 2.2 of
    https://inspirehep.net/files/57287ac8e45a976ab423f3dd456af694
    """
    # logger.debug("Calculating efficiency error.")
    # logger.debug("arr: %s", arr)
    # logger.debug("n_counts: %i", n_counts)
    # logger.debug("suppress_zero_divison_error: %s", suppress_zero_divison_error)
    # logger.debug("norm: %s", norm)
    if np.any(n_counts <= 0) and not suppress_zero_divison_error:
        raise ValueError(
            f"You passed as argument `N` {n_counts} but it has to be larger 0."
        )
    if norm:
        return np.sqrt(arr * (1 - arr) / n_counts) / arr
    return np.sqrt(arr * (1 - arr) / n_counts)


def draw_efficiency(
    hists_group,
    hist_names,
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
    if isinstance(legend_labels, list):
        assert len(legend_labels) == len(hist_names), (
            "legend_labels must have the same size as the number of samples in hists_group."
            f"Expected {len(hist_names)} labels, got {len(legend_labels)}."
        )

    fig, ax = plt.subplots()

    for sample_type, sample_hists in hists_group.items():
        for i, hists in enumerate(hist_names):
            hist_total = sample_hists[hists["total"]]
            hist_pass = sample_hists[hists["pass"]]
            hist_pass_counts = hist_pass["values"][1:-1]
            hist_total_counts = hist_total["values"][1:-1]
            bins = hist_total["edges"]
            eff = hist_pass_counts / hist_total_counts
            hist_main = HistPlottable(
                eff,
                bins,
                legend_label=legend_labels[i] if legend_labels else None,
                xlabel=xlabel,
                ylabel=ylabel,
            )
            error_attrs = (
                {"histtype": "errorbar", "solid_capstyle": "projecting", "capsize": 3}
                if draw_errors
                else {}
            )
            hplt.histplot(
                hist_total_counts / np.max(hist_total_counts),
                bins * GeV,
                ax=ax,
                histtype="fill",
                color="silver",
                alpha=0.5,
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
    ax.set_ylim(ymin=ymin, ymax=ymax * (1 + 0.2 * len(hist_names) + 0.1))
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


def draw_mHH_plane_2D_hists(
    sample_hists,
    sample_name,
    hist_name,
    energy,
    luminosity=None,
    log_z=False,
    label_z="Entries",
    xrange=(50, 200),
    yrange=(50, 200),
    log_z_min=None,
    log_z_max=None,
    third_exp_label="",
    output_dir=Path("plots"),
):
    fig, ax = plt.subplots()
    hist = sample_hists[hist_name]
    is_data = "data" in sample_name
    bins_GeV = hist["edges"] * GeV
    hist_values = hist["values"]
    # remove outliers from hist_values
    hist_values[hist_values > 2000] = 0
    print(f"Counts for {sample_name} {hist_name}: {np.sum(hist_values[1:-1, 1:-1])}")
    print(
        f"Counts for with overflow/underflow {sample_name} {hist_name}: {np.sum(hist_values)}"
    )
    pcm, cbar, _ = hplt.hist2dplot(
        hist_values[1:-1, 1:-1],
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
    # if log_z:
    #     cbar.ax.set_yscale(
    #         "symlog",
    #         base=10,
    #         linthresh=100 if hist_values.max() > 100 else 0.015,
    #         linscale=0.05 if hist_values.max() > 100 else 1,
    #         subs=[2, 3, 4, 5, 6, 7, 8, 9],
    #     )

    if log_z:
        # Set default min and max if not provided
        if log_z_min is None:
            log_z_min = 100 if hist_values.max() > 100 else 0.015
        if log_z_max is None:
            log_z_max = hist_values.max()

        cbar.ax.set_yscale(
            "symlog",
            base=10,
            linthresh=log_z_min,
            linscale=0.05 if hist_values.max() > 100 else 1,
            subs=[2, 3, 4, 5, 6, 7, 8, 9],
        )
        cbar.ax.set_ylim(log_z_min, log_z_max)  # Set the y-axis limits for the colorbar

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
    plot_label = (
        sample_name.replace("_", " ")
        .replace("k01", "$\kappa_\lambda=1$")
        .replace("k05", "$\kappa_\lambda=5$")
        .replace("k10", "$\kappa_\lambda=10$")
    )
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


def draw_mHH_plane_3D_hists(
    sample_hists,
    sample_name,
    hist_name,
    energy,
    label_z="Entries",
    xrange: tuple = None,
    yrange: tuple = None,
    third_exp_label="",
    output_dir=Path("plots"),
):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    hist = sample_hists[hist_name]
    bins_GeV = hist["edges"] * GeV
    hist_values = hist["values"][1:-1, 1:-1]
    # Remove outliers
    hist_values[hist_values > 2000] = 0

    X, Y = np.meshgrid(bins_GeV[:-1], bins_GeV[:-1])
    Z = hist_values

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        cmap=plt.cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    cbar = fig.colorbar(surf, shrink=0.5)
    cbar.set_label(label_z)
    ax.view_init(azim=45)

    ax.set_xlabel(r"$m_{H1}$ [GeV]")
    ax.set_ylabel(r"$m_{H2}$ [GeV]")
    # ax.set_zlabel(label_z)
    if xrange:
        ax.set_xlim(*xrange)
    if yrange:
        ax.set_ylim(*yrange)

    plot_label = (
        sample_name.replace("_", " ")
        .replace("k01", "$\kappa_\lambda=1$")
        .replace("k05", "$\kappa_\lambda=5$")
        .replace("k10", "$\kappa_\lambda=10$")
    )
    plot_label = f"{plot_label}, {third_exp_label}" if third_exp_label else plot_label
    ax.set_title(f"{plot_label}, {energy} TeV")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"{sample_name}_{hist_name}_3D.png"
    fig.savefig(output_dir / plot_name, bbox_inches="tight")
    plt.close(fig)


def draw_mHH_plane_projections_hists(
    sample_hists,
    sample_name,
    h1_hist_name,
    h2_hist_name,
    hh_2d_hist_name,
    energy,
    luminosity=None,
    log_z=True,
    label_z="Entries",
    xrange: tuple = (50, 200),
    yrange: tuple = (50, 200),
    third_exp_label="",
    output_dir=Path("plots"),
):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    is_data = "data" in sample_name

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    h1_hist = sample_hists[h1_hist_name]
    h1_hist_values = h1_hist["values"][1:-1]
    h1_hist_bins_GeV = h1_hist["edges"] * GeV
    h2_hist = sample_hists[h2_hist_name]
    h2_hist_values = h2_hist["values"][1:-1]
    h2_hist_bins_GeV = h2_hist["edges"] * GeV
    hh_hist = sample_hists[hh_2d_hist_name]
    hh_hist_values = hh_hist["values"][1:-1, 1:-1]
    hh_bins_GeV = hh_hist["edges"] * GeV

    pcm, cbar, _ = hplt.hist2dplot(
        hh_hist_values,
        hh_bins_GeV,
        hh_bins_GeV,
        ax=ax,
        cbarpad=0.15,
        cbarsize="5%",
        flow=None,
        cbarextend=True,
        cmap="RdBu_r",
    )
    cbar.set_label(label_z)

    ax.set_xlabel(r"$m_{H1}$ [GeV]")
    ax.set_ylabel(r"$m_{H2}$ [GeV]")
    if xrange:
        ax.set_xlim(*xrange)
    if yrange:
        ax.set_ylim(*yrange)

    if log_z:
        cbar.ax.set_yscale(
            "symlog",
            base=10,
            linthresh=100 if hh_hist_values.max() > 100 else 0.015,
            linscale=0.05 if hh_hist_values.max() > 100 else 1,
            subs=[2, 3, 4, 5, 6, 7, 8, 9],
        )

    # signal and control region countours
    X, Y = np.meshgrid(hh_bins_GeV, hh_bins_GeV)
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

    # create new Axes on the right and on the top of the current Axes
    divider = make_axes_locatable(ax)
    # below height and pad are in inches
    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

    # make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    hplt.histplot(
        h1_hist_values,
        h1_hist_bins_GeV,
        histtype="fill",
        ax=ax_histx,
        label="mH1",
    )
    hplt.histplot(
        h2_hist_values,
        h2_hist_bins_GeV,
        histtype="fill",
        ax=ax_histy,
        label="mH2",
        orientation="horizontal",
    )

    ax_histx.set_yticks([h1_hist_values.max()])
    ax_histy.set_xticks([h2_hist_values.max()])

    plot_label = (
        sample_name.replace("_", " ")
        .replace("k01", "$\kappa_\lambda=1$")
        .replace("k05", "$\kappa_\lambda=5$")
        .replace("k10", "$\kappa_\lambda=10$")
    )
    plot_label = f"{plot_label}, {third_exp_label}" if third_exp_label else plot_label
    hplt.atlas.label(label=plot_label, loc=0, ax=ax_histx, lumi=luminosity, rlabel="")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"{sample_name}_{hh_2d_hist_name}_projections.png"
    fig.savefig(output_dir / plot_name, bbox_inches="tight")
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
