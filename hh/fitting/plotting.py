import cabinetry
import mplhep as hplt
import matplotlib.pyplot as plt
from hh.shared.utils import get_com_lumi_label

plt.style.use(hplt.style.ATLAS)


def plot_limits(
    limits, exp_label, plot_label, luminosity, energy, figure_folder, **kwargs
):
    fig = cabinetry.visualize.limit(
        limits, close_figure=False, save_figure=False, **kwargs
    )
    ax = fig.gca()
    minus_2_sigma, minus_1_sigma, mu, plus_1_sigma, plus_2_sigma = expected_lim = (
        limits.expected_limit
    )
    textstr = "\n".join(
        (
            plot_label,
            r"Expected $\mu$: %.4f" % (mu,),
            r"$-2 \sigma$: %.4f (%.4f)" % (minus_2_sigma, mu - minus_2_sigma),
            r"$-1 \sigma$: %.4f (%.4f)" % (minus_1_sigma, mu - minus_1_sigma),
            r"$+1 \sigma$: %.4f (%.4f)" % (plus_1_sigma, plus_1_sigma - mu),
            r"$+2 \sigma$: %.4f (%.4f)" % (plus_2_sigma, plus_2_sigma - mu),
        )
    )
    # Plot details label
    ax.text(
        0.5,
        0.5,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
    )
    # Experiemental label
    hplt.atlas.label(
        # label="Work In Progress",
        rlabel=get_com_lumi_label(energy, luminosity) + exp_label,
        loc=4,
        ax=ax,
        pad=0.01,
    )
    fig.savefig(f"{figure_folder}/limit.pdf")
    plt.close(fig)
