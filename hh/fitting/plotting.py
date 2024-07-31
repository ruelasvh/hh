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
    textstr = "\n".join(
        (
            plot_label,
            r"Upper limit (obs): $\mu=%.4f$" % (limits.observed_limit,),
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
        rlabel=get_com_lumi_label(luminosity, energy) + exp_label,
        loc=4,
        ax=ax,
        pad=0.01,
    )
    fig.savefig(f"{figure_folder}/limit.pdf")
    plt.close(fig)
