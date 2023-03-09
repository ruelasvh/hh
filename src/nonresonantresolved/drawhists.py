import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import logging
from .utils import find_hist
from .selection import X_HH, R_CR

plt.style.use(hep.style.ATLAS)

invGeV = 1 / 1_000


def draw_hists(hists: list, sample_name: str, args: dict) -> None:
    """Draw all the histrograms"""

    logging.info(f"Drawing hitograms for sample type: {sample_name}")
    draw_mH_plane(hists, sample_name)


def draw_mH_plane(hists, sample_name):
    hist = find_hist(hists, lambda h: "mH_plane" in h.name)
    fig, ax = plt.subplots()
    binsGeV = hist.edges * invGeV
    hep.hist2dplot(
        hist.values,
        binsGeV,
        binsGeV,
        ax=ax,
        cbarpad=0.15,
        cbarsize="5%",
        # cmap=plt.get_cmap("PRGn"),
    )
    ax.set_ylabel(r"$m_{H2}$ [GeV]")
    ax.set_xlabel(r"$m_{H1}$ [GeV]")
    ax.set_ylim(50, 200)
    ax.set_xlim(50, 200)
    X, Y = np.meshgrid(binsGeV, binsGeV)
    X_HH_discrim = X_HH(X, Y)
    ax.contour(X, Y, X_HH_discrim, levels=[1.55, 1.6], colors=["red", "black"])
    R_CR_discrim = R_CR(X, Y)
    ax.contour(X, Y, R_CR_discrim, levels=[45], colors=["black"])
    fig.savefig(
        f"plots/mH_plane_{sample_name}.png",
        bbox_inches="tight",
    )
    plt.close()
