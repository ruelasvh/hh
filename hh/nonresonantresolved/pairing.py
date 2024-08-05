import itertools
import numpy as np
import awkward as ak
from hh.shared.utils import MeV


def optimizer_mass_pairing(values, **kwargs):
    """
    HH jet min deviation from Higgs mass pairing.
    Optimization is the minimization of the deviation from the Higgs mass given by
    x^2 = (m_j1j2 - m_H)^2 + (m_j3j4 - m_H)^2 and so on for all possible pairings (3 total)
    """
    combos_loss = np.vstack([values[i, :] + values[~i, :] for i in range(3)])
    combos_loss_min_idx = np.argmin(combos_loss, axis=0)
    return combos_loss_min_idx


def optimizer_mass_pairing_v2(values, **kwargs):
    """
    HH jet min deviation from Higgs mass pairing.
    Optimization is the minimization of the reconstructed Higgs mass given by
    x^2 = (m_j1j2 - m_j3j4)^2 and so on for all possible pairings (3 total)
    """
    combos_loss = np.vstack([(values[i, :] - values[~i, :]) ** 2 for i in range(3)])
    combos_loss_min_idx = np.argmin(combos_loss, axis=0)
    return combos_loss_min_idx


pairing_methods = {
    "min_deltar_pairing": {
        "label": r"$\mathrm{arg\,min\,} \Delta R_{\mathrm{jj}}^{\mathrm{HC1}}$ pairing",
        "loss": lambda js, pair: np.where(
            (js[:, pair[0][0]] + js[:, pair[0][1]]).pt
            > (js[:, pair[1][0]] + js[:, pair[1][1]]).pt,
            js[:, pair[0][0]].deltaR(js[:, pair[0][1]]),
            js[:, pair[1][0]].deltaR(js[:, pair[1][1]]),
        ),
        "optimizer": np.argmin,
    },
    "max_deltar_pairing": {
        "label": r"$\mathrm{arg\,max\,} \Delta R_{\mathrm{jj}}^{\mathrm{HC1}}$ pairing",
        "loss": lambda js, pair: np.where(
            (js[:, pair[0][0]] + js[:, pair[0][1]]).pt
            > (js[:, pair[1][0]] + js[:, pair[1][1]]).pt,
            js[:, pair[0][0]].deltaR(js[:, pair[0][1]]),
            js[:, pair[1][0]].deltaR(js[:, pair[1][1]]),
        ),
        "optimizer": np.argmax,
    },
    "min_mass_true_pairing": {
        "label": r"$\mathrm{arg\,min\,} \Sigma(m_{jj}-m_\mathrm{H})^2$ pairing",
        "loss": lambda js, pair: (
            ((js[:, pair[0][0]] + js[:, pair[0][1]]).mass - 125 * MeV) ** 2
            + ((js[:, pair[1][0]] + js[:, pair[1][1]]).mass - 125 * MeV) ** 2
        ),
        "optimizer": np.argmin,
    },
    "min_mass_diff_pairing": {
        "label": r"$\mathrm{arg\,min} (m_{jj}-m_{jj})^2$ pairing",
        "loss": lambda js, pair: (
            (js[:, pair[0][0]] + js[:, pair[0][1]]).mass
            - (js[:, pair[1][0]] + js[:, pair[1][0]]).mass
        )
        ** 2,
        "optimizer": np.argmin,
    },
    "min_mass_center_pairing": {
        "label": r"$\mathrm{arg\,min}(m_{jj}^2+m_{jj}^2)$ pairing",
        "loss": lambda js, pair: (js[:, pair[0][0]] + js[:, pair[0][1]]).mass ** 2
        + (js[:, pair[1][0]] + js[:, pair[1][1]]).mass ** 2,
        "optimizer": np.argmin,
    },
}
