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
    fourpairs_combos = list(itertools.combinations(range(4), 2))
    combos = [(fourpairs_combos[i], fourpairs_combos[~i]) for i in range(3)]
    combos = ak.Array([combos] * len(values))
    combos_fourpairs_idx = ak.Array([[(i, ~i) for i in range(3)]] * len(values))
    combos_loss = np.transpose(
        ak.Array([values[:, i] + values[:, ~i] for i in range(3)])
    )
    combos_loss_min_idx = np.argmin(combos_loss, axis=1, keepdims=True)
    selected_fourpairs_combos_idx = combos_fourpairs_idx[combos_loss_min_idx]
    h1_jj_idx, h2_jj_idx = ak.unzip(selected_fourpairs_combos_idx)
    hh_jjjj_idx = ak.concatenate([h1_jj_idx, h2_jj_idx], axis=1)
    leading_h_jj_idx = hh_jjjj_idx[:, 0:1:2]
    return leading_h_jj_idx
    ## The bottom leads to sculpting effects in the mass distribution
    # min_h_idx = np.argmin(values[hh_jjjj_idx], axis=1, keepdims=True)
    # leading_h_jj_idx = hh_jjjj_idx[min_h_idx]
    # return leading_h_jj_idx


def optimizer_mass_pairing_v2(values, **kwargs):
    """
    HH jet min deviation from Higgs mass pairing.
    Optimization is the minimization of the reconstructed Higgs mass given by
    x^2 = (m_j1j2 - m_j3j4)^2 and so on for all possible pairings (3 total)
    """
    fourpairs_combos = list(itertools.combinations(range(4), 2))
    combos = [(fourpairs_combos[i], fourpairs_combos[~i]) for i in range(3)]
    combos = ak.Array([combos] * len(values))
    combos_fourpairs_idx = ak.Array([[(i, ~i) for i in range(3)]] * len(values))
    combos_loss = np.transpose(
        ak.Array([(values[:, i] - values[:, ~i]) ** 2 for i in range(3)])
    )
    combos_loss_min_idx = np.argmin(combos_loss, axis=1, keepdims=True)
    selected_fourpairs_combos_idx = combos_fourpairs_idx[combos_loss_min_idx]
    h1_jj_idx, h2_jj_idx = ak.unzip(selected_fourpairs_combos_idx)
    hh_jjjj_idx = ak.concatenate([h1_jj_idx, h2_jj_idx], axis=1)
    leading_h_jj_idx = hh_jjjj_idx[:, 0:1:2]
    return leading_h_jj_idx
    ## The bottom leads to sculpting effects in the mass distribution
    #    min_h_idx = np.argmin(values[hh_jjjj_idx], axis=1, keepdims=True)
    #    leading_h_jj_idx = hh_jjjj_idx[min_h_idxb
    #    return leading_h_jj_idx


pairing_methods = {
    "min_deltar_pairing": {
        "label": r"$\mathrm{arg\,min\,} \Delta R_{\mathrm{jj}}^{\mathrm{HC1}}$ pairing",
        "loss": lambda j_1, j_2: j_1.deltaR(j_2),
        "optimizer": np.argmin,
    },
    "max_deltar_pairing": {
        "label": r"$\mathrm{arg\,max\,} \Delta R_{\mathrm{jj}}^{\mathrm{HC1}}$ pairing",
        "loss": lambda j_1, j_2: j_1.deltaR(j_2),
        "optimizer": np.argmax,
    },
    "min_mass_true_pairing": {
        "label": r"$\mathrm{arg\,min\,} \Sigma(m_{jj}-m_\mathrm{H})^2$ pairing",
        "loss": lambda j_1, j_2: ((j_1 + j_2).mass - 125 * MeV) ** 2,
        "optimizer": optimizer_mass_pairing,
    },
    "min_mass_diff_pairing": {
        "label": r"$\mathrm{arg\,min} (m_{jj}-m_{jj})^2$ pairing",
        "loss": lambda j_1, j_2: (j_1 + j_2).mass,
        "optimizer": optimizer_mass_pairing_v2,
    },
    "min_mass_center_pairing": {
        "label": r"$\mathrm{arg\,min}(m_{jj}^2+m_{jj}^2)$ pairing",
        "loss": lambda j_1, j_2: ((j_1 + j_2).mass - 0.125 * MeV) ** 2,
        "optimizer": optimizer_mass_pairing,
    },
}
