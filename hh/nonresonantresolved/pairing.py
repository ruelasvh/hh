import itertools
import numpy as np
import awkward as ak
from hh.shared.utils import MeV


## v13
def optimizer_delta_r_pairing(
    hc_jets, hh_jet_idx, loss, separation=np.argmin, n_jets=4, n_pairs=2, **kwargs
):
    """
    Return the the leading Higgs candidate pairings based on the delta R between the two jets
    """

    # Only consider events with 4 jets for this pairing
    n_jets = 4
    n_pairs = 2

    def pairing_lead(js, pair):
        """
        Return the loss of the leading Higgs candidate transverse momentum
        """
        # for n_jets = 4, n_pairs = 2
        # pair_to_idx = {0: ((0, 1), (2, 3)), 1: ((0, 2), (1, 3)), 2: ((0, 3), (1, 2))}
        pairings = list(itertools.combinations(range(n_jets), n_pairs))
        pair_to_idx = {i: (pairings[i], pairings[~i]) for i in range(n_pairs + 1)}
        (ia0, ia1), (ib0, ib1) = pair_to_idx[pair]
        # pt ordering
        hc_sort = (js[:, ia0] + js[:, ia1]).pt > (js[:, ib0] + js[:, ib1]).pt
        lead = np.where(
            hc_sort, loss(js[:, ia0], js[:, ia1]), loss(js[:, ib0], js[:, ib1])
        )
        return lead

    valid_event_mask = ~ak.is_none(hc_jets, axis=0)
    chosen_pair = separation(
        np.vstack([pairing_lead(hc_jets, i) for i in range(n_pairs + 1)]), axis=0
    )
    chosen_pair = ak.mask(chosen_pair, ~ak.is_none(hc_jets, axis=0))

    ja0 = hc_jets[:, 0]
    ja0_idx = hh_jet_idx[:, 0, None]
    chosen_pair_mask = ak.local_index(hc_jets) == (chosen_pair + 1)
    ja1 = hc_jets[chosen_pair_mask]
    ja1_idx = hh_jet_idx[chosen_pair_mask]

    jb0 = ak.where(chosen_pair == 0, hc_jets[:, 2], hc_jets[:, 1])
    jb0_idx = ak.where(chosen_pair == 0, hh_jet_idx[:, 2], hh_jet_idx[:, 1])
    jb0_idx = ak.mask(jb0_idx[:, None], valid_event_mask)
    jb1 = ak.where(chosen_pair == 2, hc_jets[:, 2], hc_jets[:, 3])
    jb1_idx = ak.where(chosen_pair == 2, hh_jet_idx[:, 2], hh_jet_idx[:, 3])
    jb1_idx = ak.mask(jb1_idx[:, None], valid_event_mask)

    hca = ja0 + ja1
    hca_idx = ak.concatenate([ja0_idx, ja1_idx], axis=1)
    hcb = jb0 + jb1
    hcb_idx = ak.concatenate([jb0_idx, jb1_idx], axis=1)

    # The scalar candidate with the mass closest to the Higgs mass
    # will be the Higgs candidate
    sort_mask = ak.firsts(hca.pt > hcb.pt)
    hc1_idx = ak.where(sort_mask, hca_idx, hcb_idx)
    hc2_idx = ak.where(sort_mask, hcb_idx, hca_idx)

    return hc1_idx, hc2_idx


# def optimizer_mass_pairing(loss, **kwargs):
#     """
#     HH jet min deviation from Higgs mass pairing.
#     Optimization is the minimization of the deviation from the Higgs mass given by
#     x^2 = (m_j1j2 - m_H)^2 + (m_j3j4 - m_H)^2 and so on for all possible pairings (3 total)
#     """
#     pairings = list(itertools.combinations(range(n_jets), n_pairs))
#     pair_to_idx = {i: (pairings[i], pairings[~i]) for i in range(n_pairs + 1)}
#     (ia0, ia1), (ib0, ib1) = pair_to_idx[pair]
#     # pt ordering
#     hc_sort = (js[:, ia0] + js[:, ia1]).pt > (js[:, ib0] + js[:, ib1]).pt
#     loss = lambda j_1, j_2: ((j_1 + j_2).mass - 125 * MeV) ** 2
#     losses = loss(js[:, ia0], js[:, ia1]) < loss(js[:, ib0], js[:, ib1]]
#     lead = np.where(
#         hc_sort, loss(js[:, ia0], js[:, ia1]), loss(js[:, ib0], js[:, ib1])
#     )
#     losses =
#     combos_loss = np.vstack([values[i, :] + values[~i, :] for i in range(3)])
#     combos_loss_min_idx = np.argmin(combos_loss, axis=0)
#     return combos_loss_min_idx


# v12
def optimizer_mass_pairing(values, **kwargs):
    """
    HH jet min deviation from Higgs mass pairing.
    Optimization is the minimization of the deviation from the Higgs mass given by
    x^2 = (m_j1j2 - m_H)^2 + (m_j3j4 - m_H)^2 and so on for all possible pairings (3 total)
    """
    combos_loss = np.vstack([values[i, :] + values[~i, :] for i in range(3)])
    combos_loss_min_idx = np.argmin(combos_loss, axis=0)
    return combos_loss_min_idx


# ## v13
# def optimizer_mass_pairing(
#     hc_jets, hh_jet_idx, loss, separation=np.argmin, n_jets=4, n_pairs=2, **kwargs
# ):
#     """
#     HH jet min deviation from Higgs mass pairing.
#     Optimization is the minimization of the deviation from the Higgs mass given by
#     x^2 = (m_j1j2 - m_H)^2 + (m_j3j4 - m_H)^2 and so on for all possible pairings (3 total)
#     """
#     pairings = list(itertools.combinations(range(n_jets), n_pairs))
#     loss_values = np.transpose(
#         ak.Array(loss(hc_jets[:, i], hc_jets[:, j]) for i, j in pairings)
#     )
#     loss_values = ak.mask(loss_values, ~ak.is_none(hc_jets))

#     fourpairs_combos = list(itertools.combinations(range(4), 2))
#     combos = [(fourpairs_combos[i], fourpairs_combos[~i]) for i in range(3)]
#     combos = ak.Array([combos] * len(loss_values))
#     combos_fourpairs_idx = ak.Array([[(i, ~i) for i in range(3)]] * len(loss_values))
#     combos_loss = np.transpose(
#         ak.Array([loss_values[:, i] + loss_values[:, ~i] for i in range(3)])
#     )
#     combos_loss_min_idx = separation(combos_loss, axis=1, keepdims=True)
#     selected_fourpairs_combos_idx = combos_fourpairs_idx[combos_loss_min_idx]
#     h1_jj_idx, h2_jj_idx = ak.unzip(selected_fourpairs_combos_idx)
#     hh_jjjj_idx = ak.concatenate([h1_jj_idx, h2_jj_idx], axis=1)
#     leading_h_jj_idx = hh_jjjj_idx[:, 0:1:2]

#     optimized_loss_idx = leading_h_jj_idx
#     optimized_loss_idx = ak.mask(optimized_loss_idx, ~ak.is_none(hc_jets))
#     pairings = ak.argcombinations(hc_jets, n_pairs)
#     h1_jet_idx = ak.concatenate(ak.unzip(pairings[optimized_loss_idx]), axis=1)
#     h2_jet_idx = ak.concatenate(ak.unzip(pairings[~optimized_loss_idx]), axis=1)

#     return hh_jet_idx[h1_jet_idx], hh_jet_idx[h2_jet_idx]


# def optimizer_mass_pairing_v2(values, **kwargs):
#     """
#     HH jet min deviation from Higgs mass pairing.
#     Optimization is the minimization of the reconstructed Higgs mass given by
#     x^2 = (m_j1j2 - m_j3j4)^2 and so on for all possible pairings (3 total)
#     """
#     combos_fourpairs_idx = ak.Array([[(i, ~i) for i in range(3)]] * len(values))
#     combos_loss = np.transpose(
#         ak.Array([(values[:, i] - values[:, ~i]) ** 2 for i in range(3)])
#     )
#     combos_loss_min_idx = np.argmin(combos_loss, axis=1, keepdims=True)
#     selected_fourpairs_combos_idx = combos_fourpairs_idx[combos_loss_min_idx]
#     h1_jj_idx, h2_jj_idx = ak.unzip(selected_fourpairs_combos_idx)
#     hh_jjjj_idx = ak.concatenate([h1_jj_idx, h2_jj_idx], axis=1)
#     leading_h_jj_idx = hh_jjjj_idx[:, 0:1:2]
#     return leading_h_jj_idx


# v12
def optimizer_mass_pairing_v2(values, **kwargs):
    """
    HH jet min deviation from Higgs mass pairing.
    Optimization is the minimization of the reconstructed Higgs mass given by
    x^2 = (m_j1j2 - m_j3j4)^2 and so on for all possible pairings (3 total)
    """
    combos_loss = np.vstack([(values[i, :] - values[~i, :]) ** 2 for i in range(3)])
    combos_loss_min_idx = np.argmin(combos_loss, axis=0)
    return combos_loss_min_idx


# ## v13
# def optimizer_mass_pairing_v2(
#     hc_jets, hh_jet_idx, loss, separation=np.argmin, n_jets=4, n_pairs=2, **kwargs
# ):
#     """
#     HH jet min deviation from Higgs mass pairing.
#     Optimization is the minimization of the reconstructed Higgs mass given by
#     x^2 = (m_j1j2 - m_j3j4)^2 and so on for all possible pairings (3 total)
#     """
#     pairings = list(itertools.combinations(range(n_jets), n_pairs))
#     loss_values = np.transpose(
#         ak.Array(loss(hc_jets[:, i], hc_jets[:, j]) for i, j in pairings)
#     )
#     loss_values = ak.mask(loss_values, ~ak.is_none(hc_jets))

#     fourpairs_combos = list(itertools.combinations(range(4), 2))
#     combos = [(fourpairs_combos[i], fourpairs_combos[~i]) for i in range(3)]
#     combos = ak.Array([combos] * len(loss_values))
#     combos_fourpairs_idx = ak.Array([[(i, ~i) for i in range(3)]] * len(loss_values))
#     combos_loss = np.transpose(
#         ak.Array([(loss_values[:, i] - loss_values[:, ~i]) ** 2 for i in range(3)])
#     )
#     combos_loss_min_idx = separation(combos_loss, axis=1, keepdims=True)
#     selected_fourpairs_combos_idx = combos_fourpairs_idx[combos_loss_min_idx]
#     h1_jj_idx, h2_jj_idx = ak.unzip(selected_fourpairs_combos_idx)
#     hh_jjjj_idx = ak.concatenate([h1_jj_idx, h2_jj_idx], axis=1)
#     leading_h_jj_idx = hh_jjjj_idx[:, 0:1:2]

#     optimized_loss_idx = leading_h_jj_idx
#     optimized_loss_idx = ak.mask(optimized_loss_idx, ~ak.is_none(hc_jets))
#     pairings = ak.argcombinations(hc_jets, n_pairs)
#     h1_jet_idx = ak.concatenate(ak.unzip(pairings[optimized_loss_idx]), axis=1)
#     h2_jet_idx = ak.concatenate(ak.unzip(pairings[~optimized_loss_idx]), axis=1)

#     return hh_jet_idx[h1_jet_idx], hh_jet_idx[h2_jet_idx]


pairing_methods = {
    "min_deltar_pairing": {
        "label": r"$\mathrm{arg\,min\,} \Delta R_{\mathrm{jj}}^{\mathrm{HC1}}$ pairing",
        # "loss": lambda j_1, j_2: j_1.deltaR(j_2), # v01 - v13
        "loss": lambda js, pair: np.where(
            (js[:, pair[0][0]] + js[:, pair[0][1]]).pt
            > (js[:, pair[1][0]] + js[:, pair[1][1]]).pt,
            js[:, pair[0][0]].deltaR(js[:, pair[0][1]]),
            js[:, pair[1][0]].deltaR(js[:, pair[1][1]]),
        ),  # v14
        # rewrite the loss function to include the pt ordering
        "optimizer": np.argmin,  # v12 and v14
        # "optimizer": optimizer_delta_r_pairing, #v11 and v13
        # "separation": np.argmin, #v13
    },
    "max_deltar_pairing": {
        "label": r"$\mathrm{arg\,max\,} \Delta R_{\mathrm{jj}}^{\mathrm{HC1}}$ pairing",
        # "loss": lambda j_1, j_2: j_1.deltaR(j_2),  # v01 - v13
        "loss": lambda js, pair: np.where(
            (js[:, pair[0][0]] + js[:, pair[0][1]]).pt
            > (js[:, pair[1][0]] + js[:, pair[1][1]]).pt,
            js[:, pair[0][0]].deltaR(js[:, pair[0][1]]),
            js[:, pair[1][0]].deltaR(js[:, pair[1][1]]),
        ),  # v14
        "optimizer": np.argmax,  # v12
        # "optimizer": optimizer_delta_r_pairing,  # v11 and v13
        # "separation": np.argmax,  # v13
    },
    "min_mass_true_pairing": {
        "label": r"$\mathrm{arg\,min\,} \Sigma(m_{jj}-m_\mathrm{H})^2$ pairing",
        # "loss": lambda j_1, j_2: ((j_1 + j_2).mass - 125 * MeV) ** 2,  # v01 - v13
        "loss": lambda js, pair: (
            ((js[:, pair[0][0]] + js[:, pair[0][1]]).mass - 125 * MeV) ** 2
            + ((js[:, pair[1][0]] + js[:, pair[1][1]]).mass - 125 * MeV) ** 2
        ),  # v14
        # "optimizer": optimizer_mass_pairing,  # v12 and v13
        # "separation": np.argmin,  # v13
        "optimizer": np.argmin,  # v11 and v14
    },
    "min_mass_diff_pairing": {
        "label": r"$\mathrm{arg\,min} (m_{jj}-m_{jj})^2$ pairing",
        # "loss": lambda j_1, j_2: (j_1 + j_2).mass,  # v01 - v13
        "loss": lambda js, pair: (
            (js[:, pair[0][0]] + js[:, pair[0][1]]).mass
            - (js[:, pair[1][0]] + js[:, pair[1][0]]).mass
        )
        ** 2,  # v14
        # "optimizer": optimizer_mass_pairing_v2,  # v12 - v13
        # "separation": np.argmin,  # v13
        "optimizer": np.argmin,  # v11 and v14
    },
    "min_mass_center_pairing": {
        "label": r"$\mathrm{arg\,min}(m_{jj}^2+m_{jj}^2)$ pairing",
        # "loss": lambda j_1, j_2: ((j_1 + j_2).mass) ** 2,  # v01 - v13
        "loss": lambda js, pair: (js[:, pair[0][0]] + js[:, pair[0][1]]).mass ** 2
        + (js[:, pair[1][0]] + js[:, pair[1][1]]).mass ** 2,  # v14,
        # "optimizer": optimizer_mass_pairing,  # v12 - v13
        # "separation": np.argmin,  # v13
        "optimizer": np.argmin,  # v11 and v14
    },
}
