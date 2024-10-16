import itertools as it
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


# Function to calculate the sum of squared differences for both leading and subleading Higgs
def sum_squared_diffs(m_lead, m_sub, m_X_lead, m_X_sub):
    return (m_lead - m_X_lead) ** 2 + (m_sub - m_X_sub) ** 2


# Function to find the best m_X_lead and m_X_sub from a scan
def scan_m_X(jets, m_X_lead_range, m_X_sub_range, n_jets=4, n_pairs=2):
    num_jets = np.size(jets, 1)
    if num_jets != n_jets:
        raise ValueError(f"Exactly {n_jets} jets are required.")

    best_min_sum_diff = np.inf
    best_m_X_lead = None
    best_m_X_sub = None

    # Iterate over all combinations of 2 jets from 4 (to form the first pair)
    jet_pairs = list(it.combinations(range(n_jets), n_pairs))
    jet_pairs_combos = [(jet_pairs[i], jet_pairs[~i]) for i in range(n_pairs + 1)]
    # for jet_pair_1, jet_pair_2 in it.combinations(jet_pairs, 2):
    for jet_pair_1, jet_pair_2 in jet_pairs_combos:
        # Calculate the invariant masses of both pairs
        m1 = (jets[:, jet_pair_1[0]] + jets[:, jet_pair_1[1]]).mass.to_numpy()
        m2 = (jets[:, jet_pair_2[0]] + jets[:, jet_pair_2[1]]).mass.to_numpy()

        # Assign the higher mass to leading Higgs and lower mass to subleading Higgs
        m_lead, m_sub = np.maximum(m1, m2), np.minimum(m1, m2)

        # Scan over m_X_lead and m_X_sub ranges
        for m_X_lead in m_X_lead_range:
            for m_X_sub in m_X_sub_range:
                sum_diff = sum_squared_diffs(m_lead, m_sub, m_X_lead, m_X_sub)
                min_sum_diff = np.min(sum_diff)
                if min_sum_diff < best_min_sum_diff:
                    best_min_sum_diff = min_sum_diff
                    best_m_X_lead = m_X_lead
                    best_m_X_sub = m_X_sub

    return best_m_X_lead, best_m_X_sub


pairing_methods = {
    # "min_deltar_pairing": {
    #     "label": r"$\mathrm{arg\,min\,} \Delta R_{\mathrm{jj}}^{\mathrm{HC1}}$ pairing",
    #     "loss": lambda jet_p4, jet_pair_1, jet_pair_2: np.where(
    #         (jet_p4[:, jet_pair_1[0]] + jet_p4[:, jet_pair_1[1]]).pt
    #         > (jet_p4[:, jet_pair_2[0]] + jet_p4[:, jet_pair_2[1]]).pt,
    #         jet_p4[:, jet_pair_1[0]].deltaR(jet_p4[:, jet_pair_1[1]]),
    #         jet_p4[:, jet_pair_2[0]].deltaR(jet_p4[:, jet_pair_2[1]]),
    #     ),
    #     "optimizer": np.argmin,
    # },
    # "max_deltar_pairing": {
    #     "label": r"$\mathrm{arg\,max\,} \Delta R_{\mathrm{jj}}^{\mathrm{HC1}}$ pairing",
    #     "loss": lambda jet_p4, jet_pair_1, jet_pair_2: np.where(
    #         (jet_p4[:, jet_pair_1[0]] + jet_p4[:, jet_pair_1[1]]).pt
    #         > (jet_p4[:, jet_pair_2[0]] + jet_p4[:, jet_pair_2[1]]).pt,
    #         jet_p4[:, jet_pair_1[0]].deltaR(jet_p4[:, jet_pair_1[1]]),
    #         jet_p4[:, jet_pair_2[0]].deltaR(jet_p4[:, jet_pair_2[1]]),
    #     ),
    #     "optimizer": np.argmax,
    # },
    # "min_mass_true_pairing": {
    #     "label": r"$\mathrm{arg\,min\,} \Sigma(m_{jj}-m_\mathrm{H})^2$ pairing",
    #     "loss": lambda jet_p4, jet_pair_1, jet_pair_2: (
    #         ((jet_p4[:, jet_pair_1[0]] + jet_p4[:, jet_pair_1[1]]).mass - 125 * MeV)
    #         ** 2
    #         + ((jet_p4[:, jet_pair_2[0]] + jet_p4[:, jet_pair_2[1]]).mass - 125 * MeV)
    #         ** 2
    #     ),
    #     "optimizer": np.argmin,
    # },
    # "min_mass_diff_pairing": {
    #     "label": r"$\mathrm{arg\,min} (m_{jj}-m_{jj})^2$ pairing",
    #     "loss": lambda jet_p4, jet_pair_1, jet_pair_2: (
    #         (jet_p4[:, jet_pair_1[0]] + jet_p4[:, jet_pair_1[1]]).mass
    #         - (jet_p4[:, jet_pair_2[0]] + jet_p4[:, jet_pair_2[0]]).mass
    #     )
    #     ** 2,
    #     "optimizer": np.argmin,
    # },
    # "min_mass_center_pairing": {
    #     "label": r"$\mathrm{arg\,min}(m_{jj}^2+m_{jj}^2)$ pairing",
    #     "loss": lambda jet_p4, jet_pair_1, jet_pair_2: (
    #         jet_p4[:, jet_pair_1[0]] + jet_p4[:, jet_pair_1[1]]
    #     ).mass
    #     ** 2
    #     + (jet_p4[:, jet_pair_2[0]] + jet_p4[:, jet_pair_2[1]]).mass ** 2,
    #     "optimizer": np.argmin,
    # },
    "min_mass_optimized_pairing": {
        "label": r"$\mathrm{arg\,min\,} ((m_{jj}^{lead}-m_\mathrm{X}^{lead})^2 + (m_{jj}^{sub}-m_\mathrm{X}^{sub})^2)$ pairing",
        "loss": lambda m_X_lead, m_X_sub: lambda jet_p4, jet_pair_1, jet_pair_2: (
            (
                np.maximum(
                    (jet_p4[:, jet_pair_1[0]] + jet_p4[:, jet_pair_1[1]]).mass,
                    (jet_p4[:, jet_pair_2[0]] + jet_p4[:, jet_pair_2[1]]).mass,
                )
                - m_X_lead
            )
            ** 2
            + (
                np.minimum(
                    (jet_p4[:, jet_pair_1[0]] + jet_p4[:, jet_pair_1[1]]).mass,
                    (jet_p4[:, jet_pair_2[0]] + jet_p4[:, jet_pair_2[1]]).mass,
                )
                - m_X_sub
            )
            ** 2
        ),
        "optimizer": np.argmin,
        "m_X_lead_range": np.linspace(0, 150, 16),
        "m_X_sub_range": np.linspace(0, 150, 16),
    },
    "min_mass_optimized_1d_pairing": {
        "label": r"$\mathrm{arg\,min\,} \Sigma(m_{jj}-m_\mathrm{X})^2$ pairing",
        "loss": lambda m_X: lambda jet_p4, jet_pair_1, jet_pair_2: (
            ((jet_p4[:, jet_pair_1[0]] + jet_p4[:, jet_pair_1[1]]).mass - m_X) ** 2
            + ((jet_p4[:, jet_pair_2[0]] + jet_p4[:, jet_pair_2[1]]).mass - m_X) ** 2
        ),
        "optimizer": np.argmin,
        "m_X_range": [],
    },
}
