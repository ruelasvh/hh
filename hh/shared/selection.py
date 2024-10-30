import numpy as np
import awkward as ak
import vector as p4
from hh.shared.labels import kin_labels


def calculate_discrim(x, y, x_center, y_center, x_res=0.1, y_res=0.1):
    """General function to calculate the discriminant for a given variable."""

    # first_term = np.zeros_like(x)
    # np.divide(x - x_center, x_res * x, out=first_term, where=(x != 0))
    # second_term = np.zeros_like(y)
    # np.divide(y - y_center, y_res * y, out=second_term, where=(y != 0))
    # return np.sqrt(first_term**2 + second_term**2)
    return np.sqrt(
        ((x - x_center) / (x_res * x)) ** 2 + ((y - y_center) / (y_res * y)) ** 2
    )


def X_HH(m_H1, m_H2, m_H1_center=124, m_H2_center=117):
    """Calculate signal region discriminat.

    X_HH = sqrt(
        ((m_H1 - 124 GeV) / 0.1 * m_H1)^2 + ((m_H2 - 117 GeV) / 0.1 * m_H2)^2
    )
    """

    return calculate_discrim(m_H1, m_H2, m_H1_center, m_H2_center)


def R_CR(m_H1, m_H2, m_H1_center=124, m_H2_center=117):
    """Calculate outer edge of control region discriminant.

    R_CR = sqrt(
        (m_H1 - 1.05 * 124 GeV)^2 + (m_H2 - 1.05 * 117 GeV)^2
    )
    """

    # return np.sqrt((m_H1 - 1.05 * m_H1_center) ** 2 + (m_H2 - 1.05 * m_H2_center) ** 2)
    return calculate_discrim(
        m_H1,
        m_H2,
        1.05 * m_H1_center,
        1.05 * m_H2_center,
        x_res=1 / m_H1,
        y_res=1 / m_H2,
    )


def X_Wt(m_jj, m_jjb, m_W=80.4, m_t=172.5):
    """Calculate top-veto discriminant. Where m_jj is the mass of the W candidate
    and m_jjb is the mass of the top candidate.

    X_Wt = sqrt(
        ((m_jj - 80.4 GeV) / 0.1 * m_jj)^2 + ((m_jjb - 172.5 GeV) / 0.1 * m_jjb)^2
    )
    """

    return calculate_discrim(m_jj, m_jjb, m_W, m_t)


def get_W_t_p4(jets, hh_jet_idx, non_hh_jet_idx):
    jet_idx = ak.concatenate([hh_jet_idx, non_hh_jet_idx], axis=1)
    jets = jets[jet_idx]
    bjet_p4 = jets[:, :4][jets.btag[:, :4] == 1]
    bjet_idx = ak.local_index(bjet_p4, axis=1)
    # for one event with hc_jet_indices with 4 jets, remaining_jets with 2 jets
    # jet_indices -> [[0, 1, 2, 3, 4, 5], ...]
    # hc_jet_indices -> [[0, 2, 3, 5], ...]
    jet_pair_combinations_indices = ak.argcombinations(
        jets, 2, axis=1
    )  # [[(0, 1), (0, 2), (0, 3), (0, 4), (...), ..., (2, 5), (3, 4), (3, 5), (4, 5)], ...]
    top_candidate_indices = ak.cartesian(
        [bjet_idx, jet_pair_combinations_indices], axis=1
    )  # [[(0, (0, 1)), (0, (0, 2)), (0, (0, 3)), (0, (0, 4)), (0, (0, 5)), (0, (1, 2)), (0, (1, 3)), (0, (1, 4)), (0, (1, 5)), (0, (2, 3)), (0, (2, 4)), (0, (2, 5)), (0, (3, 4)), (0, (3, 5)), (0, (4, 5))], ...]
    top_jet3_indices, top_jet12_indices = ak.unzip(
        top_candidate_indices
    )  # [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]], [[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), ...]]
    top_jet1_indices, top_jet2_indices = ak.unzip(
        top_jet12_indices
    )  # [[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, ...]], [[1, 2, 3, 4, 5, 2, 3, 4, 5, 3, ...]]
    valid_top_candidate_mask = (top_jet3_indices != top_jet2_indices) & (
        top_jet3_indices != top_jet1_indices
    )  # [[False, False, False, False, False, True, True, True, True, True, ...]]
    top_jet1_indices = top_jet1_indices[valid_top_candidate_mask]
    top_jet2_indices = top_jet2_indices[valid_top_candidate_mask]
    top_jet3_indices = top_jet3_indices[valid_top_candidate_mask]
    W_candidates_p4 = jets[top_jet1_indices] + jets[top_jet2_indices]
    top_candidates_p4 = W_candidates_p4 + jets[top_jet3_indices]
    return W_candidates_p4, top_candidates_p4


def rotate_points(points, angle_degrees):
    """Rotate points counterclockwise by a given angle in degrees."""
    theta = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    return np.dot(points, rotation_matrix)


def classify_control_events(
    points,
    angle_degrees=45,
    x_center=124,
    y_center=117,
):
    """Classify points into quadrants based on concentric ellipses and a rotated coordinate system."""
    # Translate points to the center of the ellipse
    points = points - np.array([x_center, y_center])
    # Rotate points by the specified angle
    rotated_points = rotate_points(points, angle_degrees)
    # Classify rotated points into quadrants
    quadrants = {}
    quadrants[1] = (rotated_points[:, 0] > 0) & (rotated_points[:, 1] > 0)
    quadrants[2] = (rotated_points[:, 0] < 0) & (rotated_points[:, 1] > 0)
    quadrants[3] = (rotated_points[:, 0] < 0) & (rotated_points[:, 1] < 0)
    quadrants[4] = (rotated_points[:, 0] > 0) & (rotated_points[:, 1] < 0)
    return quadrants
