import awkward as ak
import numpy as np
import vector as p4
from src.nonresonantresolved.branches import get_jet_branch_alias_names


def select_n_jets_events(
    events,
    jet_vars=None,
    pt_cut=40_000,
    eta_cut=2.5,
    njets_cut=4,
):
    """Selects events by applying the cuts specified in the arguments.
    The jet pT and eta column names should be the first and second
    items in jet_vars."""

    if jet_vars is None:
        return events
    sorted_events = sort_jets_by_pt(events, jet_vars)
    jet_pt = sorted_events[jet_vars[0]]
    jet_eta = sorted_events[jet_vars[1]]
    if pt_cut and eta_cut:
        valid_events = (jet_pt > pt_cut) & (np.abs(jet_eta) < eta_cut)
    elif pt_cut and not eta_cut:
        valid_events = jet_pt > pt_cut
    elif eta_cut and not pt_cut:
        valid_events = np.abs(jet_eta) < eta_cut
    else:
        valid_events = None
    if valid_events is not None:
        valid_events = ak.num(jet_pt[valid_events]) >= njets_cut
    else:
        valid_events = ak.num(jet_pt) >= njets_cut

    return sorted_events[valid_events]


def select_n_bjets(events, jet_vars=None, btag_cut=None, nbjets_cut=4):
    """Selects events by applying the cuts specified in the arguments.
    The jet pT and eta column names should be the first and second
    items in jet_vars. Assumes jets are already sorted by pT."""

    if jet_vars is None or btag_cut is None:
        return events

    four_plus_btags = ak.sum(events[btag_cut], axis=1) >= nbjets_cut
    events_with_four_plus_btags = events[four_plus_btags]
    if len(events_with_four_plus_btags) == 0:
        return events_with_four_plus_btags, events_with_four_plus_btags
    btag_decisions = events_with_four_plus_btags[btag_cut] == 1
    btag_indices = ak.local_index(btag_decisions)
    leading_four_bjet_indices = btag_indices[btag_decisions][:, :nbjets_cut]
    remaining_btag_indices = btag_indices[btag_decisions][:, nbjets_cut:]
    remaining_jet_indices = ak.concatenate(
        [btag_indices[~btag_decisions], remaining_btag_indices], axis=1
    )
    leading_four_bjets = events_with_four_plus_btags[jet_vars][
        leading_four_bjet_indices
    ]
    remaining_jets = events_with_four_plus_btags[jet_vars][remaining_jet_indices]
    return leading_four_bjets, remaining_jets


def sort_jets_by_pt(events, jet_vars=None):
    """Sorts events by jet pT. The jet pT column name should be the first
    item in jet_vars."""

    if jet_vars is None:
        return events
    sorted_index = ak.argsort(events[jet_vars[0]], ascending=False)
    sorted_jets = events[
        jet_vars,
        sorted_index,
    ]
    for var in jet_vars:
        sorted_events_by_jet_pt = ak.with_field(events, sorted_jets[var], var)
    return sorted_events_by_jet_pt


def X_HH(m_H1, m_H2):
    """Calculate signal region discriminat.

    X_HH = sqrt(
        ((m_H1 - 124 GeV) / 0.1 * m_H1)^2 + ((m_H2 - 117 GeV) / 0.1 * m_H2)^2
    )
    """

    first_term = np.zeros_like(m_H1)
    np.divide(m_H1 - 124, 0.1 * m_H1, out=first_term, where=(m_H1 != 0))
    second_term = np.zeros_like(m_H2)
    np.divide(m_H2 - 117, 0.1 * m_H2, out=second_term, where=(m_H2 != 0))

    return np.sqrt(first_term**2 + second_term**2)


def R_CR(m_H1, m_H2):
    """Calculate outer edge of control region discriminant.

    R_CR = sqrt(
        (m_H1 - 1.05 * 124 GeV)^2 + (m_H2 - 1.05 * 117 GeV)^2
    )
    """

    return np.sqrt((m_H1 - 1.05 * 124) ** 2 + (m_H2 - 1.05 * 117) ** 2)


def X_Wt(m_jj, m_jjb):
    """Calculate top-veto discriminant. Where m_jj is the mass of the W candidate
    and m_jjb is the mass of the top candidate.

    X_Wt = sqrt(
        ((m_jj - 80.4 GeV) / 0.1 * m_jj)^2 + ((m_jjb - 172.5 GeV) / 0.1 * m_jjb)^2
    )
    """

    m_W = 80.4 * 1e3
    m_t = 172.5 * 1e3

    return np.sqrt(
        ((m_jj - m_W) / (0.1 * m_jj)) ** 2 + ((m_jjb - m_t) / (0.1 * m_jjb)) ** 2
    )


def select_X_Wt_events(events, discriminant_cut=1.5):
    """Selects events that pass the top-veto selection.

    Events are vetoed if the minimum X_Wt over all combinations is less than 1.5

    Returns:
        Events that pass the top-veto selection
    """

    leading_four_bjets, remaining_jets = events
    if len(remaining_jets) == 0:
        return leading_four_bjets, remaining_jets, []

    leading_four_bjets_p4 = p4.zip(
        {
            "pt": leading_four_bjets.jet_pt,
            "eta": leading_four_bjets.jet_eta,
            "phi": leading_four_bjets.jet_phi,
            "mass": leading_four_bjets.jet_m,
        }
    )
    remaining_jets_p4 = p4.zip(
        {
            "pt": remaining_jets.jet_pt,
            "eta": remaining_jets.jet_eta,
            "phi": remaining_jets.jet_phi,
            "mass": remaining_jets.jet_m,
        }
    )

    W_candidate_indices = ak.argcombinations(remaining_jets_p4, 2, axis=1)
    W_candidate_firsts, W_candidate_seconds = ak.unzip(W_candidate_indices)
    W_candidates = (
        remaining_jets_p4[W_candidate_firsts] + remaining_jets_p4[W_candidate_seconds]
    )
    t_candidate_indices = ak.argcartesian([leading_four_bjets_p4, W_candidates], axis=1)
    t_candidate_bjet_indices, t_candidate_W_indices = ak.unzip(t_candidate_indices)
    t_candidates = (
        leading_four_bjets_p4[t_candidate_bjet_indices]
        + W_candidates[t_candidate_W_indices]
    )
    X_Wt_discriminant = X_Wt(
        W_candidates[t_candidate_W_indices].m,
        t_candidates.m,
    )
    keep = ak.min(X_Wt_discriminant, axis=1) > discriminant_cut
    return leading_four_bjets[keep], remaining_jets[keep], X_Wt_discriminant


def hh_reconstruct_mindeltar(events):
    """Pair 4 leading b-jets by minimum deltaR.
    Assumes b-jets are already sorted by jet pT and b-jets >= 4.

    Returns:
        2 p4 objects"""

    jets = p4.zip(
        {
            "pt": events.jet_pt,
            "eta": events.jet_eta,
            "phi": events.jet_phi,
            "m": events.jet_m,
        }
    )
    if len(jets) == 0:
        return jets, jets
    njets = 4
    four_leading_bjets = jets[:, :njets]
    # Calculate h1
    h1_j1_one_hot = np.eye(njets)[ak.local_index(four_leading_bjets, axis=1)[:, 0]]
    h1_j1_mask = h1_j1_one_hot == 1
    h1_leading_jet = four_leading_bjets[h1_j1_mask]
    deltar = four_leading_bjets[:, 1:].deltaR(h1_leading_jet)
    # Add 1 to offset the first jet
    h1_subleading_jet_index = ak.argmin(deltar, axis=1, keepdims=True) + 1
    h1_j2_one_hot = np.eye(njets)[ak.flatten(h1_subleading_jet_index)]
    h1_j2_mask = h1_j2_one_hot == 1
    h1_subleading_jet = four_leading_bjets[h1_j2_mask]
    h1 = h1_leading_jet + h1_subleading_jet
    # Calculate h2
    h2_j12_indices_mask = ~h1_j1_mask & ~h1_j2_mask
    h2_j12 = four_leading_bjets[h2_j12_indices_mask.tolist()]
    h2_j1, h2_j2 = h2_j12[:, 0], h2_j12[:, 1]
    h2 = h2_j1 + h2_j2
    return h1, h2


def select_hh_events(h1, h2, deltaeta_cut=1.5, mass_discriminant_cut=1.6):
    """Selects events that pass the hh selection.

    Returns:
        Events that pass the hh selection
    """

    delta_eta = abs(h1.eta - h2.eta)
    keep = delta_eta < deltaeta_cut
    return h1[keep], h2[keep]
