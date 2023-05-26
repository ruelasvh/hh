import awkward as ak
import numpy as np
import vector as p4
from src.nonresonantresolved.utils import inv_GeV


def select_n_jets_events(
    events,
    pt_cut=40_000,
    eta_cut=2.5,
    njets_cut=4,
):
    """Selects events by applying the cuts specified in the arguments.
    The jet pT and eta column names should be the first and second
    items in jet_vars."""

    jet_pt = events.jet_pt
    jet_eta = events.jet_eta
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

    return events[valid_events]


def select_n_bjets(events, nbjets_cut):
    """Selects events by applying the cuts specified in the arguments.
    The jet pT and eta column names should be the first and second
    items in jet_vars. Assumes jets are already sorted by pT."""

    n_plus_btags = ak.sum(events.jet_btag_default, axis=1) >= nbjets_cut
    events_with_n_plus_btags = events[n_plus_btags]
    return events_with_n_plus_btags


def select_hc_jets(events, jet_vars=None, nbjets_cut=2):
    """Selects events by applying the cuts specified in the arguments.
    The HH system is reconstructed from two Higgs candidates, which are
    themselves reconstructed from two jets each (four Higgs candidate jets in total).

    b-jets are selected first. If the event is a 4b event, the leading four
    in pT are selected. If it is a 2b event, the remaining places are filled
    by non-b-tagged jets, which are sorted in pT and the two leading jets taken

    Returns the 4 Higgs candidate jets in each event.
    """

    # btag_decisions is a boolean array of shape (n_events, n_jets)
    btag_decisions = (
        events.jet_btag_default
    )  # [[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1, 1], ...]
    jet_indices = ak.local_index(
        btag_decisions
    )  # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3, 4], ...]
    btag_decisions_mask = btag_decisions == 1
    # Higgs candidates, we want 4 jets per event
    hc_bjets_indices = jet_indices[btag_decisions_mask][
        :, :4
    ]  # [[0, 1, 2, 3], [1, 2], [2, 3, 4], ...]
    # get number of non-bjets to choose if bjets are not enough
    hc_non_bjets_to_choose_counts = 4 - ak.count(
        hc_bjets_indices, axis=1
    )  # [0, 2, 1, ...]
    hc_non_bjets_to_choose_indices = ak.Array(
        [list(range(n)) for n in hc_non_bjets_to_choose_counts]
    )
    # create array of nested arrays that creates ranges of the members of n_non_bjets_to_choose with awkward
    non_bjets_indices = jet_indices[~btag_decisions_mask]  # [[], [0, 3], [0], ...]
    # select number jets from non_bjets_indices using hc_non_bjets_to_choose_indices
    hc_non_bjets_indices = non_bjets_indices[hc_non_bjets_to_choose_indices]
    hc_indices = ak.concatenate([hc_bjets_indices, hc_non_bjets_indices], axis=1)

    return events[jet_vars][hc_indices], hc_indices


def sort_jets_by_pt(events, jet_vars=None):
    """Sorts events by jet pT. The jet pT column name should be the first
    item in jet_vars."""

    if jet_vars is None:
        return events
    pt_sorted_jets_idx = ak.argsort(events.jet_pt, ascending=False)
    pt_sorted_jets = events[
        jet_vars,
        pt_sorted_jets_idx,
    ]
    for jet_var in jet_vars:
        events = ak.with_field(events, pt_sorted_jets[jet_var], jet_var)
    return events


def X_HH(m_H1, m_H2):
    """Calculate signal region discriminat.

    X_HH = sqrt(
        ((m_H1 - 124 GeV) / 0.1 * m_H1)^2 + ((m_H2 - 117 GeV) / 0.1 * m_H2)^2
    )
    """

    m_H1_center = 124
    m_H2_center = 117

    # first_term = np.zeros_like(m_H1)
    # np.divide(m_H1 - m_H1_center, 0.1 * m_H1, out=first_term, where=(m_H1 != 0))
    # second_term = np.zeros_like(m_H2)
    # np.divide(m_H2 - m_H2_center, 0.1 * m_H2, out=second_term, where=(m_H2 != 0))
    # return np.sqrt(first_term**2 + second_term**2)

    return np.sqrt(
        ((m_H1 - m_H1_center) / (0.1 * m_H1)) ** 2
        + ((m_H2 - m_H2_center) / (0.1 * m_H2)) ** 2
    )


def R_CR(m_H1, m_H2):
    """Calculate outer edge of control region discriminant.

    R_CR = sqrt(
        (m_H1 - 1.05 * 124 GeV)^2 + (m_H2 - 1.05 * 117 GeV)^2
    )
    """

    m_H1_center = 124
    m_H2_center = 117

    return np.sqrt((m_H1 - 1.05 * m_H1_center) ** 2 + (m_H2 - 1.05 * m_H2_center) ** 2)


def X_Wt(m_jj, m_jjb):
    """Calculate top-veto discriminant. Where m_jj is the mass of the W candidate
    and m_jjb is the mass of the top candidate.

    X_Wt = sqrt(
        ((m_jj - 80.4 GeV) / 0.1 * m_jj)^2 + ((m_jjb - 172.5 GeV) / 0.1 * m_jjb)^2
    )
    """

    m_W = 80.4
    m_t = 172.5

    return np.sqrt(
        ((m_jj - m_W) / (0.1 * m_jj)) ** 2 + ((m_jjb - m_t) / (0.1 * m_jjb)) ** 2
    )


def select_X_Wt_eventsv1(events, discriminant_cut=1.5):
    """Selects events that pass the top-veto selection.
    Events are vetoed if the minimum X_Wt over all combinations is less than 1.5
    Returns:
        Events that pass the top-veto selection
    """

    hc_jets, remaining_jets = events
    hc_jets_p4 = p4.zip(
        {
            "pt": hc_jets.jet_pt,
            "eta": hc_jets.jet_eta,
            "phi": hc_jets.jet_phi,
            "mass": hc_jets.jet_m,
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
    # for one event with hc_jets with 4 jets, remaining_jets with 2 jets
    W_candidate_indices = ak.argcombinations(remaining_jets_p4, 2, axis=1)  # [[(0, 1)]]
    hc_jets_indices = ak.local_index(hc_jets_p4, axis=1)  # [[0, 1, 2, 3]]
    t_candidate_indices = ak.cartesian(
        [hc_jets_indices, W_candidate_indices], axis=1
    )  # [[(0, (0, 1)), (1, (0, 1)), (2, (0, ...)), (3, (0, 1))]]
    t_candidate_hc_jet_indices, t_candidate_W_indices = ak.unzip(
        t_candidate_indices
    )  # [[0, 1, 2, 3]], [[(0, 1), (0, 1), (0, 1), (0, 1)]]
    W_candidate_jet1, W_candidate_jet2 = ak.unzip(
        t_candidate_W_indices
    )  # [[0, 0, 0, 0]], [[1, 1, 1, 1]]
    W_candidates_p4 = (
        remaining_jets_p4[W_candidate_jet1] + remaining_jets_p4[W_candidate_jet2]
    )
    t_candidates_p4 = (
        remaining_jets_p4[W_candidate_jet1]
        + remaining_jets_p4[W_candidate_jet2]
        + hc_jets_p4[t_candidate_hc_jet_indices]
    )
    X_Wt_discriminant = X_Wt(
        W_candidates_p4.m * inv_GeV,
        t_candidates_p4.m * inv_GeV,
    )
    X_Wt_discriminant = ak.min(X_Wt_discriminant, axis=1)
    # For events with no valid combinations because there are exactly 4 bjets in the event,
    # fill with 9999.0
    X_Wt_discriminant = ak.fill_none(X_Wt_discriminant, 9999.0)
    keep = X_Wt_discriminant > discriminant_cut
    return hc_jets[keep], remaining_jets[keep], X_Wt_discriminant, keep


def select_X_Wt_events_nicole(events, idx, discriminant_cut=1.5):
    """
    Calculate the X_wt variable for the event.

    Note: consistent w/ RR, only considers the HC jets (the first 4 jets in idx) for the
    b-tagged b-jets in the top-candidate.

    Input:
    - jarr: awkward array of jet features
    - ps: awkward array of 4-vectors for the jets
    - idx: ordering for the jets (first 4 jets are the HC jets in the eveny)

    Output:
    - Xwt: The Xwt minimized over all of the valid 3-jet combinations
    """

    ps = p4.zip(
        {
            "pt": events.jet_pt,
            "eta": events.jet_eta,
            "phi": events.jet_phi,
            "mass": events.jet_m,
        }
    )

    btag = events.jet_btag_DL1dv01_77[idx]

    bjet = ps[:, :4][btag[:, :4] == 1]
    bidx = ak.Array([range(nb) for nb in ak.num(bjet)])
    # bidx = idx

    # # add a dim across the last entry
    bjet = bjet[:, :, np.newaxis]
    bidx = bidx[:, :, np.newaxis]

    w_jet_pairs = ak.combinations(ps, 2)
    w_idx_pairs = ak.argcombinations(ps, 2)

    wjet1, wjet2 = ak.unzip(w_jet_pairs[:, np.newaxis, :])
    widx1, widx2 = ak.unzip(w_idx_pairs[:, np.newaxis, :])

    # Get the corresponding combinations

    WC = wjet1 + wjet2
    tC = bjet + WC

    Xwt_combs = X_Wt(WC.mass * inv_GeV, tC.mass * inv_GeV)

    # Set as "invalid" the entries where the b-jet overlaps w/ one of the w-jets
    Xwt_mask = ak.where((bidx == widx1) | (bidx == widx2), np.inf, Xwt_combs)

    # Minimize over the possible combinations
    min_discrim = ak.min(ak.min(Xwt_mask, axis=-1), axis=-1)
    keep = min_discrim > discriminant_cut
    return events[keep], min_discrim, keep


def select_X_Wt_events(events, hc_jet_indices, discriminant_cut=1.5):
    """Selects events that pass the top-veto selection.
    Events are vetoed if the minimum X_Wt over all combinations is less than 1.5
    Returns:
        Events that pass the top-veto selection
    """

    jet_p4 = p4.zip(
        {
            "pt": events.jet_pt,
            "eta": events.jet_eta,
            "phi": events.jet_phi,
            "mass": events.jet_m,
        }
    )
    # for one event with hc_jet_indices with 4 jets, remaining_jets with 2 jets
    # jet_indices -> [[0, 1, 2, 3, 4, 5], ...]
    # hc_jet_indices -> [[0, 2, 3, 5], ...]
    jet_pair_combinations_indices = ak.argcombinations(
        jet_p4, 2, axis=1
    )  # [[(0, 1), (0, 2), (0, 3), (0, 4), (...), ..., (2, 5), (3, 4), (3, 5), (4, 5)], ...]
    top_candidate_indices = ak.cartesian(
        [hc_jet_indices, jet_pair_combinations_indices], axis=1
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
    W_candidates_p4 = jet_p4[top_jet1_indices] + jet_p4[top_jet2_indices]
    top_candidates_p4 = (
        jet_p4[top_jet1_indices] + jet_p4[top_jet2_indices] + jet_p4[top_jet3_indices]
    )
    # calculate X_Wt discriminant
    X_Wt_discriminant = X_Wt(
        W_candidates_p4.m * inv_GeV,
        top_candidates_p4.m * inv_GeV,
    )
    # select only the minimum X_Wt for each event
    X_Wt_discriminant = ak.min(X_Wt_discriminant, axis=1)
    passed_top_veto_mask = X_Wt_discriminant > discriminant_cut
    return events[passed_top_veto_mask], X_Wt_discriminant, passed_top_veto_mask


def reconstruct_hh_mindeltar(events, signal=False):
    """Pair 4 leading b-jets by minimum deltaR.
    Assumes b-jets are already sorted by jet pT and b-jets >= 4.

    Returns:
        2 p4 objects"""

    jet_p4 = p4.zip(
        {
            "pt": events.jet_pt,
            "eta": events.jet_eta,
            "phi": events.jet_phi,
            "mass": events.jet_m,
        }
    )
    if len(jet_p4) == 0:
        return jet_p4, jet_p4
    jet_indices = ak.local_index(jet_p4, axis=1)
    leading_jets = jet_indices[:, :1]
    subleading_jets = jet_indices[:, 1:]
    jet_pairs = ak.cartesian([leading_jets, subleading_jets], axis=1)
    leading_jet_indices, subleading_jet_indices = ak.unzip(jet_pairs)
    deltar = jet_p4[leading_jet_indices].deltaR(jet_p4[subleading_jet_indices])
    min_deltar = ak.argmin(deltar, axis=1, keepdims=True)
    leading_h_jet_indices = jet_pairs[min_deltar]
    leading_h_jet1_indices, leading_h_jet2_indices = ak.unzip(leading_h_jet_indices)
    leading_h_p4 = ak.flatten(
        jet_p4[leading_h_jet1_indices] + jet_p4[leading_h_jet2_indices]
    )
    # create one-hot mask for h leading jets
    leading_h_one_hot = ak.from_numpy(np.eye(4))[ak.flatten(leading_h_jet1_indices)]
    leading_h_mask = leading_h_one_hot == 1
    # create one-hot mask for h subleading jets
    subleading_h_one_hot = ak.from_numpy(np.eye(4))[ak.flatten(leading_h_jet2_indices)]
    subleading_h_mask = subleading_h_one_hot == 1
    # create one-hot mask for h2 jets
    subleading_h_mask = ~(leading_h_mask | subleading_h_mask)
    subleading_h_jet_indices = jet_indices[subleading_h_mask]
    subleading_h_p4 = (
        jet_p4[subleading_h_jet_indices][:, 0] + jet_p4[subleading_h_jet_indices][:, 1]
    )
    return (
        leading_h_p4,
        subleading_h_p4,
        leading_h_jet_indices,
        subleading_h_jet_indices,
    )


def select_hh_events(h1, h2, deltaeta_cut=None, mass_discriminant_cut=None):
    """Selects events that pass the hh selection.

    Returns:
        Events that pass the hh selection
    """

    keep = np.ones_like(h1.m, dtype=bool)
    hh_var = np.array([])
    if deltaeta_cut is not None:
        hh_var = np.abs(ak.to_numpy(h1.eta) - ak.to_numpy(h2.eta))
        keep = keep & (hh_var < deltaeta_cut)
    if mass_discriminant_cut is not None:
        if type(mass_discriminant_cut) is float:
            hh_var = X_HH(ak.to_numpy(h1.m) * inv_GeV, ak.to_numpy(h2.m) * inv_GeV)
            keep = keep & (hh_var < mass_discriminant_cut)
        if type(mass_discriminant_cut) is tuple:
            hh_low = X_HH(ak.to_numpy(h1.m) * inv_GeV, ak.to_numpy(h2.m) * inv_GeV)
            hh_high = R_CR(ak.to_numpy(h1.m) * inv_GeV, ak.to_numpy(h2.m) * inv_GeV)
            keep = (
                keep
                & (hh_low >= mass_discriminant_cut[0])
                & (hh_high <= mass_discriminant_cut[1])
            )
    return h1[keep], h2[keep], hh_var, keep


def select_correct_hh_pair_events(
    events, leading_h_jet_indices, subleading_h_jet_indices, signal=False
):
    if not signal:
        return events, ak.Array([])
    jets_truth_matched_to_hh = events["jet_truth_H_parents"]
    # leading_h_jet1_indices, leading_h_jet2_indices = ak.unzip(leading_h_jet_indices)
    leading_h_jet1_indices = leading_h_jet_indices[
        :, "0"
    ]  # "0" because it's a record array
    leading_h_jet2_indices = leading_h_jet_indices[
        :, "1"
    ]  # "1" because it's a record array
    leading_h_jet1_truth_matched = jets_truth_matched_to_hh[leading_h_jet1_indices]
    leading_h_jet2_truth_matched = jets_truth_matched_to_hh[leading_h_jet2_indices]
    leading_h_jets_have_same_parent_mask = (
        leading_h_jet1_truth_matched == leading_h_jet2_truth_matched
    )
    subleading_h_truth_matched = jets_truth_matched_to_hh[subleading_h_jet_indices]
    subleading_h_jet1_truth_matched = subleading_h_truth_matched[:, 0]
    subleading_h_jet2_truth_matched = subleading_h_truth_matched[:, 1]
    subleading_h_jets_have_same_parent_mask = (
        subleading_h_jet1_truth_matched == subleading_h_jet2_truth_matched
    )
    correct_hh_pairs_mask = (
        leading_h_jets_have_same_parent_mask & subleading_h_jets_have_same_parent_mask
    )
    correct_hh_pairs_events = events[correct_hh_pairs_mask]
    return (
        correct_hh_pairs_events,
        correct_hh_pairs_mask,
    )
