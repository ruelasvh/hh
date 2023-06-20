import awkward as ak
import numpy as np
from src.nonresonantresolved.utils import inv_GeV, get_all_trigs_or
from src.shared.utils import get_op


def select_events_passing_all_triggers_OR(events, triggers):
    all_trigs_or_decicions = get_all_trigs_or(events, triggers)
    return ak.to_numpy(all_trigs_or_decicions)


def select_n_jets_events(events, selection):
    """Selects events by applying the cuts specified in the selection."""

    pt_cut = selection["pt"]
    eta_cut = selection["eta"]
    njets_cut = selection["count"]
    jet_pt_sorted_idx = events.jet_pt_sorted_idx.ak.array
    breakpoint()
    jet_pt = events.jet_pt.ak.array[jet_pt_sorted_idx]
    jet_eta = events.jet_eta.ak.array[jet_pt_sorted_idx]
    valid_central_jets_mask = ak.num(
        (get_op(pt_cut["operator"])(jet_pt, pt_cut["value"]))
        & get_op(eta_cut["operator"])(np.abs(jet_eta), eta_cut["value"])
    )
    valid_n_central_jets_mask = get_op(njets_cut["operator"])(
        valid_central_jets_mask, njets_cut["value"]
    )

    return ak.to_numpy(valid_n_central_jets_mask)


def select_n_bjets_events(
    events,
    selection,
):
    """Selects events by applying the cuts specified in the selection."""

    n_btags_cut = selection["count"]
    n_btags = events.btag_num.values
    valid_n_bjets_mask = get_op(n_btags_cut["operator"])(n_btags, n_btags_cut["value"])
    return ak.to_numpy(valid_n_bjets_mask)


def select_hc_jets(events, nbjets_cut=4):
    """Selects events by applying the cuts specified in the arguments.
    The HH system is reconstructed from two Higgs candidates, which are
    themselves reconstructed from two jets each (four Higgs candidate jets in total).

    b-jets are selected first. If the event is a 4b event, the leading four
    in pT are selected. If it is a 2b event, the remaining places are filled
    by non-b-tagged jets, which are sorted in pT and the two leading jets taken

    Returns the 4 Higgs candidate jets in each event.
    """

    # btag_decisions is a boolean array of shape (n_events, n_jets)
    btag_decisions = events.jet_btag_default.ak.array[events.jet_pt_sorted_idx.ak.array]
    valid_event_mask = events.valid_event.values
    jet_indices = ak.mask(ak.local_index(btag_decisions), valid_event_mask)
    btag_decisions_mask = btag_decisions == 1
    # Higgs candidates, we want 4 jets per event
    hc_bjets_indices = jet_indices[btag_decisions_mask][:, :nbjets_cut]
    # get number of non-bjets to choose if bjets are not enough
    requested_hc_non_bjets_counts = nbjets_cut - ak.count(hc_bjets_indices, axis=1)
    # create array of nested arrays that creates ranges of the members of n_non_bjets_to_choose with awkward
    non_bjets_indices = jet_indices[~btag_decisions_mask]
    hc_non_bjets_to_choose_indices = ak.Array(
        [
            list(range(requested)) if requested else requested
            for requested in requested_hc_non_bjets_counts
        ]
    )
    # select number jets from non_bjets_indices using hc_non_bjets_to_choose_indices
    hc_non_bjets_indices = non_bjets_indices[hc_non_bjets_to_choose_indices]
    hc_indices = ak.concatenate([hc_bjets_indices, hc_non_bjets_indices], axis=1)
    return hc_indices


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


def select_X_Wt_events_nicole(events, selection):
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

    ps = events.jet_p4.ak.array
    idx = events.hc_jets_idx.ak.array
    btag = events.jet_btag_default.ak.array[idx]

    bjet = ps[:, :4][btag[:, :4] == 1]
    bidx = ak.Array([range(nb) for nb in ak.num(bjet)])

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
    keep = get_op(selection["operator"])(min_discrim, selection["value"])
    return ak.fill_none(keep, False), ak.fill_none(min_discrim, np.nan)


def select_X_Wt_events(events, selection):
    """Selects events that pass the top-veto selection.
    Events are vetoed if the minimum X_Wt over all combinations is less than selection.
    Returns:
        Events that pass the top-veto selection
    """

    jet_p4 = events.jet_p4.ak.array
    hc_jet_indices = events.hc_jets_idx.ak.array
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
        W_candidates_p4.mass * inv_GeV,
        top_candidates_p4.mass * inv_GeV,
    )
    # select only the minimum X_Wt for each event
    X_Wt_discriminant_min = ak.min(X_Wt_discriminant, axis=1)
    X_Wt_discriminant_min = ak.fill_none(X_Wt_discriminant_min, np.nan)
    passed_top_veto_mask = get_op(selection["operator"])(
        X_Wt_discriminant_min, selection["value"]
    )
    passed_top_veto_mask = ak.fill_none(passed_top_veto_mask, False)
    return ak.to_numpy(passed_top_veto_mask), ak.to_numpy(X_Wt_discriminant_min)


def reconstruct_hh_mindeltar(events):
    """Pair 4 leading b-jets by minimum deltaR.
    Assumes jets are already sorted by jet pT and jets == 4 per event.

    Returns:
        leading and subleading b-jet indices"""

    # get the higgs candidate jets
    hc_jet_idx = events.hc_jets_idx.ak.array
    jet_p4 = events.jet_p4.ak.array[hc_jet_idx]
    jet_indices = ak.local_index(jet_p4, axis=1)
    leading_jets = jet_indices[:, :1]
    subleading_jets = jet_indices[:, 1:]
    jet_pairs = ak.cartesian([leading_jets, subleading_jets], axis=1)
    leading_jet_indices, subleading_jet_indices = ak.unzip(jet_pairs)
    deltar = jet_p4[leading_jet_indices].deltaR(jet_p4[subleading_jet_indices])
    min_deltar = ak.argmin(deltar, axis=1, keepdims=True)
    leading_h_jet_indices = jet_pairs[min_deltar]
    leading_h_jet1_indices, leading_h_jet2_indices = ak.unzip(leading_h_jet_indices)
    # create one-hot mask for h leading jets
    leading_h_jet1_one_hot = ak.from_numpy(np.eye(4))[
        ak.flatten(leading_h_jet1_indices)
    ]
    leading_h_jet1_mask = leading_h_jet1_one_hot == 1
    # create one-hot mask for h subleading jets
    leading_h_jet2_one_hot = ak.from_numpy(np.eye(4))[
        ak.flatten(leading_h_jet2_indices)
    ]
    leading_h_jet2_mask = leading_h_jet2_one_hot == 1
    # create one-hot mask for h2 jets
    leading_h_jet_idx_mask = leading_h_jet1_mask | leading_h_jet2_mask
    subleading_h_jet_idx_mask = ~leading_h_jet_idx_mask
    return (
        jet_indices[leading_h_jet_idx_mask],
        jet_indices[subleading_h_jet_idx_mask],
    )


def select_hh_events(events, deltaeta_sel=None, mass_sel=None):
    """Selects events that pass the hh selection.

    Returns:
        Events that pass the hh selection
    """

    jet_p4 = events.jet_p4.ak.array
    h1_jets_idx = events.leading_h_jets_idx.ak.array
    h1_jet1_idx, h1_jet2_idx = (
        h1_jets_idx[:, 0, np.newaxis],
        h1_jets_idx[:, 1, np.newaxis],
    )
    h2_jets_idx = events.subleading_h_jets_idx.ak.array
    h2_jet1_idx, h2_jet2_idx = (
        h2_jets_idx[:, 0, np.newaxis],
        h2_jets_idx[:, 1, np.newaxis],
    )
    h1 = jet_p4[h1_jet1_idx] + jet_p4[h1_jet2_idx]
    h2 = jet_p4[h2_jet1_idx] + jet_p4[h2_jet2_idx]
    keep = np.ones_like(np.squeeze(h1.mass), dtype=bool)
    hh_var = np.array([])
    if deltaeta_sel is not None:
        hh_var = np.abs(np.squeeze(h1.eta) - np.squeeze(h2.eta))
        keep = keep & get_op(deltaeta_sel["operator"])(hh_var, deltaeta_sel["value"])
    if mass_sel is not None:
        if mass_sel.get("inner_boundry"):
            hh_var = X_HH(np.squeeze(h1.m) * inv_GeV, np.squeeze(h2.m) * inv_GeV)
            keep = keep & get_op(mass_sel["inner_boundry"]["operator"])(
                hh_var, mass_sel["inner_boundry"]["value"]
            )
        if mass_sel.get("outer_boundry"):
            hh_var = R_CR(np.squeeze(h1.m) * inv_GeV, np.squeeze(h2.m) * inv_GeV)
            keep = keep & get_op(mass_sel["outer_boundry"]["operator"])(
                hh_var, mass_sel["outer_boundry"]["value"]
            )
    keep = ak.fill_none(keep, False)
    hh_var = ak.fill_none(hh_var, np.nan)
    return ak.to_numpy(keep), ak.to_numpy(hh_var)


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
