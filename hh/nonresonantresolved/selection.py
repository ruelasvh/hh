import itertools
import numpy as np
import vector as p4
import awkward as ak
from hh.shared.utils import (
    get_op,
    get_trigs_bitwise_op,
    kin_labels,
    inv_GeV,
)
from hh.shared.selection import X_HH, R_CR, X_Wt


def select_events_passing_triggers(
    events,
    op: str = None,
    triggers: list = None,
):
    triggers = triggers or list(filter(lambda x: "trig_" in x, events.fields))
    passed_trigs_mask = np.ones(len(events), dtype=bool)
    if op:
        passed_trigs_mask = get_trigs_bitwise_op(events, triggers, op)
    return passed_trigs_mask


def select_n_jets_events(jets, selection, do_jvt=True):
    """Selects events by applying the cuts specified in the selection."""

    pt_sel = selection["pt"] if "pt" in selection else None
    eta_sel = selection["eta"] if "eta" in selection else None
    njets_sel = selection["count"] if "count" in selection else None
    # mask array for valid jets
    valid_jets_mask = np.ones_like(jets.pt, dtype=bool)
    if pt_sel:
        valid_jets_mask = valid_jets_mask & get_op(pt_sel["operator"])(
            jets.pt, pt_sel["value"]
        )
    if eta_sel:
        valid_jets_mask = valid_jets_mask & get_op(eta_sel["operator"])(
            np.abs(jets.eta), eta_sel["value"]
        )
    if do_jvt:
        valid_jets_mask = valid_jets_mask & jets.jvttag == 1
    if njets_sel:
        valid_events_mask = get_op(njets_sel["operator"])(
            ak.sum(valid_jets_mask, axis=1), njets_sel["value"]
        )
        valid_jets_mask = ak.mask(valid_jets_mask, valid_events_mask)
    return valid_jets_mask


def select_n_bjets_events(
    jets,
    selection,
):
    """Selects events by applying the cuts specified in the selection.

    Parameters:
        jets: A mask of the valid in each event
        selection: The selection criteria for the number of b-jets consisting of an operator and a value

    Returns:
        The mask for valid jets satisfying the selection

    """
    n_btags_operator, n_btags_value = (
        selection["count"]["operator"],
        selection["count"]["value"],
    )
    keep_events_mask = get_op(n_btags_operator)(ak.sum(jets, axis=1), n_btags_value)
    valid_jets_mask = ak.mask(jets, keep_events_mask)
    return valid_jets_mask


def select_hh_jet_candidates(jets, valid_jets_mask):
    """Selects the 4 Higgs candidate jets in each event.

    Returns:
        The 4 Higgs candidate jets mask and non-Higgs candidates in each event
    """
    jet_idx = ak.local_index(jets)
    valid_btag_jets_mask = valid_jets_mask & (jets.btag == 1)
    pt_sort = ak.argsort(jets.pt[valid_btag_jets_mask], axis=1, ascending=False)
    valid_btag_jets_idx = jet_idx[valid_btag_jets_mask][pt_sort]
    valid_no_btag_jets_mask = valid_jets_mask & (jets.btag == 0)
    pt_sort = ak.argsort(jets.pt[valid_no_btag_jets_mask], axis=1, ascending=False)
    valid_no_btag_jets_idx = jet_idx[valid_no_btag_jets_mask][pt_sort]
    valid_jets_idx = ak.concatenate(
        [valid_btag_jets_idx, valid_no_btag_jets_idx], axis=1
    )
    valid_events_mask = ~ak.is_none(valid_jets_mask, axis=0)
    hh_jet_idx = valid_jets_idx[:, :4]
    non_hh_jet_idx = ak.concatenate(
        [valid_jets_idx[:, 4:], jet_idx[~valid_jets_mask]], axis=1
    )
    hh_jet_idx = ak.mask(hh_jet_idx, valid_events_mask)
    non_hh_jet_idx = ak.mask(non_hh_jet_idx, valid_events_mask)
    return hh_jet_idx, non_hh_jet_idx


def reconstruct_hh_jet_pairs(jets, hh_jet_idx, loss, optimizer=np.argmin):
    jets = jets[hh_jet_idx]
    pt_sort = ak.argsort(jets.pt, axis=1, ascending=False)
    jets = jets[pt_sort]
    hh_jet_idx = hh_jet_idx[pt_sort]
    fourpairs = list(itertools.combinations(range(4), 2))
    loss_values = np.transpose(
        ak.Array(loss(jets[:, i], jets[:, j]) for i, j in fourpairs)
    )
    optimized_loss_idx = optimizer(loss_values, axis=1, keepdims=True)
    optimized_loss_idx = ak.mask(optimized_loss_idx, ~ak.is_none(jets, axis=0))
    fourpairs = ak.argcombinations(jets, 2)
    h1_jet_idx = ak.concatenate(ak.unzip(fourpairs[optimized_loss_idx]), axis=1)
    h2_jet_idx = ak.concatenate(ak.unzip(fourpairs[~optimized_loss_idx]), axis=1)
    return hh_jet_idx[h1_jet_idx], hh_jet_idx[h2_jet_idx]


def select_X_Wt_events(events, selection):
    """Selects events that pass the top-veto selection.
    Events are vetoed if the minimum X_Wt over all combinations is less than selection.
    Returns:
        Events that pass the top-veto selection
    """
    # reconstruct W and top candidates
    W_candidates_p4, top_candidates_p4 = get_W_t_p4(
        ak.zip({var: events[f"jet_{var}"] for var in list(kin_labels) + ["btag"]}),
        events.hh_jet_idx,
        events.non_hh_jet_idx,
    )
    # calculate X_Wt discriminant
    X_Wt_discriminant = X_Wt(
        W_candidates_p4.mass * inv_GeV,
        top_candidates_p4.mass * inv_GeV,
    )
    # select only the minimum X_Wt for each event
    X_Wt_discriminant_min = ak.min(X_Wt_discriminant, axis=1)
    passed_top_veto_mask = get_op(selection["operator"])(
        X_Wt_discriminant_min, selection["value"]
    )
    passed_top_veto_mask = ak.fill_none(passed_top_veto_mask, False)
    return passed_top_veto_mask, X_Wt_discriminant_min


def select_hh_events(events, deltaeta_sel=None, mass_sel=None):
    """Selects events that pass the hh selection.

    Returns:
        Events that pass the hh selection
    """

    jet_p4 = p4.zip(
        {var: events[f"jet_{var}"] for var in kin_labels.keys()},
    )
    h1_jets_idx = events.leading_h_jet_idx
    h1_jet1_idx, h1_jet2_idx = (
        h1_jets_idx[:, 0, np.newaxis],
        h1_jets_idx[:, 1, np.newaxis],
    )
    h2_jets_idx = events.subleading_h_jet_idx
    h2_jet1_idx, h2_jet2_idx = (
        h2_jets_idx[:, 0, np.newaxis],
        h2_jets_idx[:, 1, np.newaxis],
    )
    h1 = jet_p4[h1_jet1_idx] + jet_p4[h1_jet2_idx]
    h2 = jet_p4[h2_jet1_idx] + jet_p4[h2_jet2_idx]
    keep = np.ones(len(events), dtype=bool)
    hh_var = np.array([])
    if deltaeta_sel is not None:
        hh_var = np.abs(ak.firsts(h1.eta) - ak.firsts(h2.eta))
        keep = keep & get_op(deltaeta_sel["operator"])(hh_var, deltaeta_sel["value"])
    if mass_sel is not None:
        if "inner_boundry" in mass_sel:
            hh_var = X_HH(ak.firsts(h1.m) * inv_GeV, ak.firsts(h2.m) * inv_GeV)
            keep = keep & get_op(mass_sel["inner_boundry"]["operator"])(
                hh_var, mass_sel["inner_boundry"]["value"]
            )
        if "outer_boundry" in mass_sel:
            hh_var = R_CR(ak.firsts(h1.m) * inv_GeV, ak.firsts(h2.m) * inv_GeV)
            keep = keep & get_op(mass_sel["outer_boundry"]["operator"])(
                hh_var, mass_sel["outer_boundry"]["value"]
            )
    keep = ak.fill_none(keep, False)
    return keep, hh_var


def select_correct_hh_pair_events(h1_jets_idx, h2_jets_idx, truth_jet_H_parent_mask):
    h1_truth_matched = truth_jet_H_parent_mask[h1_jets_idx]
    h1_jet1_truth_matched = h1_truth_matched[:, 0]
    h1_jet2_truth_matched = h1_truth_matched[:, 1]
    h1_jets_have_same_parent_mask = h1_jet1_truth_matched == h1_jet2_truth_matched
    # remove extra dimension
    h2_truth_matched = truth_jet_H_parent_mask[h2_jets_idx]
    h2_jet1_truth_matched = h2_truth_matched[:, 0]
    h2_jet2_truth_matched = h2_truth_matched[:, 1]
    h2_jets_have_same_parent_mask = h2_jet1_truth_matched == h2_jet2_truth_matched
    correct_hh_pairs_mask = (
        h1_jets_have_same_parent_mask & h2_jets_have_same_parent_mask
    )
    # convert to numpy array and replace None with False
    correct_hh_pairs_mask = ak.mask(correct_hh_pairs_mask, correct_hh_pairs_mask)
    return correct_hh_pairs_mask


def select_truth_matched_jets(truth_matched_jets_mask, valid_jets_mask):
    """Selects jets that are truth-matched to the Higgs bosons.

    Jets marked with 1 or 2 are truth-matched to the Higgs bosons.
    Jets marked with 0 are not truth-matched to the Higgs bosons.
    Jets marked with 3 are truth-matched to both Higgs bosons.

    Parameters:
        truth_matched_jets_mask: The truth mask for the diHiggs jet candidates
        valid_jets_mask: The mask for valid jets

    Returns:
        The truth-matched jets mask
    """

    valid_truth_matched_jets = valid_jets_mask & truth_matched_jets_mask
    keep_event_mask = ak.sum(valid_truth_matched_jets, axis=1) > 3
    valid_truth_matched_jet_mask = ak.mask(valid_truth_matched_jets, keep_event_mask)
    return valid_truth_matched_jet_mask


def get_W_t_p4(jets, hh_jet_idx, non_hh_jet_idx):
    jet_idx = ak.concatenate([hh_jet_idx, non_hh_jet_idx], axis=1)
    jets = jets[jet_idx]
    jet_p4 = p4.zip({var: jets[var] for var in kin_labels})
    bjet_p4 = jet_p4[:, :4][jets.btag[:, :4] == 1]
    bjet_idx = ak.local_index(bjet_p4, axis=1)
    # for one event with hc_jet_indices with 4 jets, remaining_jets with 2 jets
    # jet_indices -> [[0, 1, 2, 3, 4, 5], ...]
    # hc_jet_indices -> [[0, 2, 3, 5], ...]
    jet_pair_combinations_indices = ak.argcombinations(
        jet_p4, 2, axis=1
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
    W_candidates_p4 = jet_p4[top_jet1_indices] + jet_p4[top_jet2_indices]
    top_candidates_p4 = W_candidates_p4 + jet_p4[top_jet3_indices]
    return W_candidates_p4, top_candidates_p4


def get_hh_p4(jets, leading_h_jet_idx, subleading_h_jet_idx):
    """
    Gets the 4-momentum of the HH system from the 4 Higgs candidate jets.

    Returns:
        The 4-momentum of the HH system
    """
    jet_p4 = p4.zip(
        {var: jets[var].tolist() for var in kin_labels.keys()},
    )
    h1_jets_idx = leading_h_jet_idx
    h1_jet1_idx, h1_jet2_idx = (
        h1_jets_idx[:, 0, np.newaxis],
        h1_jets_idx[:, 1, np.newaxis],
    )
    h2_jets_idx = subleading_h_jet_idx
    h2_jet1_idx, h2_jet2_idx = (
        h2_jets_idx[:, 0, np.newaxis],
        h2_jets_idx[:, 1, np.newaxis],
    )
    h1 = jet_p4[h1_jet1_idx] + jet_p4[h1_jet2_idx]
    h2 = jet_p4[h2_jet1_idx] + jet_p4[h2_jet2_idx]

    return h1[:, 0], h2[:, 0]


def calculate_scale_factors(events):
    """Calculates the scale factors for each event.

    Returns:
        The scale factors for each event
    """
    # jsfs = t.arrays('sf', aliases=jalias, cut=f'(pt > {pT_min}) & (abs(eta) < {eta_max}) & {jvtCut}')
    # mc_sf = ak.prod(jsfs.sf[:,:,0],axis=-1).to_numpy()
    sf = np.ones(len(events))
    return sf
