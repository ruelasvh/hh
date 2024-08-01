import itertools
import numpy as np
import vector as p4
import awkward as ak
from hh.shared.utils import (
    get_op,
    format_btagger_model_name,
    get_trigs_logical_op,
    kin_labels,
    GeV,
)
from hh.shared.selection import X_Wt, get_W_t_p4


def select_events_passing_triggers(
    events,
    op: str = None,
    triggers: list = None,
):
    triggers = triggers or list(filter(lambda x: "trig_" in x, events.fields))
    passed_trigs_mask = np.ones(len(events), dtype=bool)
    if op:
        passed_trigs_mask = get_trigs_logical_op(events, triggers, op)
        passed_trigs_mask = ak.fill_none(passed_trigs_mask, False)
    return passed_trigs_mask


def select_n_jets_events(
    jets,
    selection,
    do_jvt=True,
):
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
        # apply JVT cut only on jets that have pt < 60 GeV
        valid_jets_mask = ak.mask(valid_jets_mask, jets.pt < 60_000)
        valid_jets_mask = valid_jets_mask & (jets.jvttag == 1)
        ak.fill_none(valid_jets_mask, True, axis=1)
    if njets_sel:
        valid_events_mask = get_op(njets_sel["operator"])(
            ak.sum(valid_jets_mask, axis=1), njets_sel["value"]
        )
        valid_jets_mask = ak.mask(valid_jets_mask, valid_events_mask)
    return valid_jets_mask


def select_n_bjets_events(
    jets,
    where,
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
    condition_mask = get_op(n_btags_operator)(ak.sum(where, axis=1), n_btags_value)
    valid_events_mask = ~ak.is_none(jets, axis=0)
    valid_jets_mask = ak.mask(jets, valid_events_mask & condition_mask)
    return valid_jets_mask


def select_hh_jet_candidates(jets, valid_jets_mask, n_jets=4):
    """Selects the 4 Higgs candidate jets in each event.

    Returns:
        The 4 Higgs candidate jets mask and non-Higgs candidates in each event
    """
    jet_idx = ak.local_index(jets)
    valid_jets = jets[valid_jets_mask]
    valid_jets_idx = jet_idx[valid_jets_mask]
    pt_sort = ak.argsort(valid_jets.pt, axis=1, ascending=False)
    valid_jets_pt_sorted = valid_jets[pt_sort]
    valid_jets_pt_sorted_idx = valid_jets_idx[pt_sort]
    btag_sort = ak.argsort(valid_jets_pt_sorted.btag, axis=1, ascending=False)
    valid_jets_btag_sorted_idx = valid_jets_pt_sorted_idx[btag_sort]
    hh_jet_idx = valid_jets_btag_sorted_idx[:, :n_jets]
    non_hh_jet_idx = valid_jets_btag_sorted_idx[:, n_jets:]
    return hh_jet_idx, non_hh_jet_idx


# def reconstruct_hh_jet_pairs(
#     jets_p4, hh_jet_idx, loss, optimizer=np.argmin, n_jets=4, n_pairs=2
# ):
#     hh_jets = jets_p4[hh_jet_idx]
#     pairings = list(itertools.combinations(range(n_jets), n_pairs))
#     loss_values = np.transpose(
#         ak.Array(loss(hh_jets[:, i], hh_jets[:, j]) for i, j in pairings)
#     )
#     loss_values = ak.mask(loss_values, ~ak.is_none(hh_jets))
#     pt_values = np.transpose(
#         ak.Array((hh_jets[:, i] + hh_jets[:, j]).pt for i, j in pairings)
#     )
#     optimized_loss_idx = optimizer(loss_values, pt_values, axis=1, keepdims=True)
#     optimized_loss_idx = ak.mask(optimized_loss_idx, ~ak.is_none(hh_jets))
#     pairings = ak.argcombinations(hh_jets, n_pairs)
#     h1_jet_idx = ak.concatenate(ak.unzip(pairings[optimized_loss_idx]), axis=1)
#     h2_jet_idx = ak.concatenate(ak.unzip(pairings[~optimized_loss_idx]), axis=1)
#     return hh_jet_idx[h1_jet_idx], hh_jet_idx[h2_jet_idx]


def reconstruct_hh_jet_pairs(
    jets_p4, hh_jet_idx, loss, optimizer=np.argmin, n_jets=4, n_pairs=2
):

    def pairing_lead(js, pair):
        """
        Return the loss of the leading HC
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

    hc_jets = jets_p4[hh_jet_idx]
    valid_event_mask = ~ak.is_none(hc_jets, axis=0)
    chosen_pair = optimizer(
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


def select_top_veto_events(jets, hh_jet_idx, non_hh_jet_idx, selection):
    """Selects events that pass the top-veto selection.
    Events are vetoed if the minimum X_Wt over all combinations is less than selection.
    Returns:
        Events that pass the top-veto selection
    """
    # reconstruct W and top candidates
    W_candidates_p4, top_candidates_p4 = get_W_t_p4(
        jets,
        hh_jet_idx,
        non_hh_jet_idx,
    )
    # calculate X_Wt discriminant
    X_Wt_discriminant = X_Wt(
        W_candidates_p4.mass * GeV,
        top_candidates_p4.mass * GeV,
    )
    # select only the minimum X_Wt for each event
    X_Wt_discriminant_min = ak.min(X_Wt_discriminant, axis=1)
    passed_top_veto_mask = get_op(selection["operator"])(
        X_Wt_discriminant_min, selection["value"]
    )
    passed_top_veto_mask = ak.fill_none(passed_top_veto_mask, False)
    return passed_top_veto_mask, X_Wt_discriminant_min


def select_discrim_events(discrim, selection=None):
    """Selects events that pass the selection for the given discriminant.

    Returns:
        Events that pass the discriminant selection
    """
    keep = np.ones(len(discrim[0]), dtype=bool)
    if selection is not None:
        if all([k in selection for k in ["operator", "value"]]):
            keep = keep & get_op(selection["operator"])(discrim[0], selection["value"])
        else:
            for i, sel in enumerate(selection.values()):
                if isinstance(sel, dict):
                    keep = keep & select_discrim_events((discrim[i],), sel)
    keep = ak.fill_none(keep, False).to_numpy()
    return keep


def select_correct_hh_pair_events(h1_jets_idx, h2_jets_idx, truth_jet_H_parent_mask):
    h1_truth_matched = truth_jet_H_parent_mask[h1_jets_idx]
    h1_jet1_truth_matched = h1_truth_matched[:, 0]
    h1_jet2_truth_matched = h1_truth_matched[:, 1]
    h1_jets_have_same_parent_mask = h1_jet1_truth_matched == h1_jet2_truth_matched
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


def select_truth_matched_jets(
    truth_matched_jets_mask, valid_jets_mask, n_truth_matched=4
):
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
    keep_event_mask = ak.sum(valid_truth_matched_jets, axis=1) >= n_truth_matched
    valid_truth_matched_jet_mask = ak.mask(valid_truth_matched_jets, keep_event_mask)
    return valid_truth_matched_jet_mask


def calculate_scale_factors(events):
    """Calculates the scale factors for each event.

    Returns:
        The scale factors for each event
    """
    # jsfs = t.arrays('sf', aliases=jalias, cut=f'(pt > {pT_min}) & (abs(eta) < {eta_max}) & {jvtCut}')
    # mc_sf = ak.prod(jsfs.sf[:,:,0],axis=-1).to_numpy()
    sf = np.ones(len(events))
    return sf
