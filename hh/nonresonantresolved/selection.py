import awkward as ak
import numpy as np
import vector as p4
from hh.shared.utils import get_op, get_all_trigs_or, kin_labels, inv_GeV
from hh.shared.selection import X_HH, R_CR, X_Wt


def reconstruct_hh_mindeltar(jets, hc_jet_idx):
    """Pair 4 Higgs candidate jets by pT and minimum deltaR.

    Three possible pairing permutations of 4 jets into the two Higgs candidates:
    ((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))

    For each pairing option, the leading Higgs candidate is the pair with the highest pT and the distance between its two jet constituents is dR_leading(jj).

    The pairing option with the smallest dR_leading(jj) is chosen. The other two pairing options are used to construct the subleading Higgs candidate.

    Signals with harder pT Higgs tend to have more collimated jet pairs, resulting in higher pairing accuracy. This is because the harder pT Higgs is more likely to have a larger pT jet in its pair, which is more likely to be the leading jet in the event.

    Returns:
        leading and subleading b-jet indices
    """

    jet_p4 = p4.zip(
        {var: jets[f"{var}"][hc_jet_idx] for var in kin_labels.keys()},
    )
    jet_sorted_idx = ak.argsort(jet_p4.pt, axis=1, ascending=False)
    jet_p4_sorted = jet_p4[jet_sorted_idx]
    hc_jet_sorted_idx = hc_jet_idx[jet_sorted_idx]
    leading_jets = jet_sorted_idx[:, :1]
    subleading_jets = jet_sorted_idx[:, 1:]
    jet_pairs = ak.cartesian([leading_jets, subleading_jets], axis=1)
    leading_jet_idx, subleading_jet_idx = ak.unzip(jet_pairs)
    deltar = jet_p4_sorted[leading_jet_idx].deltaR(jet_p4_sorted[subleading_jet_idx])
    min_deltar = ak.argmin(deltar, axis=1, keepdims=True)
    leading_h_jet_indices = jet_pairs[min_deltar]
    leading_h_jet1_indices, leading_h_jet2_indices = ak.unzip(leading_h_jet_indices)
    # do this to avoid ignoring invalid events when creating one-hot mask
    leading_h_jet1_indices = ak.fill_none(leading_h_jet1_indices, [None], axis=0)
    leading_h_jet2_indices = ak.fill_none(leading_h_jet2_indices, [None], axis=0)
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
        hc_jet_sorted_idx[leading_h_jet_idx_mask],
        hc_jet_sorted_idx[subleading_h_jet_idx_mask],
    )


def select_events_passing_all_triggers_OR(events, triggers: list = None):
    triggers = triggers or list(filter(lambda x: "trig_passed_" in x, events.fields))
    passed_all_trigs_OR_mask = get_all_trigs_or(events, triggers)
    return passed_all_trigs_OR_mask


def select_n_jets_events(jets, selection, do_jvt=True):
    """Selects events by applying the cuts specified in the selection."""

    pt_cut = selection["pt"] if "pt" in selection else None
    eta_cut = selection["eta"] if "eta" in selection else None
    njets_cut = selection["count"] if "count" in selection else None
    # mask array for valid jets
    valid_jets_mask = np.ones_like(jets.pt, dtype=bool)
    if pt_cut:
        valid_jets_mask = valid_jets_mask & get_op(pt_cut["operator"])(
            jets.pt, pt_cut["value"]
        )
    if eta_cut:
        valid_jets_mask = valid_jets_mask & get_op(eta_cut["operator"])(
            np.abs(jets.eta), eta_cut["value"]
        )
    if do_jvt:
        jvttag_mask = jets.jvttag == 1
        valid_jets_mask = valid_jets_mask & jvttag_mask
    # mask array for valid events
    valid_events_mask = np.ones(len(jets.pt), dtype=bool)
    if njets_cut:
        n_jets = ak.sum(valid_jets_mask, axis=1)
        valid_events_mask = valid_events_mask & get_op(njets_cut["operator"])(
            n_jets, njets_cut["value"]
        )
    # jets mask for valid events (mantains original size of events array)
    valid_n_jets_mask = ak.where(valid_events_mask, valid_jets_mask, False)
    return valid_n_jets_mask, valid_events_mask


def select_n_bjets_events(
    jets,
    selection,
):
    """Selects events by applying the cuts specified in the selection."""

    valid_events_mask = np.ones(len(jets.btag), dtype=bool)
    n_btags_cut = selection.get("count")
    if n_btags_cut:
        n_btags = ak.sum(jets.btag[jets.valid], axis=1)
        valid_events_mask = valid_events_mask & get_op(n_btags_cut["operator"])(
            n_btags, n_btags_cut["value"]
        )
    # jets mask for valid events (mantains original size of events array)
    valid_n_bjets_mask = ak.where(valid_events_mask, jets.valid, False)
    return valid_n_bjets_mask, valid_events_mask


def select_hc_jets(jets, nbjets_cut=4):
    """Selects events by applying the cuts specified in the arguments.
    The HH system is reconstructed from two Higgs candidates, which are
    themselves reconstructed from two jets each (four Higgs candidate jets in total).

    b-jets are selected first. If the event is a 4b event, the leading four
    in pT are selected. If it is a 2b event, the remaining places are filled
    by non-b-tagged jets, which are sorted in pT and the two leading jets taken

    Returns the 4 Higgs candidate jets in each event.
    """
    jet_pt = ak.mask(jets.pt, jets.valid)
    jet_btag = ak.mask(jets.btag, jets.valid)
    jet_idx = ak.mask(ak.local_index(jet_pt), jets.valid)
    pt_sort_idx = ak.argsort(jet_pt, ascending=False)
    jet_btag = jet_btag[pt_sort_idx]
    jet_idx = jet_idx[pt_sort_idx]
    btag_sort_idx = ak.argsort(jet_btag, ascending=False)
    jet_idx = jet_idx[btag_sort_idx]
    hc_jet_idx = jet_idx[:, :nbjets_cut]
    non_hc_jet_idx = jet_idx[:, nbjets_cut:]
    return hc_jet_idx, non_hc_jet_idx


def select_X_Wt_events(events, selection):
    """Selects events that pass the top-veto selection.
    Events are vetoed if the minimum X_Wt over all combinations is less than selection.
    Returns:
        Events that pass the top-veto selection
    """
    jet_keys = [f"jet_{var}" for var in kin_labels.keys()]
    jet_keys += ["jet_btag"]
    jet_record = events[jet_keys]
    jet_idx = ak.concatenate([events.hc_jet_idx, events.non_hc_jet_idx], axis=1)
    jet_record = jet_record[jet_idx]
    jet_p4 = p4.zip(
        {var: jet_record[f"jet_{var}"] for var in kin_labels.keys()},
    )
    bjet_p4 = jet_p4[events.hc_jet_idx][jet_record.jet_btag[events.hc_jet_idx] == 1]
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
    # X_Wt_discriminant_min = ak.fill_none(X_Wt_discriminant_min, np.nan)
    passed_top_veto_mask = get_op(selection["operator"])(
        X_Wt_discriminant_min, selection["value"]
    )
    # passed_top_veto_mask = ak.fill_none(passed_top_veto_mask, False)
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
        if mass_sel.get("inner_boundry"):
            hh_var = X_HH(ak.firsts(h1.m) * inv_GeV, ak.firsts(h2.m) * inv_GeV)
            keep = keep & get_op(mass_sel["inner_boundry"]["operator"])(
                hh_var, mass_sel["inner_boundry"]["value"]
            )
        if mass_sel.get("outer_boundry"):
            hh_var = R_CR(ak.firsts(h1.m) * inv_GeV, ak.firsts(h2.m) * inv_GeV)
            keep = keep & get_op(mass_sel["outer_boundry"]["operator"])(
                hh_var, mass_sel["outer_boundry"]["value"]
            )
    return keep, hh_var


def select_correct_hh_pair_events(
    jets_truth_matched_to_hh, leading_h_jet_idx, subleading_h_jet_idx
):
    leading_h_truth_matched = jets_truth_matched_to_hh[leading_h_jet_idx]
    leading_h_jet1_truth_matched = leading_h_truth_matched[:, 0, np.newaxis]
    leading_h_jet2_truth_matched = leading_h_truth_matched[:, 1, np.newaxis]
    leading_h_jets_have_same_parent_mask = (
        leading_h_jet1_truth_matched == leading_h_jet2_truth_matched
    )
    subleading_h_truth_matched = jets_truth_matched_to_hh[subleading_h_jet_idx]
    subleading_h_jet1_truth_matched = subleading_h_truth_matched[:, 0, np.newaxis]
    subleading_h_jet2_truth_matched = subleading_h_truth_matched[:, 1, np.newaxis]
    subleading_h_jets_have_same_parent_mask = (
        subleading_h_jet1_truth_matched == subleading_h_jet2_truth_matched
    )
    correct_hh_pairs_mask = ak.firsts(
        leading_h_jets_have_same_parent_mask & subleading_h_jets_have_same_parent_mask
    )
    # convert to numpy array and replace None with False
    correct_hh_pairs_mask = ak.fill_none(correct_hh_pairs_mask, False).to_numpy()
    return correct_hh_pairs_mask


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


def select_buckets(events, leading_jet_pt_cut=170, third_jet_pt_cut=70):
    jet_pt = events.jet_pt[events.n_central_bjets]
    leading_jet_pt = jet_pt[:, 0]
    third_jet_pt = jet_pt[:, 2]
    passed_king_cut = (leading_jet_pt > leading_jet_pt_cut) & (
        third_jet_pt > third_jet_pt_cut
    )


def calculate_scale_factors(events):
    """Calculates the scale factors for each event.

    Returns:
        The scale factors for each event
    """
    # jsfs = t.arrays('sf', aliases=jalias, cut=f'(pt > {pT_min}) & (abs(eta) < {eta_max}) & {jvtCut}')
    # mc_sf = ak.prod(jsfs.sf[:,:,0],axis=-1).to_numpy()
    sf = np.ones(len(events))
    return sf
