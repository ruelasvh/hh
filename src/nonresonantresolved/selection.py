import awkward as ak
import numpy as np
import vector


def select_n_jets_events(
    events,
    jet_vars=None,
    pt_cut=None,
    eta_cut=None,
    njets=4,
):
    if jet_vars is None:
        return events
    sorted_events = sort_jets_by_pt(events, jet_vars)
    jet_pt_var, jet_eta_var, *_ = jet_vars
    jet_pt = sorted_events[jet_pt_var]
    jet_eta = sorted_events[jet_eta_var]
    if pt_cut and eta_cut:
        valid_events = (jet_pt > pt_cut) & (np.abs(jet_eta) < eta_cut)
    elif pt_cut and not eta_cut:
        valid_events = jet_pt > pt_cut
    elif eta_cut and not pt_cut:
        valid_events = np.abs(jet_eta) < eta_cut
    else:
        valid_events = None
    if valid_events is not None:
        valid_events = ak.num(jet_pt[valid_events]) >= njets
    else:
        valid_events = ak.num(jet_pt) >= njets

    return sorted_events[valid_events]


def sort_jets_by_pt(events, jet_vars=None):
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


def X_HH(mH1, mH2):
    """Calculate signal region discriminat.

    X_HH = sqrt(
        ((mH1 - 124 GeV) / 0.1 * mH1)^2 + ((mH2 - 117 GeV) / 0.1 * mH2)^2
    )
    """

    first_term = np.zeros_like(mH1)
    np.divide(mH1 - 124, 0.1 * mH1, out=first_term, where=(mH1 != 0))
    second_term = np.zeros_like(mH2)
    np.divide(mH2 - 117, 0.1 * mH2, out=second_term, where=(mH2 != 0))

    return np.sqrt(first_term**2 + second_term**2)


def R_CR(mH1, mH2):
    """Calculate outer edge of control region discriminant.

    R_CR = sqrt(
        (mH1 - 1.05 * 124 GeV)^2 + (mH2 - 1.05 * 117 GeV)^2
    )
    """

    return np.sqrt((mH1 - 1.05 * 124) ** 2 + (mH2 - 1.05 * 117) ** 2)


#
# Very much WIP
#
def select_X_Wt_events(events, m_w=80.4, m_t=172.5):
    """Calculate top-veto discriminant. All masses in GeV.

    X_Wt = min(
        sqrt(
            ((m_jj - m_W) / 0.1 * m_jj)^2 + ((m_jjb - m_t) / 0.1 * m_jjb)^2
        )
    )

    m_jj = W candidate (2 jets)
    m_jjb = top quark candidate (W candidate paired with a b-jet use in reconstructing Higgses)
    """

    jets_p4 = vector.zip(
        {
            "pt": events.jet_pt,
            "eta": events.jet_eta,
            "phi": events.jet_phi,
            "mass": events.jet_m,
        }
    )
    W1, W2 = ak.unzip(ak.combinations(jets_p4, 2))
    # print(W1, W2)
    bjet_decisions = events.btag_DL1dv01_70 == 1
    bjets_p4 = jets_p4[bjet_decisions]
    non_bjets_p4 = jets_p4[~bjet_decisions]
    print("jets per event", ak.num(jets_p4))
    print("b-jets per event", ak.num(bjets_p4))
    print("non b-jets per event", ak.num(non_bjets_p4))
    # W, b = ak.cartesian([ak.combinations(non_bjets_p4, 2), bjets_p4])
    # print(b)
