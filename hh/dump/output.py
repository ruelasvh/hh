from enum import Enum


class OutputVariables(Enum):
    EVENT_NUMBER = "event_number"
    EVENT_WEIGHT = "event_weight"
    LABEL_HH = "label_HH"
    LABEL_TTBAR = "label_ttbar"
    LABEL_QCD = "label_QCD"
    JET_PX = "jet_px"
    JET_PY = "jet_py"
    JET_PZ = "jet_pz"
    N_JETS = "n_jets"
    N_BTAGS = "n_btags"
    JET_DL1DV01_PB = "jet_btag_DL1dv01_pb"
    JET_DL1DV01_PC = "jet_btag_DL1dv01_pc"
    JET_DL1DV01_PU = "jet_btag_DL1dv01_pu"
    JET_DL1DV01_DISCRIMINANT = "jet_btag_DL1dv01_discriminant"
    JET_BTAG_GN2V01_PB = "jet_btag_GN2v01_pb"
    JET_BTAG_GN2V01_PC = "jet_btag_GN2v01_pc"
    JET_BTAG_GN2V01_PU = "jet_btag_GN2v01_pu"
    JET_BTAG_GN2V01_DISCRIMINANT = "jet_btag_GN2v01_discriminant"
    M_4B = "m_4b"
    PT_4B = "pt_4b"
    ETA_4B = "eta_4b"
    PHI_4B = "phi_4b"
    BB_DM = "bb_dM"
    BB_DR = "bb_dR"
    BB_DETA = "bb_dEta"
    JET_PT = "jet_pt"
    JET_ETA = "jet_eta"
    JET_PHI = "jet_phi"
    JET_MASS = "jet_mass"
    DELTAETA_HH = "dEta_HH"
    X_WT = "X_Wt"
    X_HH = "X_HH"
    YEAR = "year"
    HH_TRUTH_M = "hh_truth_mass"
    HH_TRUTH_PT = "hh_truth_pt"
    HH_TRUTH_ETA = "hh_truth_eta"
    HH_TRUTH_PHI = "hh_truth_phi"
    H1_TRUTH_MASS = "h1_truth_mass"
    H1_TRUTH_PT = "h1_truth_pt"
    H1_TRUTH_ETA = "h1_truth_eta"
    H1_TRUTH_PHI = "h1_truth_phi"
    H2_TRUTH_MASS = "h2_truth_mass"
    H2_TRUTH_PT = "h2_truth_pt"
    H2_TRUTH_ETA = "h2_truth_eta"
    H2_TRUTH_PHI = "h2_truth_phi"
    H1_RECO_PT = "h1_reco_pt"
    H1_RECO_ETA = "h1_reco_eta"
    H1_RECO_PHI = "h1_reco_phi"
    H1_RECO_MASS = "h1_reco_mass"
    H2_RECO_PT = "h2_reco_pt"
    H2_RECO_ETA = "h2_reco_eta"
    H2_RECO_PHI = "h2_reco_phi"
    H2_RECO_MASS = "h2_reco_mass"
    HH_RECO_MASS = "hh_reco_mass"
    HH_RECO_PT = "hh_reco_pt"
    HH_RECO_ETA = "hh_reco_eta"
    HH_RECO_PHI = "hh_reco_phi"

    @classmethod
    def get_all(cls):
        return [output_variable.value for output_variable in cls]

    @classmethod
    def contains_all(cls, output_variables):
        return all(
            [output_variable in cls.get_all() for output_variable in output_variables]
        )
