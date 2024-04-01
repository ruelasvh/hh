from enum import Enum


class Features(Enum):
    JET_PT = "jet_pt"
    JET_ETA = "jet_eta"
    JET_PHI = "jet_phi"
    JET_MASS = "jet_mass"
    JET_X = "jet_x"
    JET_Y = "jet_y"
    JET_Z = "jet_z"
    JET_NUM = "jet_num"
    JET_BTAG = "jet_btag"
    JET_NBTAGS = "jet_nbtags"
    JET_DL1DV01_PB = "jet_btag_DL1dv01_pb"
    JET_DL1DV01_PC = "jet_btag_DL1dv01_pc"
    JET_DL1DV01_PU = "jet_btag_DL1dv01_pu"
    JET_BTAG_GN2V01_PB = "jet_btag_GN2v01_pb"
    JET_BTAG_GN2V01_PC = "jet_btag_GN2v01_pc"
    JET_BTAG_GN2V01_PU = "jet_btag_GN2v01_pu"
    EVENT_M_4B = "m_4b"
    EVENT_BB_RMH = "bb_RmH"
    EVENT_BB_DR = "bb_dR"
    EVENT_BB_DETA = "bb_dEta"
    EVENT_DELTAETA_HH = "deltaEta_hh"
    EVENT_X_WT = "X_Wt"
    EVENT_X_HH = "X_hh"
    EVENT_WEIGHT = "event_weight"
    MC_EVENT_WEIGHT = "mc_event_weight"

    @classmethod
    def get_all(cls):
        return [feature.value for feature in cls]

    @classmethod
    def contains_all(cls, features):
        return all([feature in cls.get_all() for feature in features])


class Labels(Enum):
    LABEL_HH = "label_HH"
    LABEL_TTBAR = "label_ttbar"
    LABEL_QCD = "label_QCD"

    @classmethod
    def get_all(cls):
        return [label.value for label in cls]

    @classmethod
    def contains_all(cls, labels):
        return all([label in cls.get_all() for label in labels])
