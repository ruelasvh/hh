import numpy as np


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

    return np.sqrt((m_H1 - 1.05 * m_H1_center) ** 2 + (m_H2 - 1.05 * m_H2_center) ** 2)


def X_Wt(m_jj, m_jjb, m_W=80.4, m_t=172.5):
    """Calculate top-veto discriminant. Where m_jj is the mass of the W candidate
    and m_jjb is the mass of the top candidate.

    X_Wt = sqrt(
        ((m_jj - 80.4 GeV) / 0.1 * m_jj)^2 + ((m_jjb - 172.5 GeV) / 0.1 * m_jjb)^2
    )
    """

    return calculate_discrim(m_jj, m_jjb, m_W, m_t)
