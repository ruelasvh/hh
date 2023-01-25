import numpy as np
from scipy import stats


def EfficiencyErrorBayesian(k, n, bUpper):
    """Error estimation based on Bayesian methods. Calculation is per bin.
    Parameters
    ----------
        k       :   0 <= number of passed events <= total
        n       :   number of total events
        bUpper  :   true - upper boundary is returned
                    false - lower boundary is returned

    https://arxiv.org/abs/physics/0701199v1
    """
    if n == 0:
        if bUpper:
            return 0
        else:
            return 1

    firstTerm = ((k + 1) * (k + 2)) / ((n + 2) * (n + 3))
    secondTerm = ((k + 1) ** 2) / ((n + 2) ** 2)
    error = np.sqrt(firstTerm - secondTerm)
    mean = k / n
    if bUpper:
        if (mean + error) >= 1:
            return 1.0
        else:
            return mean + error
    else:
        if (mean - error) <= 0:
            return 0.0
        else:
            return mean - error


def getEfficiencyWithUncertainties(passed, total):
    """Get relative upper and lower error bar positions"""
    upper_err = np.array(
        [
            EfficiencyErrorBayesian(passed, total, bUpper=True)
            for passed, total in zip(passed, total)
        ]
    )
    lower_err = np.array(
        [
            EfficiencyErrorBayesian(passed, total, bUpper=False)
            for passed, total in zip(passed, total)
        ]
    )

    # TODO: Root implementation
    # upper_err = np.array(
    #     [Bayesian(passed, total, bUpper=True) for passed, total in zip(passed, total)],
    #     dtype=np.float32,
    # )
    # lower_err = np.array(
    #     [Bayesian(passed, total, bUpper=False) for passed, total in zip(passed, total)],
    #     dtype=np.float32,
    # )

    efficiency = passed / total
    relative_errors = np.array([efficiency - lower_err, upper_err - efficiency])
    return efficiency, relative_errors


# TODO: as implemented in Root
def Bayesian(total, passed, level=0.682689492137, alpha=1.0, beta=1.0, bUpper=False):
    a = passed + alpha
    b = (total - passed) + beta
    return BetaCentralInterval(level, a, b, bUpper)


def BetaCentralInterval(level, a, b, bUpper):
    if bUpper:
        if (a > 0) & (b > 0):
            return stats.beta.ppf((1 + level) / 2, a, b)
        else:
            return 1.0
    else:
        if (a > 0) & (b > 0):
            return stats.beta.ppf((1 - level) / 2, a, b)
        else:
            0
