import numpy as np
from scipy import stats


def efficiency_error_bayesian(k, n, bUpper):
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
        if (mean + error) > 1:
            return 1.0
        else:
            return mean + error
    else:
        if (mean - error) < 0:
            return 0.0
        else:
            return mean - error


def get_efficiency_with_uncertainties(passed, total):
    """Get relative upper and lower error bar positions"""
    upper_err = np.array(
        [efficiency_error_bayesian(p, t, bUpper=True) for p, t in zip(passed, total)]
    )
    lower_err = np.array(
        [efficiency_error_bayesian(p, t, bUpper=False) for p, t in zip(passed, total)]
    )

    # TODO: Root implementation
    # upper_err = np.array(
    #     [bayesian(p, t, bUpper=True) for p, t in zip(passed, total)],
    #     dtype=np.float32,
    # )
    # lower_err = np.array(
    #     [bayesian(p, t, bUpper=False) for p, t in zip(passed, total)],
    #     dtype=np.float32,
    # )

    # cap efficiency at 1, dirty fix
    efficiency = np.minimum(passed / total, 1)
    relative_errors = np.array([efficiency - lower_err, upper_err - efficiency])
    return efficiency, relative_errors


# TODO: as implemented in Root
def bayesian(total, passed, level=0.682689492137, alpha=1.0, beta=1.0, bUpper=False):
    a = passed + alpha
    b = (total - passed) + beta
    return beta_central_interval(level, a, b, bUpper)


def beta_central_interval(level, a, b, bUpper):
    if bUpper:
        if (a > 0) & (b > 0):
            return stats.beta.ppf((1 + level) / 2, a, b)
        else:
            return 1.0
    else:
        if (a > 0) & (b > 0):
            return stats.beta.ppf((1 - level) / 2, a, b)
        else:
            return 0.0


def get_symmetric_bin_errors(data, weights, bins):
    """Computs bin error bars for a histogram. Mathematically they are calculated as:

    err(bin) = sqrt(sum(w_i^2)),

    where w_i is the weight of the i-th event in the bin.

    Parameters
    ----------
    data : array-like
        The data to be histogrammed.
    weights : array-like
        The weights of the data.
    bins : array-like
        The bin edges.
    """

    bin_edges = bins

    bin_errors = np.zeros(len(bin_edges) - 1)
    for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (low <= data) & (data < high)
        bin_errors[i] = np.sqrt(np.sum(weights[mask] ** 2))

    return bin_errors


def calculate_2d_bin_errors(data, weights, bins_x, bins_y):
    """
    Calculate bin errors for a 2D histogram using np.digitize.

    Parameters:
    - data: ndarray of shape (2, N)
        The input data to be histogrammed. The first row corresponds to the x-values,
        and the second row corresponds to the y-values.
    - weights: ndarray of shape (N,)
        The weights associated with each data point.
    - bins_x: ndarray
        The bin edges for the x-dimension.
    - bins_y: ndarray
        The bin edges for the y-dimension.

    Returns:
    - bin_centers_x: ndarray
        The centers of the x-bins.
    - bin_centers_y: ndarray
        The centers of the y-bins.
    - bin_errors: ndarray of shape (len(bins_x)-1, len(bins_y)-1)
        The errors for each 2D bin.
    """

    # Ensure data is a 2D array with shape (2, N)
    data = np.asarray(data)
    if data.shape[0] != 2:
        raise ValueError("Data should have shape (2, N) for a 2D histogram.")

    # Digitize the data to find bin indices
    bin_indices_x = np.digitize(data[0], bins_x) - 1
    bin_indices_y = np.digitize(data[1], bins_y) - 1

    # Initialize bin errors array
    bin_errors = np.zeros((len(bins_x) - 1, len(bins_y) - 1))

    # Calculate bin errors
    for i in range(len(bins_x) - 1):
        for j in range(len(bins_y) - 1):
            mask = (bin_indices_x == i) & (bin_indices_y == j)
            bin_errors[i, j] = np.sqrt(np.sum(weights[mask] ** 2))

    return bin_errors


def propagate_errors(
    sigmaA=None,
    sigmaB=None,
    operation=None,
    A=None,
    B=None,
    exp=None,
):
    """
    calculate propagated error based from
    https://en.wikipedia.org/wiki/Propagation_of_uncertainty

    Parameters
    ----------
    sigmaA : ndarray
        standard error of A
    sigmaB : ndarray
        standard error of B
    operation : str
        +, -, *, /, ^
    A : ndarray, optional
        A values, by default None
    B : ndarray, optional
        B values, by default None
    exponent : ndarray
        for power operation on A
    Returns
    -------
    np.ndarray
        propagated error
    """

    if "+" in operation or "-" in operation:
        error = np.sqrt(np.power(sigmaA, 2) + np.power(sigmaB, 2))
    elif "*" in operation:
        error = np.abs(A * B) * np.sqrt(
            np.power(np.divide(sigmaA, A, out=np.zeros_like(sigmaA), where=A != 0), 2)
            + np.power(np.divide(sigmaB, B, out=np.zeros_like(sigmaB), where=B != 0), 2)
        )
    elif "/" in operation:
        error = np.abs(A / B) * np.sqrt(
            np.power(np.divide(sigmaA, A, out=np.zeros_like(sigmaA), where=A != 0), 2)
            + np.power(np.divide(sigmaB, B, out=np.zeros_like(sigmaB), where=B != 0), 2)
        )
    elif "^" in operation:
        error = np.abs(np.power(A, exp) / A * (exp * sigmaA))
    else:
        raise ValueError("Operation not supported")

    return error
