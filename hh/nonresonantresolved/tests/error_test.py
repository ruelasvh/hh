import unittest
import numpy as np
from hh.shared.error import get_symmetric_bin_errors, propagate_errors


class TestErrors(unittest.TestCase):
    def test_get_symmetric_bin_errors(self):
        data = np.array([1, 2, 3, 4, 5])
        weights = np.array([1, 1, 1, 1, 1])
        bins = np.array([0, 2, 4, 6])
        expected_bin_errors = np.array([np.sqrt(2), np.sqrt(2), 1])
        actual_bin_errors = get_symmetric_bin_errors(data, weights, bins)
        np.testing.assert_array_equal(actual_bin_errors, expected_bin_errors)

    def test_propagate_errors(self):
        sigmaA = np.array([1, 1, 1])
        sigmaB = np.array([1, 1, 1])
        operation = "+"
        A = np.array([1, 2, 3])
        B = np.array([4, 5, 6])
        expected_error = np.array([np.sqrt(2), np.sqrt(2), np.sqrt(2)])
        actual_error = propagate_errors(
            sigmaA=sigmaA, sigmaB=sigmaB, operation=operation, A=A, B=B
        )
        np.testing.assert_array_equal(actual_error, expected_error)


if __name__ == "__main__":
    unittest.main()

# def make_plot():
#     import mplhep as hep
#     import matplotlib.pyplot as plt

#     data = np.array([1, 8, 5, 4, 1, 10, 8, 3, 6, 7])
#     weights = np.array([1.3, 0.2, 0.01, 0.9, 0.4, 1.05, 0.6, 0.6, 0.8, 1.8])
#     # create data2 similar to in shape but different values
#     data2 = np.array([5, 1, 3, 3, 2, 1, 2, 2, 7, 6])
#     weights2 = np.array([0.2, 1.3, 0.9, 0.01, 1.05, 0.4, 1.8, 0.6, 0.6, 0.8])
#     bins = np.array([0.0, 2.5, 5.0, 7.5, 10.0])  # np.linspace(0, 10, 5)

#     fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
#     axs = axs.flatten()

#     _hist = np.zeros(bins.size - 1, dtype=float)
#     _errors = np.zeros(bins.size - 1, dtype=float)
#     _hist += np.histogram(data, bins=bins, weights=weights)[0]
#     _errors += get_symmetric_bin_errors(data, weights, bins)
#     _hist += np.histogram(data2, bins=bins, weights=weights2)[0]
#     _errors = propagate_errors(
#         _errors, get_symmetric_bin_errors(data2, weights2, bins), operation="+"
#     )

#     axs[0].set_title("Wgts batched", fontsize=18)
#     hep.histplot(_hist, bins, yerr=_errors, ax=axs[0])

#     hist = np.histogram(
#         np.concatenate([data, data2]),
#         bins=bins,
#         weights=np.concatenate([weights, weights2]),
#     )[0]
#     errors = get_symmetric_bin_errors(
#         np.concatenate([data, data2]),
#         weights=np.concatenate([weights, weights2]),
#         bins=bins,
#     )
#     axs[1].set_title("Wgts concatenated", fontsize=18)
#     hep.histplot(hist, bins, yerr=errors, ax=axs[1])

#     fig.tight_layout()
#     fig.savefig("errorbars.png")
