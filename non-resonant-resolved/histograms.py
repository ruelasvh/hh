import numpy as np
from abc import ABC, abstractmethod
from error_estimation import getEfficiencyWithUncertainties


class Histogram(ABC):
    @abstractmethod
    def fill(self, vals):
        pass

    # @abstractmethod
    # def write(self, group):
    #     pass


class IntHistogram(Histogram):
    def __init__(self, name, binrange, bins=100, compress=True):
        self._name = name
        self._bins = np.linspace(*binrange, bins)
        self._hist = np.zeros(self._bins.size - 1, dtype=np.int64)
        self._compression = dict(compression="gzip") if compress else {}

    def fill(self, vals):
        hist = np.histogram(vals, self._bins)[0]
        self._hist = self._hist + hist

    @property
    def values(self):
        return self._hist

    @property
    def edges(self):
        return self._bins


class IntHistogramdd(IntHistogram):
    def fill(self, *vals):
        hist = np.array([np.histogram(data, self._bins)[0] for data in vals])
        self._hist = self._hist + hist


class EffHistogram(IntHistogramdd):
    def fill(self, *vals):
        assert (
            len(vals) == 2
        ), "EffHistogram only accepts 2 input params, h_pass and h_total"
        hist = np.array([np.histogram(data, self._bins)[0] for data in vals])
        self._hist = self._hist + hist

    @property
    def values(self):
        passed, total = self._hist
        eff, err = getEfficiencyWithUncertainties(passed, total)
        return eff, err


class EffHistogramdd(Histogram):
    def __init__(self, name, binrange, bins=100, compress=True):
        self._name = name
        self._bins = np.linspace(*binrange, bins)
        self._hist = np.zeros((self._bins.size - 1, self._bins.size - 1), dtype=float)
        self._compression = dict(compression="gzip") if compress else {}

    def fill(self, *vals):
        assert (
            len(vals) == 2
        ), "EffHistogram only accepts 2 input params, h_pass and h_total"
        hist = np.array(
            [np.histogramdd(data, bins=(self._bins, self._bins))[0] for data in vals]
        )
        self._hist = self._hist + np.array(hist)

    @property
    def values(self):
        passed, total = self._hist
        eff = passed / total
        return eff, None

    @property
    def edges(self):
        return self._bins
