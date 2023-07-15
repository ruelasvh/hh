import numpy as np
from abc import ABC, abstractmethod


class BaseHistogram(ABC):
    @abstractmethod
    def fill(self, vals):
        pass

    @abstractmethod
    def write(self, group):
        pass


class Histogram(BaseHistogram):
    def __init__(self, name, binrange, bins=100, compress=True):
        self._name = name
        self._binning = np.linspace(*binrange, bins)
        self._hist = np.zeros(self._binning.size - 1, dtype=np.float64)
        self._compression = dict(compression="gzip") if compress else {}

    def fill(self, values, weights=None):
        hist, _ = np.histogram(
            values,
            self._binning,
            weights=weights,
        )
        self._hist = self._hist + hist

    @property
    def name(self):
        return self._name

    @property
    def values(self):
        return self._hist

    @property
    def edges(self):
        return self._binning

    def write(self, group, name=None):
        hgroup = group.create_group(name or self._name)
        hgroup.attrs["type"] = "float"
        hist = hgroup.create_dataset("values", data=self._hist, **self._compression)
        ax = hgroup.create_dataset("edges", data=self._binning, **self._compression)
        ax.make_scale("edges")
        hist.dims[0].attach_scale(ax)


class Histogramddv2(Histogram):
    def __init__(self, name, binrange, bins=100, compress=True):
        self._name = name
        self._binning = np.linspace(*binrange, bins)
        self._hist = np.zeros(
            (self._binning.size - 1, self._binning.size - 1), dtype=np.float64
        )
        self._compression = dict(compression="gzip") if compress else {}

    def fill(self, vals, weights=None):
        hist = np.histogramdd(
            vals, bins=(self._binning, self._binning), weights=weights
        )[0]
        self._hist = self._hist + hist
