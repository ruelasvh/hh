import numpy as np
from abc import ABC, abstractmethod
from hh.shared.error import get_symmetric_bin_errors, propagate_errors


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
        self._hist = np.zeros(self._binning.size - 1, dtype=float)
        self._error = np.zeros(self._binning.size - 1, dtype=float)
        self._compression = dict(compression="gzip") if compress else {}

    def fill(self, values, weights=None):
        hist, _ = np.histogram(values, bins=self._binning, weights=weights)
        self._hist = self._hist + hist
        if weights is not None:
            self._error = propagate_errors(
                self._error,
                get_symmetric_bin_errors(values, weights, self._binning),
                operation="+",
            )

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


class Histogram2d(Histogram):
    def __init__(self, name, binrange, bins=100, compress=True):
        self._name = name
        self._binning = np.linspace(*binrange, bins)
        self._hist = np.zeros(
            (self._binning.size - 1, self._binning.size - 1), dtype=float
        )
        self._compression = dict(compression="gzip") if compress else {}

    def fill(self, vals, weights=None):
        hist = np.histogramdd(
            vals, bins=(self._binning, self._binning), weights=weights
        )[0]
        self._hist = self._hist + hist


class HistogramDynamic(Histogram):
    def __init__(self, name, bins=100, dtype=np.int64, compress=True):
        self._name = name
        self._bins = bins
        self._dtype = dtype
        self._data = []
        self._compression = dict(compression="gzip") if compress else {}

    def fill(self, values):
        self._data += values.tolist() if isinstance(values, np.ndarray) else values

    def write(self, group, name=None):
        hgroup = group.create_group(name or self._name)
        hgroup.attrs["type"] = self._dtype.__name__
        data = np.array(self._data, dtype=self._dtype)
        if np.issubdtype(self._dtype, int):
            bin_size = int(np.ceil((data.max() - data.min()) / self._bins))
            hist, edges = np.histogram(
                self._data, bins=range(data.min(), data.max() + bin_size, bin_size)
            )
        else:
            hist, edges = np.histogram(self._data, bins=self._bins)
        hist = hgroup.create_dataset("values", data=hist, **self._compression)
        ax = hgroup.create_dataset("edges", data=edges, **self._compression)
        ax.make_scale("edges")
        hist.dims[0].attach_scale(ax)
