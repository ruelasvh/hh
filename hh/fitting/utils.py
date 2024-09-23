import uproot
import numpy as np
import h5py
import hist as bh
from hh.shared.error import propagate_errors


def create_hist(counts, binning, errors=None, name="", axis_name=r"m_\mathrm{HH}"):
    """Create a Hist object from counts, binning, and errors."""
    hist = bh.Hist(
        bh.axis.Variable(binning, underflow=False, overflow=False, name=axis_name),
        storage=bh.storage.Weight(),
        name=name,
    )
    hist[...] = np.stack([counts, errors], axis=-1) if errors is not None else counts
    return hist


def create_hist_v2(counts, binning, errors=None):
    hist = uproot.writing.identify.to_TH1x(
        fName="h1",
        fTitle="title",
        data=np.array([0] + counts.tolist() + [0], np.float64),
        fEntries=0.0,
        fTsumw=0.0,
        fTsumw2=0.0,
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fSumw2=np.array([0] + errors.tolist() + [0], np.float64),
        fXaxis=uproot.writing.identify.to_TAxis(
            fName="xaxis",
            fTitle="",
            fNbins=len(counts),
            fXmin=binning[0],
            fXmax=binning[-1],
        ),
    )
    return hist


def save_to_root(hists, config, regions):
    """Convert a dictionary of sample histograms to a ROOT file."""
    hist_path = config["General"]["InputPath"].split(":")[0]
    with uproot.recreate(hist_path) as file_root:
        for base_reg in config["Regions"]:
            for reg in regions:
                region_path = f"{base_reg['RegionPath']}_{reg}"
                variation_path = f"{config['General']['VariationPath']}_{reg}"
                for sample in config["Samples"]:
                    if sample.get("Data", False):
                        continue
                    sample_name = sample["Name"]
                    sample_path = sample["SamplePath"]
                    if "Background" == sample_name:
                        bkg_keys = ["multijet", "ttbar"]
                        bkg_keys = [
                            k
                            for k in hists.keys()
                            if any([bkg in k for bkg in bkg_keys])
                        ]
                        background_counts = np.sum(
                            [
                                hists[k][variation_path]["values"][1:-1]
                                for k in bkg_keys
                            ],
                            axis=0,
                        )
                        background_errors = np.sqrt(
                            np.sum(
                                [
                                    hists[k][variation_path]["errors"][1:-1] ** 2
                                    for k in bkg_keys
                                ],
                                axis=0,
                            )
                        )
                        binning = hists[bkg_keys[0]][variation_path]["edges"]
                        file_root[f"{region_path}/{sample_path}/{variation_path}"] = (
                            create_hist(
                                background_counts,
                                binning,
                                background_errors,
                                name=sample_path,
                            )
                        )
                    else:
                        merged_sample_names = [
                            key for key in hists.keys() if sample_name in key
                        ]
                        merged_counts = np.sum(
                            [
                                hists[key][variation_path]["values"][1:-1]
                                for key in merged_sample_names
                            ],
                            axis=0,
                        )
                        merged_errors = np.sqrt(
                            np.sum(
                                [
                                    hists[key][variation_path]["errors"][1:-1] ** 2
                                    for key in merged_sample_names
                                ],
                                axis=0,
                            )
                        )
                        binning = hists[merged_sample_names[0]][variation_path]["edges"]
                        file_root[f"{region_path}/{sample_path}/{variation_path}"] = (
                            create_hist(
                                merged_counts,
                                binning,
                                merged_errors,
                                name=sample_path,
                            )
                        )


def save_to_h5(hists, name="histograms.h5", compress=True):
    """Convert a dictionary of sample histograms to a HDF5 file."""
    compression = dict(compression="gzip") if compress else {}
    with h5py.File(name, "w") as out_h5:
        for sample_key, sample_hists in hists.items():
            sgroup = out_h5.create_group(sample_key)
            sgroup.attrs["type"] = "sample_type"
            for sample_hist_key, sample_hist in sample_hists.items():
                hgroup = sgroup.create_group(sample_hist_key)
                hgroup.attrs["type"] = "float"
                hist = hgroup.create_dataset(
                    "values", data=sample_hist["values"], **compression
                )
                ax = hgroup.create_dataset(
                    "edges", data=sample_hist["edges"], **compression
                )
                ax.make_scale("edges")
                hist.dims[0].attach_scale(ax)
                if "errors" in sample_hist:
                    hgroup.create_dataset(
                        "errors", data=sample_hist["errors"], **compression
                    )
