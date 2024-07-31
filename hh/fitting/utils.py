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


def save_to_root(hists, config):
    """Convert a dictionary of sample histograms to a ROOT file."""
    hist_path = config["General"]["InputPath"].split(":")[0]
    with uproot.recreate(hist_path) as file_root:
        for region in config["Regions"]:
            region_name = region["Name"]
            region_path = region["RegionPath"]
            variation_path = config["General"]["VariationPath"]
            variation_path_region = f"{variation_path}_{region_name}"
            for sample in config["Samples"]:
                if sample.get("Data", False):
                    continue
                sample_key = sample["Name"]
                if "ggF" in sample_key:
                    signal_sample_keys = [key for key in hists.keys() if "ggF" in key]
                    signal_counts = np.sum(
                        [
                            hists[key][variation_path_region]["values"][1:-1]
                            for key in signal_sample_keys
                        ],
                        axis=0,
                    )
                    signal_errors = np.sqrt(
                        np.sum(
                            [
                                hists[key][variation_path_region]["errors"][1:-1] ** 2
                                for key in signal_sample_keys
                            ],
                            axis=0,
                        )
                    )
                    binning = hists[signal_sample_keys[0]][variation_path_region][
                        "edges"
                    ]
                    file_root[
                        f"{region_path}/{sample['SamplePath']}/{variation_path}"
                    ] = create_hist(
                        signal_counts,
                        binning,
                        signal_errors,
                        name=sample_key,
                    )
                if "Background" == sample_key:
                    bkg_keys = ["multijet", "ttbar"]
                    bkg_keys = [
                        k for k in hists.keys() if any([bkg in k for bkg in bkg_keys])
                    ]
                    background_counts = np.sum(
                        [
                            hists[k][variation_path_region]["values"][1:-1]
                            for k in bkg_keys
                        ],
                        axis=0,
                    )
                    background_errors = np.sqrt(
                        np.sum(
                            [
                                hists[k][variation_path_region]["errors"][1:-1] ** 2
                                for k in bkg_keys
                            ],
                            axis=0,
                        )
                    )
                    binning = hists[bkg_keys[0]][variation_path_region]["edges"]
                    file_root[f"{region_path}/{sample_key}/{variation_path}"] = (
                        create_hist(
                            background_counts,
                            binning,
                            background_errors,
                            name=sample_key,
                        )
                    )


# def save_to_root(hists, config):
#     """Convert a dictionary of sample histograms to a ROOT file."""
#     hist_path = config["General"]["InputPath"].split(":")[0]
#     with uproot.recreate(hist_path) as file_root:
#         for region in config["Regions"]:
#             region_name = region["Name"]
#             region_path = region["RegionPath"]
#             variation_path = config["General"]["VariationPath"]
#             variation_path_region = f"{variation_path}_{region_name}"
#             for sample in config["Samples"]:
#                 if sample.get("Data", False):
#                     continue
#                 sample_key = sample["Name"]
#                 if "ggF" in sample_key:
#                     hist = hists[sample_key][variation_path_region]
#                     hist_counts = hist["values"][1:-1]
#                     hist_edges = hist["edges"]
#                     hist_errors = hist["errors"][1:-1]
#                     file_root[
#                         f"{region_path}/{sample['SamplePath']}/{variation_path}"
#                     ] = create_hist(
#                         hist_counts,
#                         hist_edges,
#                         hist_errors,
#                         name=sample_key,
#                     )
#                 if "Background" == sample_key:
#                     bkg_keys = ["multijet", "ttbar"]
#                     bkg_keys = [
#                         k for k in hists.keys() if any([bkg in k for bkg in bkg_keys])
#                     ]
#                     background_hist_counts = []
#                     background_hist_errors = []
#                     for k in bkg_keys:
#                         hist = hists[k][variation_path_region]
#                         hist_edges = hist["edges"]
#                         if len(background_hist_counts) == 0:
#                             background_hist_counts = hist["values"][1:-1]
#                             background_hist_errors = hist["errors"][1:-1]
#                         else:
#                             background_hist_counts += hist["values"][1:-1]
#                             background_hist_errors = propagate_errors(
#                                 background_hist_errors,
#                                 hist["errors"][1:-1],
#                                 operation="+",
#                             )
#                     file_root[f"{region_path}/{sample_key}/{variation_path}"] = (
#                         create_hist(
#                             background_hist_counts,
#                             hist_edges,
#                             background_hist_errors,
#                             name=sample_key,
#                         )
#                     )


# def save_to_root(hists, config, regions):
#     hist_path = config["General"]["InputPath"].split(":")[0]
#     with uproot.recreate(hist_path) as f:
#         for region in config["Regions"]:
#             variation_path = config["General"]["VariationPath"]
#             hist_edges = np.array(region["Binning"])
#             for region_name in regions:
#                 for sample in config["Samples"]:
#                     sample_name = sample["Name"]
#                     variation_path_region = f"{variation_path}_{region_name}"
#                     if sample.get("Data", False):
#                         continue
#                         f[
#                             f"{region['RegionPath']}_{reg}/{sample['SamplePath']}/{config['General']['VariationPath']}"
#                         ] = (np.zeros(len(hist_edges) - 1), hist_edges)
#                     else:
#                         sample_hist_name = (
#                             sample["Name"],
#                             f"{variation_path}_{region_name}",
#                         )
#                         f[
#                             f"{region['RegionPath']}_{region_name}/{sample['SamplePath']}/{variation_path}"
#                         ] = (
#                             hists[sample_hist_name[0]][sample_hist_name[1]]["values"][
#                                 2:-1
#                             ],  # Remove underflow and overflow bins
#                             hist_edges,
#                         )

#     return


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
