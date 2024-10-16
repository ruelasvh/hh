import os
import re
import h5py
import glob
import types
import uproot
import builtins
import operator
import itertools
import numpy as np
import awkward as ak
from pathlib import Path
import logging, coloredlogs
from functools import reduce
from collections import defaultdict
from hh.dump.output import Features, Labels
from hh.shared.error import propagate_errors


logger = logging.getLogger("hh4b-analysis")

nth = {1: "first", 2: "second", 3: "third", 4: "fourth"}

kin_labels = {"pt": r"$p_T$", "eta": r"$\eta$", "phi": r"$\phi$", "mass": r"$m$"}

jz_leading_jet_pt = {
    "min": {
        0: -1,
        1: 20,
        2: 60,
        3: 160,
        4: 400,
        5: 800,
        6: 1300,
        7: 1800,
        8: 2500,
        9: 3200,
        10: 3900,
        11: 4600,
        12: 5300,
    },
    "max": {
        0: 20,
        1: 60,
        2: 160,
        3: 400,
        4: 800,
        5: 1300,
        6: 1800,
        7: 2500,
        8: 3200,
        9: 3900,
        10: 4600,
        11: 5300,
        12: 7000,
    },
}


MeV = 1_000

GeV = 1 / MeV


def setup_logger(loglevel, filename=None):
    logger.setLevel(loglevel)
    if not filename:
        filename = f"hh4b-analysis_{os.getpgid(os.getpid())}.log"
    logging_handler = logging.FileHandler(filename)
    logging_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging_handler.setFormatter(logging_formatter)
    logger.addHandler(logging_handler)
    coloredlogs.install(level=logger.level, logger=logger)
    return logger


def get_com_lumi_label(com=13, lumi=None):
    com_label = r"$\sqrt{s} = \mathrm{" + str(com) + "\ TeV}"
    if lumi is not None:
        lumi_label = (
            ",\ " + str(format(lumi, ".1f")) + r"\ \mathrm{fb}^{-1}$" if lumi else "$"
        )
    else:
        lumi_label = "$"
    return com_label + lumi_label


def concatenate_cutbookkeepers(sample_path):
    sample_files = glob.glob(f"{sample_path}*.root")

    cutbookkeepers = {}
    cutbookkeepers["initial_events"] = 0
    cutbookkeepers["initial_sum_of_weights"] = 0
    cutbookkeepers["initial_sum_of_weights_squared"] = 0
    for file_path in sample_files:
        with uproot.open(file_path) as f:
            for key in f.keys():
                if "CutBookkeeper" and "NOSYS" in key:
                    cbk = f[key].to_numpy()
                    cutbookkeepers["initial_events"] += cbk[0][0]
                    cutbookkeepers["initial_sum_of_weights"] += cbk[0][1]
                    cutbookkeepers["initial_sum_of_weights_squared"] += cbk[0][2]
    return cutbookkeepers


def get_sample_weight(sample_metadata, cbk):
    sum_weights = cbk["initial_sum_of_weights"]
    filter_efficiency = float(sample_metadata["genFiltEff"])
    k_factor = float(sample_metadata["kFactor"])
    cross_section = float(sample_metadata["crossSection"]) * 1e6  # nb to fb
    luminosity = float(sample_metadata["luminosity"])  # fb^-1
    return (filter_efficiency * k_factor * cross_section * luminosity) / sum_weights


def get_datasetname_query(filepath):
    path_parts = filepath.split("/")
    project_name_part = list(
        filter(lambda part: re.search("(mc|data)[+0-9]+_[0-9]+", part), path_parts)
    )
    if project_name_part:
        project_name = project_name_part[0].split(".")[0]
    else:
        project_name = "%"
    dataset_localname = path_parts[-2].replace("_TREE", "")
    dataset_query_parts = dataset_localname.split(".")[-3:]
    dataset_query = ".%".join([project_name, *dataset_query_parts])
    return dataset_query


def write_hists(sample_hists, sample_name, output):
    sample_out = output.create_group(sample_name)
    sample_out.attrs["type"] = "sample_type"
    for hist in sample_hists:
        hist.write(sample_out, hist.name)


def get_op(op):
    return {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        "and": np.logical_and,
        "or": np.logical_or,
    }[op]


def find_hists(
    iteratable,
    pred=None,
):
    """Returns the found values in the iterable given pred.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.

    """
    return list(filter(pred, iteratable))


def find_hist(
    iteratable,
    pred=None,
    default=False,
):
    """Returns the first true value in the iterable.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.

    """
    return next(filter(pred, iteratable), default)


def find_hists_by_name(hists, delimeter):
    """Returns the list of hists that match the delimeter"""
    prog = re.compile(delimeter + "$")
    return list(filter(lambda h: prog.match(h.name), hists))


def get_trigs_logical_op(events, trigs, op="or", skip_trig=None):
    """Returns the decision of all trigs except skip_trig using bitwise operations"""
    trigs = list(filter(lambda trig: trig != skip_trig, trigs))
    trig_decisions = events[trigs]
    trig_decisions_or_mask = reduce(get_op(op), trig_decisions)
    return trig_decisions_or_mask


def get_trigs_logical_op(events, trigs, op="or", skip_trig=None):
    """Returns the OR decision of all trigs except skip_trig"""
    trigs = list(filter(lambda trig: trig != skip_trig, trigs))
    return reduce(
        # lambda acc, it: acc | events[it],
        lambda acc, it: get_op(op)(acc, events[it]),
        trigs,
        events[trigs[0]],
    )


def format_btagger_model_name(model, eff):
    # concatenate model and eff, e.g. "DL1dv00_77". eff could be a float or int
    if isinstance(eff, float):
        return f"{model}_{eff*100:.0f}"
    else:
        return f"{model}_{eff}"


# write function that 1: gets this projects root path, 2: lists directories under it,  3: gets path as argument, 4: returns path arg using root path as base
def get_root_path():
    return Path(__file__).parent.parent.parent


def get_dirs(root_path):
    return [d for d in root_path.iterdir() if d.is_dir()]


def resolve_project_path(path, root_path=None):
    project_root_path = root_path if root_path else get_root_path()
    project_dirs = get_dirs(project_root_path)
    if isinstance(path, list):
        return [resolve_project_path(p, project_root_path) for p in path]
    # check if path starts with any of the dirs in project_dirs
    # if it does return path prepended with project_root_path else return path
    for project_dir in project_dirs:
        if path.startswith(project_dir.name):
            return f"{project_root_path}/{path}"
    return path


# write function that takes a dictionary as a parameter and through recursion goes through all the values and if they are strings, it checks if they start with any of the dirs in project_dirs and if they do, it prepends the project_root_path to it
def resolve_project_paths(config, path_delimiter="path"):
    for key, value in config.items():
        if path_delimiter in key:
            config[key] = resolve_project_path(value)
        if isinstance(value, dict):
            resolve_project_paths(value)
        elif isinstance(value, list):
            for c in value:
                if isinstance(c, dict):
                    resolve_project_paths(c)
    return config


def get_feature_types():
    type_dict = {}
    type_dict[Features.JET_NUM.value] = "i8"
    type_dict[Features.JET_NBTAGS.value] = "i8"
    type_dict[Features.JET_BTAG.value] = "var * int8"
    type_dict[Features.JET_DL1DV01_PB.value] = "var * float32"
    type_dict[Features.JET_DL1DV01_PC.value] = "var * float32"
    type_dict[Features.JET_DL1DV01_PU.value] = "var * float32"
    type_dict[Features.JET_BTAG_GN2V01_PB.value] = "var * float32"
    type_dict[Features.JET_BTAG_GN2V01_PC.value] = "var * float32"
    type_dict[Features.JET_BTAG_GN2V01_PU.value] = "var * float32"
    type_dict[Features.JET_PT.value] = "var * float32"
    type_dict[Features.JET_ETA.value] = "var * float32"
    type_dict[Features.JET_PHI.value] = "var * float32"
    type_dict[Features.JET_MASS.value] = "var * float32"
    type_dict[Features.JET_PX.value] = "var * float32"
    type_dict[Features.JET_PY.value] = "var * float32"
    type_dict[Features.JET_PZ.value] = "var * float32"
    type_dict[Features.EVENT_M_4B.value] = "f4"
    type_dict[Features.EVENT_BB_RMH.value] = ("f8", 6)
    type_dict[Features.EVENT_BB_DR.value] = ("f8", 6)
    type_dict[Features.EVENT_BB_DETA.value] = ("f8", 6)
    type_dict[Features.EVENT_DELTAETA_HH.value] = "f4"
    type_dict[Features.EVENT_X_WT.value] = "f4"
    type_dict[Features.EVENT_X_HH.value] = "f4"
    type_dict[Features.EVENT_WEIGHT.value] = "float64"
    type_dict[Features.MC_EVENT_WEIGHT.value] = "float64"
    type_dict[Features.EVENT_NUMBER.value] = "i8"
    type_dict[Labels.LABEL_HH.value] = "i4"
    type_dict[Labels.LABEL_TTBAR.value] = "i4"
    type_dict[Labels.LABEL_QCD.value] = "i4"
    return type_dict


def write_out_h5(sample_output, sample_name, output_name):
    """Writes the sample_output to an hdf5 file with the sample_name as the key."""
    output_df = ak.to_dataframe(sample_output, how="outer")
    output_df.to_hdf(output_name, key=sample_name, mode="w")


def write_out_root(sample_output, sample_name, output_name):
    """Writes the sample_output to a root file with the sample_name as the tree name."""
    with uproot.recreate(output_name) as f:
        type_dict = get_feature_types()
        f.mktree(sample_name, type_dict)
        f[sample_name].extend(
            {field: sample_output[field] for field in sample_output.fields}
        )


def make_4jet_comb_array(a, op):
    fourpairs = list(itertools.combinations(range(4), 2))
    return np.transpose(ak.Array(op(a[:, i], a[:, j]) for i, j in fourpairs))


def get_common(lst1, lst2):
    """Get the common elements between two lists."""
    return list(set(lst1) & set(lst2))


def get_jet_branch_alias_names(aliases):
    jet_alias_names = list(filter(lambda alias: "jet_" in alias, aliases))
    return jet_alias_names


def get_legend_label(sample_id, sample_dict=None, prefix=None, postfix=None):
    label = sample_dict[sample_id] if sample_dict else sample_id
    label = label if not prefix else prefix + label
    label = label if not postfix else label + postfix
    return label


def bottom_offset(self, bboxes, bboxes2):
    top = self.axes.bbox.ymax
    self.offsetText.set(va="top", ha="left")
    self.offsetText.set_position((0, top + 25))


def register_bottom_offset(axis, func=bottom_offset):
    axis._update_offset_text_position = types.MethodType(func, axis)


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


def merge_sample_files(
    inputs,
    hists=None,
    save_to="merged_histograms.h5",
    merge_jz_regex=None,
    merge_mc_regex=False,
):
    """Merges histograms from multiple h5 files into a single dictionary."""

    # check that file doesn't already exist
    if hists is None and Path(save_to).exists():
        # print warning and ask user if they want to overwrite
        overwrite = builtins.input(
            f"Output file '{save_to}' already exists, do you want to overwrite it? (N/y) "
        )
        if overwrite == "" or overwrite.lower() == "n":
            save_to = None

    _hists = (
        hists
        if hists is not None
        else defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    )
    # checks if JZ0-9 is in the sample name
    for input in inputs:
        if input.is_dir():
            files_in_dir = input.glob("*.h5")
            merge_sample_files(
                files_in_dir, _hists, save_to, merge_jz_regex, merge_mc_regex
            )
            continue
        with h5py.File(input, "r") as hists_file:
            for sample_name in hists_file:
                if merge_jz_regex and merge_jz_regex.search(sample_name):
                    merged_sample_name = "_".join(sample_name.split("_")[:-2])
                else:
                    merged_sample_name = sample_name
                if merge_mc_regex and merge_mc_regex.search(merged_sample_name):
                    merged_sample_name = re.sub(r"[ade]", "", merged_sample_name)
                else:
                    merged_sample_name = merged_sample_name
                for hist_name in hists_file[sample_name]:
                    hist_edges = hists_file[sample_name][hist_name]["edges"][:]
                    hist_values = hists_file[sample_name][hist_name]["values"][:]
                    _hists[merged_sample_name][hist_name]["edges"] = hist_edges
                    if _hists[merged_sample_name][hist_name]["values"] is None:
                        _hists[merged_sample_name][hist_name]["values"] = hist_values
                    else:
                        _hists[merged_sample_name][hist_name]["values"] += hist_values
                    if "errors" in hists_file[sample_name][hist_name]:
                        hist_errors = hists_file[sample_name][hist_name]["errors"][:]
                        if _hists[merged_sample_name][hist_name]["errors"] is None:
                            _hists[merged_sample_name][hist_name][
                                "errors"
                            ] = hist_errors
                        else:
                            _hists[merged_sample_name][hist_name]["errors"] = (
                                propagate_errors(
                                    _hists[merged_sample_name][hist_name]["errors"],
                                    hist_errors,
                                    operation="+",
                                )
                            )

    if save_to:
        save_to_h5(_hists, name=save_to)

    return _hists
