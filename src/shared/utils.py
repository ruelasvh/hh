import uproot
import glob
import re
import logging
import operator
import itertools
import numpy as np
from pathlib import Path
from functools import reduce
import awkward as ak
from src.dump.output import Features, Labels


logger = logging.getLogger("plot-hh4b-analysis")

nth = {1: "first", 2: "second", 3: "third", 4: "fourth"}

kin_labels = {"pt": r"$p_T$", "eta": r"$\eta$", "phi": r"$\phi$", "mass": r"$m$"}

GeV = 1_000

inv_GeV = 1 / GeV


def get_com_lumi_label(lumi, com=13.6):
    com_label = r"$\sqrt{s} = \mathrm{" + str(com) + "\ TeV}"
    lumi_label = (
        ",\ " + str(format(lumi, ".1f")) + r"\ \mathrm{fb}^{-1}$" if lumi else "$"
    )
    return com_label + lumi_label


def concatenate_cutbookkeepers(files, file_delimeter=None):
    if isinstance(files, list):
        _files = files
    else:
        _dirs = glob.glob(files)
        _dir = (
            files
            if not _dirs
            else list(filter(lambda _dir: _dir in file_delimeter, _dirs))[0]
        )
        # _files = glob.glob(f"{_dir}/*.root")
        _files = glob.glob(f"{_dir}*.root")

    cutbookkeepers = {}
    cutbookkeepers["initial_events"] = 0
    cutbookkeepers["initial_sum_of_weights"] = 0
    cutbookkeepers["initial_sum_of_weights_squared"] = 0
    for file_path in _files:
        with uproot.open(file_path) as file:
            for key in file.keys():
                if "CutBookkeeper" and "NOSYS" in key:
                    cbk = file[key].to_numpy()
                    cutbookkeepers["initial_events"] += cbk[0][0]
                    cutbookkeepers["initial_sum_of_weights"] += cbk[0][1]
                    cutbookkeepers["initial_sum_of_weights_squared"] += cbk[0][2]
    return cutbookkeepers


def get_total_weight(metadata, sum_weights=1.0):
    filter_efficiency = float(metadata["genFiltEff"])
    k_factor = float(metadata["kFactor"])
    cross_section = float(metadata["crossSection"]) * 1e6
    luminosity = float(metadata["luminosity"])
    return filter_efficiency * k_factor * cross_section * luminosity * sum_weights**-1


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


def get_all_trigs_or(events, trigs, skip_trig=None):
    """Returns the OR decision of all trigs except skip_trig"""
    trigs = list(filter(lambda trig: trig != skip_trig, trigs))
    return reduce(
        lambda acc, it: acc | events[it],
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


def concatenate_datasets(processed_batch, out, is_mc):
    # concatenate datasets. Processed batch is an Akward array of events. Each event in an Awkward record. Second argument is also an Awkward array of events. Each event is an Awkward record. Concatenate the two arrays and return the concatenated array
    return ak.concatenate([processed_batch, out])


def write_out(sample_output, sample_name, output_name):
    with uproot.recreate(output_name) as f:
        # Declare all branches
        f.mktree(
            sample_name,
            {
                Features.JET_NUM.value: "i4",
                Features.JET_NBTAGS.value: "i4",
                Features.JET_BTAG.value: "var * int8",
                Features.JET_PT.value: "var * float32",
                Features.JET_ETA.value: "var * float32",
                Features.JET_PHI.value: "var * float32",
                Features.JET_MASS.value: "var * float32",
                Features.JET_X.value: "var * float32",
                Features.JET_Y.value: "var * float32",
                Features.JET_Z.value: "var * float32",
                Features.EVENT_M_4B.value: "f4",
                Features.EVENT_BB_RMH.value: ("f8", 6),
                Features.EVENT_BB_DR.value: ("f8", 6),
                Features.EVENT_BB_DETA.value: ("f8", 6),
                Features.EVENT_MCWEIGHT.value: "float32",
                Features.EVENT_PUWEIGHT.value: "float32",
                Features.EVENT_XWEIGHT.value: "float32",
                Labels.LABEL_HH.value: "i4",
                Labels.LABEL_TTBAR.value: "i4",
                Labels.LABEL_QCD.value: "i4",
            },
        )
        f[sample_name].extend(
            {field: sample_output[field] for field in sample_output.fields}
        )


def make_4jet_comb_array(a, op):
    fourpairs = list(itertools.combinations(range(4), 2))
    return np.transpose(ak.Array(op(a[:, i], a[:, j]) for i, j in fourpairs))
