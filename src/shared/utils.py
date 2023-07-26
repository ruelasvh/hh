import uproot
import glob
import re
import logging
import operator
from functools import reduce
import pandas as pd


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
