import uproot
import glob
import re
import logging
import operator


logger = logging.getLogger("plot-hh4b-analysis")


def concatenate_cutbookkeepers(files, file_delimeter=None):
    if isinstance(files, list):
        _files = files
    else:
        _dirs = glob.glob(files)
        _dir = list(filter(lambda _dir: _dir in file_delimeter, _dirs))[0]
        _files = glob.glob(f"{_dir}/*.root")

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
    return (
        cutbookkeepers["initial_events"],
        cutbookkeepers["initial_sum_of_weights"],
        cutbookkeepers["initial_sum_of_weights_squared"],
    )


def get_luminosity_weight(metadata, sum_weights=1.0):
    filter_efficiency = float(metadata["genFiltEff"])
    k_factor = float(metadata["kFactor"])
    cross_section = float(metadata["crossSection"]) * 1e6
    luminosity = float(metadata["luminosity"])

    return (
        sum_weights**-1 * filter_efficiency * k_factor * (cross_section) * luminosity
    )


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


def write_hists(group, output):
    for sample_name, sample_hists in group.items():
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
