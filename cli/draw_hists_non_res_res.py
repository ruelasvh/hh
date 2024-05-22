#!/usr/bin/env python3

import re
import time
import h5py
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from hh.shared.error import propagate_errors
from hh.shared.utils import logger, setup_logger
from hh.nonresonantresolved import drawhistsdiagnostics


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hists_files", type=Path, nargs="+", help="Input files")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default="plots",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--btag",
        help="btag category",
        type=str,
        default=None,
        choices=["4b", "3b", "2b"],
    )
    parser.add_argument(
        "-l",
        "--luminosity",
        type=float,
        default=1.0,
        help="Luminosity in fb^-1 (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--energy",
        type=float,
        default=13.0,
        help="Center of mass energy in TeV (default: %(default)s)",
    )
    # add bool argument to merge JZ0-9 samples
    parser.add_argument(
        "-j",
        "--split-jz",
        action="store_true",
        help="Split JZ0-9 samples (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print info statements",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    return parser.parse_args()


def merge_sample_files(inputs, hists=None, merge_jz_regex=None):
    _hists = (
        hists
        if hists is not None
        else defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    )
    # checks if JZ0-9 is in the sample name
    for input in inputs:
        if input.is_dir():
            files_in_dir = input.glob("*.h5")
            merge_sample_files(files_in_dir, _hists, merge_jz_regex)
            continue
        with h5py.File(input, "r") as hists_file:
            for sample_name in hists_file:
                if merge_jz_regex and merge_jz_regex.search(sample_name):
                    merged_sample_name = "_".join(sample_name.split("_")[:-2])
                else:
                    merged_sample_name = sample_name
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

    return _hists


def main():
    starttime = time.time()

    args = get_args()

    if args.loglevel:
        setup_logger(args.loglevel)

    # Ask user if they want to overwrite the plots output directory, if not exit
    if args.output_dir.exists():
        overwrite = input(
            f"Output directory '{args.output_dir}' already exists, do you want to overwrite it? (y/n) "
        )
        if overwrite.lower() != "y":
            exit(0)
        else:
            shutil.rmtree(args.output_dir)
            args.output_dir.mkdir(parents=True)
    else:
        args.output_dir.mkdir(parents=True)
        logger.info(f"Saving plots to '{args.output_dir}'")

    hists = merge_sample_files(
        args.hists_files,
        merge_jz_regex=None if args.split_jz else re.compile(r"JZ[0-9]"),
    )

    drawhistsdiagnostics.draw_hists(hists, args)

    if logger.level == logging.DEBUG:
        logger.debug(
            f"Loading data & processing events execution time: {time.time() - starttime} seconds"
        )


if __name__ == "__main__":
    main()
