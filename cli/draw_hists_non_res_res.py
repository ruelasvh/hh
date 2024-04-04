#!/usr/bin/env python3

import re
import h5py
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from hh.shared.utils import logger, setup_logger
from hh.nonresonantresolved import drawhistsdiagnostics
from hh.nonresonantresolved import drawhistsbkgest


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
        help="Luminosity in fb^-1",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-e",
        "--energy",
        help="Center of mass energy in TeV",
        type=float,
        default=13,
    )
    parser.add_argument(
        "-w",
        "--bkg-weight",
        help="Bkg estimation weight",
        type=float,
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


def merge_sample_files(inputs, hists=None):
    _hists = (
        hists
        if hists is not None
        else defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.array([]))))
    )
    # checks if JZ0-9 is in the sample name
    jz_regex = re.compile(r"JZ[0-9]")
    for input in inputs:
        if input.is_dir():
            files_in_dir = input.glob("*.h5")
            merge_sample_files(files_in_dir, _hists)
            continue
        with h5py.File(input, "r") as hists_file:
            for sample_name in hists_file:
                if jz_regex.search(sample_name):
                    merged_sample_name = (
                        "_".join(sample_name.split("_")[:-2]) + "_multijet"
                    )
                    merged_sample_name = merged_sample_name.replace(
                        "multijet_multijet", "multijet"
                    )
                else:
                    merged_sample_name = sample_name
                for hist_name in hists_file[sample_name]:
                    hist_edges = hists_file[sample_name][hist_name]["edges"][:]
                    hist_values = hists_file[sample_name][hist_name]["values"][:]
                    _hists[merged_sample_name][hist_name]["edges"] = hist_edges
                    if (
                        hist_values.shape
                        != _hists[merged_sample_name][hist_name]["values"].shape
                    ):
                        _hists[merged_sample_name][hist_name]["values"] = hist_values
                    else:
                        _hists[merged_sample_name][hist_name]["values"] += hist_values
    return _hists


def main():
    args = get_args()

    if args.loglevel:
        setup_logger(args.loglevel)

    # check if output_dir exists, if not create it
    if args.output_dir.exists():
        logger.warning(f"Output directory '{args.output_dir}' already exists")
        exit(1)
    else:
        args.output_dir.mkdir(parents=True)
        logger.info(f"Saving plots to '{args.output_dir}'")

    hists = merge_sample_files(args.hists_files)

    if args.bkg_weight:
        drawhistsbkgest.draw_hists(hists, args)
    else:
        drawhistsdiagnostics.draw_hists(hists, args)


if __name__ == "__main__":
    main()
