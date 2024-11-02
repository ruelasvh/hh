#!/usr/bin/env python3

import re
import time
import shutil
import logging
import argparse
from pathlib import Path
from hh.nonresonantresolved import drawhistsdiagnostics
from hh.shared.utils import logger, setup_logger, merge_sample_files


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
        default=None,
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
    # add bool argument to merge mc campaigns
    parser.add_argument(
        "-m",
        "--split-mc",
        action="store_true",
        help="Split mc campaigns (default: %(default)s)",
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


def main():
    starttime = time.time()

    args = get_args()

    if args.loglevel:
        setup_logger(args.loglevel)

    # Ask user if they want to overwrite the plots output directory, if not exit
    if args.output_dir.exists():
        overwrite = input(
            f"Output directory '{args.output_dir}' already exists, do you want to overwrite it? (Y/n) "
        )
        if overwrite == "" or overwrite.lower() == "y":
            shutil.rmtree(args.output_dir)
            args.output_dir.mkdir(parents=True)
        else:
            exit(0)

    else:
        args.output_dir.mkdir(parents=True)
        logger.info(f"Saving plots to '{args.output_dir}'")

    hists = merge_sample_files(
        args.hists_files,
        save_to="merged_histograms.h5",
        merge_jz_regex=None if args.split_jz else re.compile(r"jz[0-9]", re.IGNORECASE),
        merge_mc_regex=(
            None if args.split_mc else re.compile(r"mc[1-2][0-9][ade]", re.IGNORECASE)
        ),
    )

    drawhistsdiagnostics.draw_hists(hists, args)

    if logger.level == logging.DEBUG:
        logger.debug(
            f"Loading data & processing events execution time: {time.time() - starttime} seconds"
        )


if __name__ == "__main__":
    main()
