#!/usr/bin/env python3

import h5py
import argparse
import logging
from pathlib import Path
from src.nonresonantresolved.drawhists import draw_hists


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
        default="4b",
        choices=["4b", "3b", "2b"],
    )
    parser.add_argument(
        "-l",
        "--luminosity",
        help="Luminosity in fb^-1",
        type=float,
        # default=1.12552,
    )
    parser.add_argument(
        "-p",
        "--plots-postfix",
        help="Postfix to save plots",
        type=str,
        default="baseline",
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
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    return parser.parse_args()


# def main():
#     args = get_args()
#     hists = {}
#     for file in args.hists_files:
#         # append hists from each file to hists
#         with h5py.File(file, "r") as hists_file:
#             hists = {**hists, **hists_file}

#     draw_hists(hists, args.luminosity, args.btag, args.plots_postfix, args.output_dir)


def main():
    args = get_args()
    for file in args.hists_files:
        with h5py.File(file, "r") as hists:
            draw_hists(
                hists,
                args.luminosity,
                args.btag,
                args.plots_postfix,
                args.output_dir.with_name(f"{args.output_dir.stem}_{file.stem}"),
            )


if __name__ == "__main__":
    main()
