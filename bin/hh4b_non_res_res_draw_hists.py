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


def main():
    args = get_args()
    if len(args.hists_files) == 1:
        for file in args.hists_files:
            with h5py.File(file, "a") as hists:
                del hists["source"]
                draw_hists(hists, args.luminosity, args.btag, args.plots_postfix)
    elif len(args.hists_files) == 2:
        with h5py.File(args.hists_files[0], "r") as hists1, h5py.File(
            args.hists_files[1], "r"
        ) as hists2:
            hists = {**hists1, **hists2}
            draw_hists(hists, args.luminosity, args.btag, args.plots_postfix)
    else:
        print("Can only take at most two hists files")
        exit(1)


if __name__ == "__main__":
    main()
