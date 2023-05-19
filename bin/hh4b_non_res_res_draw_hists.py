#!/usr/bin/env python3

from h5py import File
import argparse
import logging
from pathlib import Path
from src.nonresonantresolved.drawhists import draw_hists


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hists_path", type=Path)
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
    with File(args.hists_path, "r") as hists:
        draw_hists(hists)


if __name__ == "__main__":
    main()
