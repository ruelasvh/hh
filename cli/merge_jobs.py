#!/usr/bin/env python3

"""
Merge jobs processed with HTCondor.
"""

import os
import json
import argparse
import htcondor
from htcondor import dags
from pathlib import Path
from hh.shared.utils import root_output_merger, h5_output_merger


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        default=[],
        help="List of paths to output directories.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file name postfix (default: %(default)s)",
        metavar="",
    )
    parser.add_argument(
        "-j",
        "--split-jz",
        action="store_true",
        help="Split JZ0-9 samples (default: %(default)s)",
    )

    return parser.parse_known_args()


def main():
    args, restargs = get_args()
    # get list of files in all input directories
    files = []
    for input_dir in args.inputs:
        files.extend([str(f) for f in Path(input_dir).rglob("*") if f.is_file()])
    # check if the files are root or h5
    if files[0].endswith(".root"):
        root_output_merger(files, args.output)
    elif files[0].endswith(".h5"):
        h5_output_merger(
            files,
            args.output,
            merge_jz_regex=None if args.split_jz else re.compile(r"JZ[0-9]"),
        )
    else:
        raise ValueError("Unknown file format.")


if __name__ == "__main__":
    main()
