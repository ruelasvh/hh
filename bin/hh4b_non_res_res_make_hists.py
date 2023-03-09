#!/usr/bin/env python3

"""
build plots of everything
"""

import uproot
import numpy as np
import argparse
import json
import time
import logging
from pathlib import Path
import concurrent.futures

# Package modules
from src.nonresonantresolved.selection import select_n_jets_events, select_X_Wt_events
from src.nonresonantresolved.drawhists import draw_hists
from src.nonresonantresolved.inithists import init_hists
from src.nonresonantresolved.fillhists import fill_hists
from src.nonresonantresolved.branches import (
    names as branch_names,
    aliases as branch_aliases,
)

np.seterr(divide="ignore", invalid="ignore")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(help="default: %(default)s")
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("hists.h5"),
        **defaults,
    )
    parser.add_argument(
        "-s",
        "--signal",
        action="store_true",
        help="sample is signal, process truth info",
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
    logging.basicConfig(level=args.loglevel)
    with open(args.input) as inputfile:
        input = json.load(inputfile)
    hists = init_hists(input, args)
    if args.loglevel == logging.DEBUG:
        starttime = time.time()
    for sample_name, sample_path in input.items():
        for events, report in uproot.iterate(
            f"{sample_path}*.root:AnalysisMiniTree",
            branch_names,
            aliases=branch_aliases,
            step_size="1 GB",
            report=True,
            decompression_executor=concurrent.futures.ThreadPoolExecutor(8 * 32),
            interpretation_executor=concurrent.futures.ThreadPoolExecutor(8 * 32),
        ):
            logging.info(report)
            if args.loglevel == logging.DEBUG:
                events.type.show()

            four_central_jets_selected_events = select_n_jets_events(
                events, eta_cut=2.5, njets=4
            )
            fill_hists(
                four_central_jets_selected_events,
                hists[sample_name],
                args,
                uncut_events=events,
            )
    if args.loglevel == logging.DEBUG:
        logging.debug(f"Execution time: {time.time() - starttime} seconds")

    for sample_name in input.keys():
        draw_hists(hists[sample_name], sample_name, args)


if __name__ == "__main__":
    main()
