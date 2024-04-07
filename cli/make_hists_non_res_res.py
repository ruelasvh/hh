#!/usr/bin/env python3

"""
build plots of everything
"""

import os
import json
import time
import h5py
import uproot
import argparse
import coloredlogs, logging
from pathlib import Path

from hh.nonresonantresolved.inithists import init_hists
from hh.nonresonantresolved.branches import (
    get_branch_aliases,
)
from hh.nonresonantresolved.processbatches import (
    process_batch,
)
from hh.nonresonantresolved.fillhists import fill_hists
from hh.shared.utils import (
    logger,
    setup_logger,
    concatenate_cutbookkeepers,
    get_sample_weight,
    write_hists,
    resolve_project_paths,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(help="default: %(default)s")
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("hists.h5"),
        **defaults,
    )
    parser.add_argument(
        "-w",
        "--sample-weight",
        type=float,
        default=None,
        **defaults,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=lambda x: int(x) if x.isdigit() else x,
        default="100 MB",
        **defaults,
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        default=3,
        type=lambda x: min(os.cpu_count(), max(1, int(x))),
        help=f"Number of workers for reading files (max: {os.cpu_count()})",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="print info statements",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    return parser.parse_args()


def process_sample_worker(
    sample_name: str,
    sample_path: Path,
    sample_metadata: list,
    selections: dict,
    hists: list,
    args: argparse.Namespace,
) -> None:
    is_mc = "data" not in sample_name
    trig_set = None
    if "trigs" in selections:
        trig_set = selections["trigs"].get("value")
    branch_aliases = get_branch_aliases(is_mc, trig_set)
    if args.sample_weight is None and is_mc:
        cbk = concatenate_cutbookkeepers(sample_path)
        sample_weight = get_sample_weight(sample_metadata, cbk)
    else:
        sample_weight = 1.0 if args.sample_weight is None else args.sample_weight
    for batch_events, batch_report in uproot.iterate(
        f"{sample_path}*.root:AnalysisMiniTree",
        expressions=branch_aliases.keys(),
        aliases=branch_aliases,
        num_workers=args.num_workers,
        step_size=args.batch_size,
        allow_missing=True,
        report=True,
    ):
        logger.info(f"Processing batch: {batch_report}")
        logger.debug(f"Columns: {batch_events.fields}")
        # select analysis events, calculate analysis variables (e.g. X_hh, deltaEta_hh, X_Wt) and fill the histograms
        processed_batch = process_batch(
            batch_events,
            selections,
            sample_weight,
            is_mc,
        )
        # if no events pass the selection, skip filling histograms
        if len(processed_batch) == 0:
            continue
        # fill histograms
        fill_hists(processed_batch, hists[sample_name], selections, is_mc)
        # save histograms to file
        output_name = args.output.with_name(
            f"{args.output.stem}_{sample_name}_{os.getpgid(os.getpid())}.h5"
        )
        with h5py.File(output_name, "w") as output_file:
            logger.info(f"Saving histograms to file: {output_name}")
            write_hists(hists[sample_name], sample_name, output_file)


def main():
    starttime = time.time()

    args = get_args()

    if args.loglevel:
        setup_logger(args.loglevel)
        coloredlogs.install(level=logger.level, logger=logger)

    with open(args.config) as cf:
        config = resolve_project_paths(config=json.load(cf))

    samples, event_selection = config["samples"], config["event_selection"]
    hists = init_hists(samples, args)

    worker_items = [
        (
            sample["label"],
            sample_path,
            sample["metadata"][idx] if "metadata" in sample else None,
            event_selection,
            hists,
            args,
        )
        for sample in samples
        for idx, sample_path in enumerate(sample["paths"])
    ]
    for worker_item in worker_items:
        process_sample_worker(*worker_item)

    if logger.level == logging.DEBUG:
        logger.debug(
            f"Loading data & filling histograms execution time: {time.time() - starttime} seconds"
        )


if __name__ == "__main__":
    main()
