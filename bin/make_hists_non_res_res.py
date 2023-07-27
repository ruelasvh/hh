#!/usr/bin/env python3

"""
build plots of everything
"""

import uproot
import argparse
import json
import time
import coloredlogs, logging
from pathlib import Path
import h5py
import multiprocessing
import os

from src.nonresonantresolved.inithists import init_hists
from src.nonresonantresolved.branches import (
    get_branch_aliases,
)
from src.nonresonantresolved.processbatches import (
    process_batch,
)
from src.nonresonantresolved.fillhists import fill_hists
from src.shared.utils import (
    logger,
    concatenate_cutbookkeepers,
    get_total_weight,
    write_hists,
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
        "-r",
        "--run",
        type=int,
        default=2,
        choices=[2, 3],
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
        "-j",
        "--jobs",
        default=1,
        type=lambda x: min(os.cpu_count(), max(1, int(x))),
        help=f"Number of jobs to run in parallel (default: 1, max: {os.cpu_count()})",
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
    event_selection: dict,
    hists: list,
    args: argparse.Namespace,
) -> None:
    is_mc = "data" not in sample_name
    branch_aliases = get_branch_aliases(is_mc, args.run)
    total_weight = 1.0
    current_file_path = ""
    for batch_events, batch_report in uproot.iterate(
        f"{sample_path}*.root:AnalysisMiniTree",
        expressions=branch_aliases.keys(),
        aliases=branch_aliases,
        num_workers=args.jobs * 2,
        step_size=args.batch_size,
        allow_missing=True,
        report=True,
    ):
        logger.info(f"Processing batch: {batch_report}")
        logger.debug(f"Columns: {batch_events.fields}")
        if (current_file_path != batch_report.file_path) and is_mc:
            current_file_path = batch_report.file_path
            # concatenate cutbookkeepers for each sample
            cbk = concatenate_cutbookkeepers(sample_path, batch_report.file_path)
            logger.debug(f"Metadata: {sample_metadata}")
            total_weight = get_total_weight(
                sample_metadata, cbk["initial_sum_of_weights"]
            )
        logger.debug(f"Total weight: {total_weight}")
        # select analysis events, calculate analysis variables (e.g. X_hh, deltaEta_hh, X_Wt) and fill the histograms
        processed_batch = process_batch(
            batch_events,
            event_selection,
            total_weight,
            is_mc,
        )
        # check if in multiprocessing context
        if args.jobs > 1:
            with multiprocessing.Lock():
                # NOTE: important: copy the hists back (otherwise parent process won't see the changes)
                hists[sample_name] = fill_hists(
                    processed_batch, hists[sample_name], is_mc
                )
                output_name = args.output.with_name(
                    f"{args.output.stem}_{sample_name}_{os.getpgid(os.getpid())}.h5"
                )
                with h5py.File(output_name, "w") as output_file:
                    logger.info(f"Saving histograms to file: {output_name}")
                    write_hists(hists[sample_name], sample_name, output_file)
        else:
            hists[sample_name] = fill_hists(processed_batch, hists[sample_name], is_mc)
            output_name = args.output.with_name(f"{args.output.stem}_{sample_name}.h5")
            with h5py.File(output_name, "w") as output_file:
                logger.info(f"Saving histograms to file: {output_name}")
                write_hists(hists[sample_name], sample_name, output_file)


def main():
    starttime = time.time()

    args = get_args()

    if args.loglevel:
        logger.setLevel(args.loglevel)
        coloredlogs.install(level=logger.level, logger=logger)

    with open(args.config) as cf:
        config = json.load(cf)

    samples, event_selection = config["samples"], config["event_selection"]
    manager = multiprocessing.Manager()
    hists = manager.dict(init_hists(samples, args))
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
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            pool.starmap(process_sample_worker, worker_items)
    else:
        for worker_item in worker_items:
            process_sample_worker(*worker_item)

    if logger.level == logging.DEBUG:
        logger.debug(
            f"Loading data & filling histograms execution time: {time.time() - starttime} seconds"
        )


if __name__ == "__main__":
    main()
