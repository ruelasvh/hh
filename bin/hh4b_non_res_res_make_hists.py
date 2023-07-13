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
from h5py import File

# Package modules
from src.nonresonantresolved.inithists import init_hists
from src.nonresonantresolved.branches import (
    get_branch_aliases,
)
from shared.utils import (
    logger,
    concatenate_cutbookkeepers,
    get_total_weight,
    write_hists,
)
from nonresonantresolved.processbatches import (
    process_batch,
)
from src.nonresonantresolved.fillhists import fill_hists


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
        type=str,
        default="1 GB",
        **defaults,
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


def main():
    starttime = time.time()

    args = get_args()
    if args.loglevel:
        logger.setLevel(args.loglevel)
        coloredlogs.install(level=logger.level, logger=logger)

    with open(args.config) as cf:
        config = json.load(cf)

    hists = init_hists(config["samples"], args)
    for sample in config["samples"]:
        sample_label, sample_paths, sample_metadata = (
            sample["label"],
            sample["paths"],
            sample["metadata"],
        )
        is_mc = "data" not in sample_label
        branch_aliases = get_branch_aliases(is_mc, args.run)
        total_weight = 1.0
        current_file_path = ""
        for idx, sample_path in enumerate(sample_paths):
            for batch_events, batch_report in uproot.iterate(
                f"{sample_path}*.root:AnalysisMiniTree",
                expressions=branch_aliases.keys(),
                aliases=branch_aliases,
                step_size=args.batch_size,
                allow_missing=True,
                report=True,
            ):
                logger.info(f"Processing batch: {batch_report}")
                logger.debug(f"Columns: {batch_events.fields}")
                if (current_file_path != batch_report.file_path) and is_mc:
                    current_file_path = batch_report.file_path
                    # concatenate cutbookkeepers for each sample
                    cbk = concatenate_cutbookkeepers(
                        sample_path, batch_report.file_path
                    )
                    metadata = sample_metadata[idx]
                    logger.debug(f"Metadata: {metadata}")
                    total_weight = get_total_weight(
                        metadata, cbk["initial_sum_of_weights"]
                    )
                logger.debug(f"Total weight: {total_weight}")
                # select analysis events, calculate analysis variables (e.g. X_hh, deltaEta_hh, X_Wt) and fill the histograms
                processed_batch = process_batch(
                    batch_events,
                    config,
                    hists[sample_label],
                    total_weight,
                    is_mc,
                )
                fill_hists(processed_batch, hists[sample_label], is_mc)
                logger.info("Saving histograms")
                with File(args.output, "w") as hout:
                    write_hists(hists, hout)

    if logger.level == logging.DEBUG:
        logger.debug(
            f"Loading data & filling histograms execution time: {time.time() - starttime} seconds"
        )


if __name__ == "__main__":
    main()
