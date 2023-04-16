#!/usr/bin/env python3

"""
build plots of everything
"""

import uproot
import numpy as np
import argparse
import json
import time
import coloredlogs, logging
from pathlib import Path
import concurrent.futures

# Package modules
from src.nonresonantresolved.drawhists import draw_hists
from src.nonresonantresolved.inithists import init_hists
from src.nonresonantresolved.fillhists import fill_hists
from src.nonresonantresolved.branches import get_branch_aliases
from shared.utils import (
    logger,
    concatenate_cutbookkeepers,
    get_luminosity_weight,
    get_datasetname_query,
)
from shared.api import get_metadata

np.seterr(divide="ignore", invalid="ignore")


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
    if args.loglevel:
        logger.setLevel(args.loglevel)
    coloredlogs.install(level=logger.level, logger=logger)
    with open(args.config) as cf:
        config = json.load(cf)
    hists = init_hists(config["inputs"], args)
    if logger.level == logging.DEBUG:
        starttime = time.time()
    branch_aliases = get_branch_aliases(args.signal)
    branch_names = branch_aliases.keys()
    for sample_name, sample_path in config["inputs"].items():
        datasetname_query = ""
        luminosity_weight = 1
        for events, report in uproot.iterate(
            f"{sample_path}*.root:AnalysisMiniTree",
            branch_names,
            aliases=branch_aliases,
            step_size="1 GB",
            report=True,
            decompression_executor=concurrent.futures.ThreadPoolExecutor(8 * 32),
            interpretation_executor=concurrent.futures.ThreadPoolExecutor(8 * 32),
        ):
            logger.info(f"Batch: {report}")
            if logger.level == logging.DEBUG:
                logger.debug("Columns: /n")
                events.type.show()

            current_datasetname_query = get_datasetname_query(report.file_path)
            sample_is_data = (
                "data" in sample_name or "data" in current_datasetname_query
            )
            if current_datasetname_query != datasetname_query and not sample_is_data:
                datasetname_query = current_datasetname_query
                metadata = get_metadata(datasetname_query)
                logger.debug(f"Metadata: {metadata}")
                _, sum_weights, _ = concatenate_cutbookkeepers(
                    sample_path, report.file_path
                )
                luminosity_weight = get_luminosity_weight(metadata, sum_weights)
            logger.debug(f"Luminosity weight: {luminosity_weight}")

            # fill the histograms with batch
            fill_hists(
                events,
                hists[sample_name],
                luminosity_weight,
                args,
            )

    if logger.level == logging.DEBUG:
        logger.debug(
            f"Loading data & filling histograms execution time: {time.time() - starttime} seconds"
        )

    draw_hists(hists, config["inputs"], args)


if __name__ == "__main__":
    main()
