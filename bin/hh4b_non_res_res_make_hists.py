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
from h5py import File

# Package modules
from src.nonresonantresolved.inithists import init_hists
from src.nonresonantresolved.cutflow import cut_flow
from src.nonresonantresolved.branches import (
    get_branch_aliases,
    add_default_branches_from_config,
)
from shared.utils import (
    logger,
    concatenate_cutbookkeepers,
    get_luminosity_weight,
    get_datasetname_query,
    write_hists,
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

    hists = init_hists(config["inputs"], args)
    branch_aliases = get_branch_aliases(args.signal)
    branch_aliases = add_default_branches_from_config(
        branch_aliases, config["event_selection"]
    )

    for input in config["inputs"]:
        sample_label, sample_path = input["label"], input["path"]
        datasetname_query = ""
        luminosity_weight = 1
        for events, report in uproot.iterate(
            f"{sample_path}*.root:AnalysisMiniTree",
            expressions=branch_aliases.keys(),
            aliases=branch_aliases,
            step_size="1 GB",
            report=True,
        ):
            logger.info(f"Batch: {report}")

            if logger.level == logging.DEBUG:
                logger.debug("Columns: /n")
                events.type.show()

            current_datasetname_query = get_datasetname_query(report.file_path)
            sample_is_data = (
                "data" in sample_label or "data" in current_datasetname_query
            )
            if current_datasetname_query != datasetname_query and not sample_is_data:
                datasetname_query = current_datasetname_query
                _, sum_weights, _ = concatenate_cutbookkeepers(
                    sample_path, report.file_path
                )
                metadata = get_metadata(datasetname_query)
                logger.debug(f"Metadata: {metadata}")
                luminosity_weight = get_luminosity_weight(metadata, sum_weights)
            logger.debug(f"Luminosity weight: {luminosity_weight}")

            # select events and fill the histograms with batch
            cut_flow(
                events,
                hists[sample_label],
                luminosity_weight,
                config,
                args,
            )

    if logger.level == logging.DEBUG:
        logger.debug(
            f"Loading data & filling histograms execution time: {time.time() - starttime} seconds"
        )

    logger.info("Saving histograms")
    with File(args.output, "w") as hout:
        write_hists(hists, hout)


if __name__ == "__main__":
    main()
