#!/usr/bin/env python3

"""
Dumps the analysis variables from the ntuples.
"""

import os
import json
import time
import uproot
import argparse
import awkward as ak
from pathlib import Path
import coloredlogs, logging

from hh.dump.processbatches import (
    process_batch,
)
from hh.dump.output import Features, Labels
from hh.nonresonantresolved.branches import (
    get_branch_aliases,
)
from hh.shared.utils import (
    logger,
    concatenate_cutbookkeepers,
    get_sample_weight,
    resolve_project_paths,
    concatenate_datasets,
    write_out,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(help="default: %(default)s")
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("train.root"),
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
    class_label: str,
    features: dict,
    output: dict,
    args: argparse.Namespace,
) -> None:
    is_mc = "data" not in sample_name
    trig_set = None
    if "events" in selections and "trigs" in selections["events"]:
        trig_set = selections["events"]["trigs"]["value"]
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
        allow_missing=False,
        report=True,
    ):
        logger.info(f"Processing batch: {batch_report}")
        logger.debug(f"Columns: {batch_events.fields}")
        # select analysis events, calculate analysis variables (e.g. X_hh, deltaEta_hh, X_Wt) and fill the histograms
        processed_batch = process_batch(
            batch_events,
            selections,
            features,
            class_label,
            sample_weight,
            is_mc,
        )
        if len(processed_batch) == 0:
            continue

        logger.info(f"Merging batches for sample: {sample_name}")
        # NOTE: important: copy the out back (otherwise parent process won't see the changes)
        output[sample_name] = concatenate_datasets(processed_batch, output[sample_name])
        output_name = args.output.with_name(
            f"{args.output.stem}_{sample_name}_{os.getpgid(os.getpid())}.root"
        )
        logger.info(f"Writing {sample_name} to {output_name}")
        write_out(output[sample_name], sample_name, output_name)


def main():
    starttime = time.time()

    args = get_args()

    if args.loglevel:
        logger.setLevel(args.loglevel)
        coloredlogs.install(level=logger.level, logger=logger)

    with open(args.config) as cf:
        config = resolve_project_paths(config=json.load(cf))

    if "path" in config["features"]:
        with open(config["features"]["path"]) as ff:
            config["features"] = json.load(ff)

    assert Features.contains_all(
        config["features"]["out"]
    ), f"Invalid features: {set(config['features']['out']) - set(Features.get_all())}"

    assert Labels.contains_all(
        config["features"]["classes"]
    ), f"Invalid labels: {set(config['features']['classes']) - set(Labels.get_all())}"

    samples, features = config["samples"], config["features"]
    output = {sample["label"]: ak.Array([]) for sample in samples}

    worker_items = [
        (
            sample["label"],
            sample_path,
            sample["metadata"][idx],
            sample["selections"],
            sample["class_label"],
            features,
            output,
            args,
        )
        for sample in samples
        for idx, sample_path in enumerate(sample["paths"])
    ]

    for worker_item in worker_items:
        process_sample_worker(*worker_item)

    if logger.level == logging.DEBUG:
        logger.debug(
            f"Loading data & processing events execution time: {time.time() - starttime} seconds"
        )


if __name__ == "__main__":
    main()
