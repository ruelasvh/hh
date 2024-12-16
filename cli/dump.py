#!/usr/bin/env python3

"""
Dumps the analysis variables from the ntuples.
"""

import os
import json
import time
import uproot
import logging
import argparse
import awkward as ak
from pathlib import Path
from hh.dump.processbatches import (
    process_batch,
)
from hh.dump.output import Features, Labels
from hh.nonresonantresolved.branches import (
    get_branch_aliases,
)
from hh.shared.utils import (
    logger,
    setup_logger,
    concatenate_cutbookkeepers,
    get_sample_weight,
    resolve_project_paths,
    write_out_h5,
    write_out_root,
    write_out_parquet,
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
    sample_id: int,
    sample_name: str,
    sample_path: Path,
    sample_metadata: list,
    selections: dict,
    class_label: str,
    features: dict,
    args: argparse.Namespace,
) -> None:
    is_mc = sample_metadata["isMC"]
    sample_metadata["label"] = class_label

    # get the sample weight
    sample_weight = 1.0
    if is_mc:
        initial_sum_of_weights = sample_metadata.get("initialSumWeights", None)
        if initial_sum_of_weights is None:
            cbk = concatenate_cutbookkeepers(sample_path)
            initial_sum_of_weights = cbk["initial_sum_of_weights"]
        sample_weight = get_sample_weight(sample_metadata, initial_sum_of_weights)

    trig_set = None
    if "trigs" in selections:
        trig_set = selections["trigs"]["value"]
    branch_aliases = get_branch_aliases(is_mc, trig_set, sample_metadata)

    out = []
    cutflow = {}
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
        # select analysis events, calculate analysis variables (e.g. X_hh, deltaEta_hh, X_Wt) and fill the histograms
        processed_batch, cutflow = process_batch(
            batch_events,
            selections,
            features,
            class_label,
            sample_weight,
            is_mc,
            cutflow,
        )
        if len(processed_batch) == 0:
            continue
        logger.info(f"Merging batches for sample: {sample_name}")
        out.append(processed_batch)

    out = ak.concatenate(out)
    out_filename_stem = args.output.with_name(
        f"{args.output.stem}_{sample_name}_{sample_id}_{os.getpgid(os.getpid())}"
    )
    if args.output.suffix == ".h5":
        out_filename = out_filename_stem.with_suffix(".h5")
        logger.info(f"Writing {sample_name} to {out_filename}")
        write_out_h5(out, sample_name, out_filename)
    elif args.output.suffix == ".root":
        out_filename = out_filename_stem.with_suffix(".root")
        logger.info(f"Writing {sample_name} to {out_filename}")
        write_out_root(out, sample_name, out_filename)
    elif args.output.suffix == ".parquet":
        out_filename = out_filename_stem.with_suffix(".parquet")
        logger.info(f"Writing {sample_name} to {out_filename}")
        write_out_parquet(
            processed_batch,
            sample_name,
            out_filename,
            metadata=sample_metadata,
            cutflow=cutflow,
        )
    else:
        raise ValueError(f"Invalid output file format: {args.output.suffix}")


def main():
    starttime = time.time()

    args = get_args()

    if args.loglevel:
        setup_logger(args.loglevel)

    with open(args.config) as cf:
        config = resolve_project_paths(config=json.load(cf))

    if "path" in config["features"]:
        with open(config["features"]["path"]) as ff:
            config["features"] = json.load(ff)

    assert Features.contains_all(
        config["features"]["features"]
    ), f"Invalid features: {set(config['features']['features']) - set(Features.get_all())}"

    assert Labels.contains_all(
        config["features"]["labels"]
    ), f"Invalid labels: {set(config['features']['labels']) - set(Labels.get_all())}"

    samples, features = config["samples"], config["features"]

    worker_items = [
        (
            sample_id,
            sample["label"],
            sample["paths"][sample_id],
            sample["metadata"][sample_id],
            sample["selections"],
            sample["class_label"],
            features,
            args,
        )
        for sample in samples
        for sample_id in range(len(sample["paths"]))
    ]

    for worker_item in worker_items:
        process_sample_worker(*worker_item)

    if logger.level == logging.DEBUG:
        logger.debug(
            f"Loading data & processing events execution time: {time.time() - starttime} seconds"
        )


if __name__ == "__main__":
    main()
