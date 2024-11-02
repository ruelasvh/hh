#!/usr/bin/env python3

"""
build plots of everything
"""
from __future__ import annotations

import os
import sys
import json
import time
import h5py
import uproot
import argparse
import logging
import numpy as np
import awkward as ak
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
    format_btagger_model_name,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(help="default: %(default)s")
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("resolved_nominal.parquet"),
        **defaults,
    )
    parser.add_argument(
        "-s",
        "--skip-hists",
        action="store_true",
        help="Skip saving histograms",
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
    hists: dict | None,
    args: argparse.Namespace,
) -> None:
    is_mc = "data" not in sample_name
    trig_set = None
    if "trigs" in selections:
        trig_set = selections["trigs"].get("value")
    branch_aliases = get_branch_aliases(is_mc, trig_set, sample_metadata)
    if args.sample_weight is None and is_mc:
        cbk = concatenate_cutbookkeepers(sample_path)
        sample_weight = get_sample_weight(sample_metadata, cbk)
    else:
        sample_weight = 1.0 if args.sample_weight is None else args.sample_weight
    batches = []
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
        processed_batch = process_batch(
            batch_events,
            selections,
            is_mc,
        )
        if len(processed_batch) == 0:
            continue
        processed_batch["event_weight"] = processed_batch.event_weight * sample_weight
        valid_events = np.zeros(len(processed_batch), dtype=bool)
        if "jets" in selections and "btagging" in selections["jets"]:
            bjets_sel = selections["jets"]["btagging"]
            if isinstance(bjets_sel, dict):
                bjets_sel = [bjets_sel]
            for i_bjets_sel in bjets_sel:
                btag_model = i_bjets_sel["model"]
                btag_eff = i_bjets_sel["efficiency"]
                n_btags = i_bjets_sel["count"]["value"]
                btagger = format_btagger_model_name(
                    btag_model,
                    btag_eff,
                )
                valid_events = (
                    valid_events
                    | processed_batch[f"valid_{n_btags}_btag_{btagger}_events"]
                )
        processed_batch = processed_batch[valid_events]
        batches.append(processed_batch)
        if not args.skip_hists:
            fill_hists(processed_batch, hists[sample_name], selections, is_mc)
    if not args.skip_hists:
        hists_output_name = args.output.with_name(
            f"hists_{args.output.stem}_{sample_name}_{os.getpgid(os.getpid())}.h5"
        )
        with h5py.File(hists_output_name, "w") as output_file:
            logger.info(f"Saving histograms to file: {hists_output_name}")
            write_hists(hists[sample_name], sample_name, output_file)
    # concatenate all batches
    batches = ak.concatenate(batches)
    # bad_branches = [
    #     "hh_4b_GN2v01_77_truth_matched_jet_idx",
    #     "non_hh_4b_GN2v01_77_truth_matched_jet_idx",
    #     "H1_4b_GN2v01_77_min_deltar_pairing_truth_matched_jet_idx",
    #     "H2_4b_GN2v01_77_min_deltar_pairing_truth_matched_jet_idx",
    #     "H1_4b_GN2v01_77_min_mass_optimized_1D_medium_pairing_truth_matched_jet_idx",
    #     "H2_4b_GN2v01_77_min_mass_optimized_1D_medium_pairing_truth_matched_jet_idx",
    # ]
    # for bad_branch in bad_branches:
    #     del batches[bad_branch]
    output_name = args.output.with_name(
        f"{args.output.stem}_{sample_name}_{os.getpgid(os.getpid())}{args.output.suffix}"
    )
    logger.info(f"Saving processed events to {output_name}")
    ak.to_parquet(batches, output_name)


def main():
    starttime = time.time()

    args = get_args()

    if args.loglevel:
        setup_logger(args.loglevel)

    # check that output file extension is .parquet
    if args.output.suffix != ".parquet":
        logger.error("Only parquet file output supported!")
        sys.exit(1)

    with open(args.config) as cf:
        config = resolve_project_paths(config=json.load(cf))

    samples, event_selection = config["samples"], config["event_selection"]
    hists = None if args.skip_hists else init_hists(samples, event_selection, args)

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
