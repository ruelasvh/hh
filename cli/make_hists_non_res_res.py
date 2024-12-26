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
import onnxruntime as ort
from hh.nonresonantresolved.inithists import init_hists
from hh.nonresonantresolved.branches import (
    get_branch_aliases,
)
from hh.nonresonantresolved.processbatches import (
    process_batch,
)
from hh.nonresonantresolved.fillhists import (
    fill_analysis_hists,
    fill_leading_jets_histograms,
    fill_n_true_bjet_composition_histograms,
)
from hh.nonresonantresolved.pairing import pairing_methods as all_pairing_methods
from hh.shared.utils import (
    logger,
    setup_logger,
    concatenate_cutbookkeepers,
    get_sample_weight,
    write_hists,
    resolve_project_paths,
    format_btagger_model_name,
    write_batches,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(help="default: %(default)s")

    def check_file_extension(value, extension):
        if not value.endswith(extension):
            raise argparse.ArgumentTypeError(f"File must have a {extension} extension")
        return Path(value)

    parser.add_argument("config", type=lambda x: check_file_extension(x, ".json"))
    # Make arguments -p and -o mutually exclusive. If postprocessed, shouldn't save again
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-o",
        "--output",
        type=lambda x: check_file_extension(x, ".parquet"),
        **defaults,
    )
    group.add_argument(
        "-p",
        "--postprocessed-path",
        type=Path,
        help="Path to the postprocessed parquet files",
    )
    parser.add_argument(
        "-s",
        "--skip-hists",
        action="store_true",
        help="Skip saving histograms",
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
    sample_hists: dict | None,
    args: argparse.Namespace,
) -> None:
    """
    Apply the event selection, fill histograms, and save the output
    """

    is_mc = sample_metadata["isMC"]

    # get the sample weight
    sample_weight = 1.0
    if is_mc:
        initial_sum_of_weights = sample_metadata.get("initialSumWeights", None)
        if initial_sum_of_weights is None:
            cbk = concatenate_cutbookkeepers(sample_path)
            initial_sum_of_weights = cbk["initial_sum_of_weights"]
        sample_weight = get_sample_weight(sample_metadata, initial_sum_of_weights)

    # get the branch aliases
    trig_set = set()
    if "trigs" in selections:
        trig_set.add(selections["trigs"]["value"])
    branch_aliases = get_branch_aliases(is_mc, list(trig_set), sample_metadata)

    # iterate over the files in the sample
    batches = []
    ort_session = None
    if "CLAHH_discriminant" in selections:
        ort_session = ort.InferenceSession(selections["CLAHH_discriminant"]["model"])
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

        batch_events = ak.with_name(batch_events, sample_name)

        # set the total event weight
        if is_mc:
            batch_events["event_weight"] = (
                np.prod(
                    [batch_events.mc_event_weights[:, 0], batch_events.pileup_weight],
                    axis=0,
                )
                * sample_weight
            )
            del batch_events["mc_event_weights"]
        else:
            batch_events["event_weight"] = np.ones(len(batch_events), dtype=float)

        # fill the leading jets and bjet composition histograms before any selections
        if not args.skip_hists:
            fill_leading_jets_histograms(
                batch_events,
                sample_hists,
                jet_type="truth_jet",
                hist_prefix="leading_truth_jet",
            )
            fill_n_true_bjet_composition_histograms(batch_events, sample_hists)

        processed_batch = process_batch(
            batch_events,
            selections,
            is_mc,
            year=sample_metadata["dataTakingYear"],
            clahh_model_ort=ort_session,
        )
        if processed_batch is None:
            continue

        if not args.skip_hists and sample_hists is not None:
            fill_analysis_hists(processed_batch, sample_hists, selections, is_mc)

        # only keep events that pass the btagging selection to keep the file size down
        valid_events = np.zeros(len(processed_batch), dtype=bool)
        pairing_methods = {}
        if "pairing" in selections:
            pairing_methods = {
                k: v
                for k, v in all_pairing_methods.items()
                if k in selections["pairing"]
            }
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
                    # | processed_batch[f"valid_2btags_{btagger}_events"]
                    | processed_batch[f"valid_{n_btags}btags_{btagger}_events"]
                )
                # These branches create issues for multijet samples
                bad_branches = [
                    f"hh_{n_btags}btags_{btagger}_truth_matched_jet_idx",
                    f"non_hh_{n_btags}btags_{btagger}_truth_matched_jet_idx",
                ]
                bad_branches += [
                    f"H{i}_{n_btags}btags_{btagger}_{pairing}_truth_matched_jet_idx"
                    for pairing in pairing_methods
                    for i in [1, 2]
                ]
                for bad_branch in bad_branches:
                    processed_batch[bad_branch] = ak.fill_none(
                        processed_batch[bad_branch], [], axis=0
                    )
        processed_batch = processed_batch[valid_events]
        batches.append(processed_batch)

    output_stemname = f"{sample_name}_{os.getpgid(os.getpid())}"
    if args.output is not None:
        output_stemname = args.output.with_name(f"{args.output.stem}_{output_stemname}")

    # save the histograms
    if not args.skip_hists:
        hists_output_name = f"{output_stemname}_hists.h5"
        with h5py.File(hists_output_name, "w") as output_file:
            logger.info(f"Saving histograms to file: {hists_output_name}")
            write_hists(sample_hists, sample_name, output_file)

    # return if no valid events found
    if len(batches) == 0:
        logger.info("No valid events found!")
        return

    # save the processed batches if an output file is specified
    if args.output is not None:
        # concatenate all processed batches
        batches = ak.concatenate(batches)
        output_name = f"{output_stemname}{args.output.suffix}"
        logger.info(f"Saving processed events to {output_name}")
        write_batches(batches, output_name)


def fill_output_histograms(postprocessed_path, selections, hists, args):
    """
    Process output samples by reading Parquet files and filling histograms.

    Parameters:
    - args: Namespace
        The arguments containing the postprocessed_path.
    - fill_analysis_hists: function
        The function to fill histograms with the data from each Parquet file.
    """

    # Ensure the path exists
    if not postprocessed_path.exists():
        raise FileNotFoundError(
            f"Postprocessed path {postprocessed_path} does not exist."
        )

    # Iterate through Parquet files in the specified path
    files = list(postprocessed_path.glob("*.parquet"))
    for i, file_path in enumerate(files):
        # Extract sample name from the file name
        sample_filename = file_path.stem
        # Read the Parquet file
        dataset = ak.from_parquet(file_path)
        metadata = ak.parameters(dataset)
        if metadata:
            sample_name = metadata["__record__"]
        else:
            sample_name = next(key for key in hists.keys() if key in sample_filename)
        is_mc = "data" not in sample_name.lower()
        # create variable called sample_name which is the match of the hists.keys() to the sample_filename
        logger.info(
            f"(Progress: {i + 1} out of {len(files)}) Filling histograms for sample {sample_name} from {file_path}"
        )
        fill_analysis_hists(dataset, hists[sample_name], selections, is_mc)

    # save the histograms
    if not args.skip_hists:
        for sample_name in hists.keys():
            hists_output_name = f"{sample_name}_hists.h5"
            with h5py.File(hists_output_name, "w") as output_file:
                logger.info(f"Saving histograms to file: {hists_output_name}")
                write_hists(hists[sample_name], sample_name, output_file)


def main():
    starttime = time.time()

    args = get_args()

    if args.loglevel:
        setup_logger(args.loglevel)

    with open(args.config) as cf:
        config = resolve_project_paths(config=json.load(cf))

    samples, event_selection = config["samples"], config["event_selection"]
    hists = None if args.skip_hists else init_hists(samples, event_selection, args)

    if args.postprocessed_path is not None:
        fill_output_histograms(args.postprocessed_path, event_selection, hists, args)
    else:
        worker_items = [
            (
                sample["label"],
                sample_path,
                sample["metadata"][idx] if "metadata" in sample else None,
                event_selection,
                hists[sample["label"]] if hists is not None else None,
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
