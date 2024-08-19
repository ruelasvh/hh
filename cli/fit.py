#!/usr/bin/env python3

import re
import copy
import time
import shutil
import logging
import argparse
import cabinetry
from pathlib import Path
import hh.fitting as fitting
from hh.shared.labels import sample_labels, sample_types
from hh.nonresonantresolved.pairing import pairing_methods
from hh.shared.utils import logger, setup_logger, merge_sample_files


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="Cabinetry fitting config file for existing template histograms",
    )
    parser.add_argument(
        "-i",
        "--histograms",
        type=Path,
        nargs="+",
        help="Path to directory with histograms to merge",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default="results",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "-l",
        "--luminosity",
        type=float,
        default=1.0,
        help="Luminosity in fb^-1 (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--energy",
        type=float,
        default=13,
        help="Center of mass energy in TeV (default: %(default)s)",
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
        help="Print info statements",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    return parser.parse_args()


def main():
    starttime = time.time()

    args = get_args()

    if args.loglevel:
        setup_logger(args.loglevel)
        cabinetry.set_logging()

    # Ask user if they want to overwrite the plots output directory, if not exit
    if args.output_dir.exists():
        overwrite = input(
            f"Output directory '{args.output_dir}' already exists, do you want to overwrite it? (Y/n) "
        )
        if overwrite == "" or overwrite.lower() == "y":
            shutil.rmtree(args.output_dir)
            args.output_dir.mkdir(parents=True)
        else:
            exit(0)

    else:
        args.output_dir.mkdir(parents=True)
        logger.info(f"Saving plots to '{args.output_dir}'")

    # Load cabinetry config for already made histograms
    cabinetry_config_base = cabinetry.configuration.load(args.config)
    # Get the path to the merged input histograms
    input_hist_path = Path(cabinetry_config_base["General"]["InputPath"].split(":")[0])
    if (not input_hist_path.exists()) and (not args.histograms):
        raise FileNotFoundError(
            f"No merged input histograms found at {input_hist_path}. "
            f"A path to the histograms to merge must be provided."
        )
    if not input_hist_path.exists() and args.histograms:
        # Merge h5 histograms from multiple files
        hists = merge_sample_files(
            args.histograms,
            merge_jz_regex=re.compile(r"JZ[0-9]"),
            save_to="merged_histograms.h5",
        )
        fitting.save_to_root(hists, cabinetry_config_base)
    for pairing in pairing_methods:
        # Create a new cabinetry config for each pairing method
        cabinetry_config_pairing = copy.deepcopy(cabinetry_config_base)
        regions = [
            {
                **region,
                "Name": f"{region['Name']}_{pairing}",
                "RegionPath": f"{region['RegionPath']}_{pairing}",
            }
            for region in cabinetry_config_pairing["Regions"]
        ]
        cabinetry_config_pairing["Regions"] = regions
        cabinetry_config_pairing["General"]["HistogramFolder"] = (
            args.output_dir / f"histograms_{pairing}"
        )
        signal_sample = [
            sample
            for sample in cabinetry_config_pairing["Samples"]
            if sample["SamplePath"].lower() == "signal"
        ][0]
        # Set bounds for max ΔR specifically
        if pairing == "max_deltar_pairing":
            cabinetry_config_pairing["NormFactors"][0]["Bounds"] = [
                v * 200 for v in cabinetry_config_pairing["NormFactors"][0]["Bounds"]
            ]

        # Print overview of the configuration if in debug mode
        cabinetry.configuration.print_overview(cabinetry_config_pairing)

        # Postprocess template histograms
        cabinetry.templates.collect(cabinetry_config_pairing, method="uproot")
        cabinetry.templates.postprocess(cabinetry_config_pairing)

        # Load workspace
        workspace_path = args.output_dir / f"cabinetry_workspace_{pairing}.json"
        ws = cabinetry.workspace.build(cabinetry_config_pairing)
        cabinetry.workspace.save(ws, workspace_path)

        # Load data and model
        ws = cabinetry.workspace.load(workspace_path)
        model, data = cabinetry.model_utils.model_and_data(ws)
        plots_path = args.output_dir / pairing
        cabinetry.visualize.modifier_grid(model, figure_folder=plots_path)

        # visualize templates
        model_pred = cabinetry.model_utils.prediction(model)
        cabinetry.visualize.data_mc(
            model_pred,
            data,
            config=cabinetry_config_pairing,
            figure_folder=plots_path,
        )

        limit_results = cabinetry.fit.limit(model, data)
        plot_details = (
            "\nMC20 2016-2018\n"
            + sample_types[signal_sample["Name"]]
            + r", $b$-filtered mulitjet and $t\bar{t}$"
        )
        fitting.plot_limits(
            limit_results,
            exp_label=plot_details,
            plot_label=pairing_methods[pairing]["label"],
            luminosity=args.luminosity,
            energy=args.energy,
            figure_folder=plots_path,
        )

    if logger.level == logging.DEBUG:
        logger.debug(
            f"Loading data & processing events execution time: {time.time() - starttime} seconds"
        )


if __name__ == "__main__":
    main()
