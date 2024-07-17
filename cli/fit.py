#!/usr/bin/env python3

import re
import time
import shutil
import logging
import argparse
import cabinetry
import numpy as np
from pathlib import Path
from hh.shared.utils import logger, setup_logger, merge_sample_files


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "histograms", type=Path, nargs="+", help="Path to directory with histograms"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Cabinetry fitting config file for existing template histograms",
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
        default=13.0,
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

    # # # Merge h5 histograms from multiple files
    # hists = merge_sample_files(
    #     args.histograms,
    #     merge_jz_regex=re.compile(r"JZ[0-9]"),
    # )
    # hists = {
    #     "mc20_ggF_k01": {
    #         "hh_mass_reco_signal_4b_GN2v01_77_min_deltar_pairing": {
    #             "values": np.array(
    #                 [
    #                     0,
    #                     0.013758369810071391,
    #                     0.8187499569821232,
    #                     0.6288284511376767,
    #                     0.21472926142753618,
    #                     0,
    #                 ]
    #             )
    #         }
    #     },
    #     "mc20_multijet": {
    #         "hh_mass_reco_signal_4b_GN2v01_77_min_deltar_pairing": {
    #             "values": np.array(
    #                 [
    #                     0,
    #                     70.90728876386044,
    #                     342.2130848865363,
    #                     57.76969233717141,
    #                     12.232545468438332,
    #                     0,
    #                 ]
    #             )
    #         }
    #     },
    #     "mc20_ttbar": {
    #         "hh_mass_reco_signal_4b_GN2v01_77_min_deltar_pairing": {
    #             "values": np.array(
    #                 [
    #                     0,
    #                     0.21922563621774316,
    #                     2.2289197265927214,
    #                     0.9012602276197867,
    #                     0.18140052212402225,
    #                     0,
    #                 ]
    #             )
    #         }
    #     },
    # }

    # Load cabinetry config for already made histograms
    cabinetry_config_histograms = cabinetry.configuration.load(args.config)
    cabinetry.configuration.print_overview(cabinetry_config_histograms)
    # Convert h5 histograms to ROOT format
    # fitting.utils.convert_hists_2_root(hists, cabinetry_config_histograms)
    cabinetry.templates.collect(cabinetry_config_histograms, method="uproot")
    # Postprocess template histograms
    cabinetry.templates.postprocess(cabinetry_config_histograms)
    # Load workspace
    workspace_path = args.output_dir / "cabinetry_workspace.json"
    ws = cabinetry.workspace.build(cabinetry_config_histograms)
    cabinetry.workspace.save(ws, workspace_path)
    # Load data and model
    ws = cabinetry.workspace.load(workspace_path)
    model, data = cabinetry.model_utils.model_and_data(ws, asimov=True)
    cabinetry.visualize.modifier_grid(model)

    # visualize templates
    model_pred = cabinetry.model_utils.prediction(model)
    cabinetry.visualize.data_mc(model_pred, data, config=cabinetry_config_histograms)

    # fit!
    tolerance = 100
    fit_results = cabinetry.fit.fit(
        model, data, goodness_of_fit=True, tolerance=tolerance
    )
    data = cabinetry.model_utils.asimov_data(model=model, fit_results=fit_results)
    # pull plot
    cabinetry.visualize.pulls(fit_results, exclude=["Signal_norm"])
    limit_results = cabinetry.fit.limit(model, data, tolerance=tolerance)
    cabinetry.visualize.limit(limit_results)

    if logger.level == logging.DEBUG:
        logger.debug(
            f"Loading data & processing events execution time: {time.time() - starttime} seconds"
        )


if __name__ == "__main__":
    main()
