#!/usr/bin/env python3

"""
build plots of everything
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import uproot
import numpy as np
import triggers
from cuts import (
    select_ge4_central_jets_events,
    select_jets_sorted_by_pt,
    select_trigger_decisions,
)
from fill_hists import (
    init_leading_jets_passed_trig_hists,
    init_mH_passed_trig_hists,
    init_mH_plane_passed_trig_hists,
    fill_leading_jet_pt_passed_trig_hists,
    fill_mH_passed_trig_hists,
    fill_mH_plane_passed_trig_hists,
)
from draw_hists import draw_hists
import time
import logging


np.seterr(divide="ignore", invalid="ignore")


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            if ".json" in value:
                with open(Path(value)) as ifile:
                    values = json.load(ifile)
                for key, value in values.items():
                    getattr(namespace, self.dest)[key] = value
            else:
                key, value = value.split("=")
                getattr(namespace, self.dest)[key] = value


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(help="default: %(default)s")
    parser.add_argument("inputs", nargs="*", action=ParseKwargs)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("hists.h5"),
        **defaults,
    )
    parser.add_argument(
        "-t", "--test", action="store_true", help="run on small test files"
    )
    return parser.parse_args()


def main():
    args = get_args()
    inputs = args.inputs
    if args.test and not bool(inputs):
        inputs = {
            "k01": Path("test-data/analysis-variables-mc21-ggF-k01"),
            "k10": Path("test-data/analysis-variables-mc21-ggF-k10"),
        }
    outputs = defaultdict(lambda: defaultdict(int))
    st = time.time()
    for sample_name, sample_path in inputs.items():
        outputs[sample_name][
            "leading_jets_passed_trig_hists"
        ] = init_leading_jets_passed_trig_hists()
        outputs[sample_name]["mH_passed_trig_hists"] = init_mH_passed_trig_hists()
        outputs[sample_name][
            "mH_plane_passed_trig_hists"
        ] = init_mH_plane_passed_trig_hists()
        for events, report in uproot.iterate(
            f"{sample_path}*.root:AnalysisMiniTree",
            filter_name=[
                "/recojet_antikt4_NOSYS_(pt|eta|phi|m)/",
                "/recojet_antikt4_NOSYS_(DL1dv01|GN120220509)_FixedCutBEff_77/",
                "/truth_H1_(pt|eta|phi|m)/",
                "/truth_H2_(pt|eta|phi|m)/",
                "/resolved_DL1dv01_FixedCutBEff_70_h(1|2)_m/",
                *[f"trigPassed_{trig}" for trig in triggers.run3_all],
            ],
            step_size="1 GB",
            report=True,
        ):
            print(report)
            # print(events.type.show())
            cut_events = select_ge4_central_jets_events(events, eta_cut=2.5)
            # print(cut_events.type.show())
            sorted_jets = select_jets_sorted_by_pt(cut_events)
            # print(sorted_jets.type.show())
            trig_decisioins = select_trigger_decisions(cut_events)
            # print(trigs.type.show())
            fill_leading_jet_pt_passed_trig_hists(
                sorted_jets["recojet_antikt4_NOSYS_pt"],
                trig_decisioins,
                outputs[sample_name]["leading_jets_passed_trig_hists"],
            )
            fill_mH_passed_trig_hists(
                events,
                select_trigger_decisions(events),
                outputs[sample_name]["mH_passed_trig_hists"],
            )
            fill_mH_plane_passed_trig_hists(
                events,
                select_trigger_decisions(events),
                outputs[sample_name]["mH_plane_passed_trig_hists"],
            )
    et = time.time()
    logging.info("Execution time:", et - st, "seconds")
    draw_hists(outputs)


if __name__ == "__main__":
    main()
