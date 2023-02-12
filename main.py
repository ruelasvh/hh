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

mc21_ggF_k01_small = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/run/analysis-variables-run3-k01"
mc21_ggF_k10_small = "/lustre/fs22/group/atlas/ruelasv/hh4b-analysis-r22-build/run/analysis-variables-run3-k10"
mc21_ggF_k01 = "/lustre/fs22/group/atlas/ruelasv/samples/mc21_13p6TeV.hh4b.ggF/output/user.viruelas.HH4b.ggF.2022_12_15.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5440_TREE/"
mc21_ggF_k10 = "/lustre/fs22/group/atlas/ruelasv/samples/mc21_13p6TeV.hh4b.ggF/output/user.viruelas.HH4b.ggF.2022_12_15.601480.PhPy8EG_HH4b_cHHH10d0.e8472_s3873_r13829_p5440_TREE/"


def main():
    inputs = {
        "k01": mc21_ggF_k01_small,
        "k10": mc21_ggF_k10_small,
        # "k01": mc21_ggF_k01,
        # "k10": mc21_ggF_k10,
    }
    outputs = {}
    st = time.time()
    for sample_name, sample_path in inputs.items():
        outputs[sample_name] = {}
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
