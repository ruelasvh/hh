#!/usr/bin/env python3

import uproot
import numpy as np
import awkward as ak
import vector as p4
import hep2plts as plts

SAMPLES_PATH = "/lustre/fs22/group/atlas/ruelasv/samples/mc20_13TeV.gHH4b/output"

mG3000_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514741.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M3000.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352670._000001.output-hh4b.root"
mG2500_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514739.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M2500.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352658._000001.output-hh4b.root"
mG2000_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514737.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M2000.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352641._000001.output-hh4b.root"
mG1500_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514734.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M1500.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352621._000001.output-hh4b.root"
mG1000_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514729.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M1000.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352577._000001.output-hh4b.root"
mG900_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514728.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M900.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352567._000001.output-hh4b.root"
mG800_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514727.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M800.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352557._000001.output-hh4b.root"
mG700_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514726.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M700.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352546._000001.output-hh4b.root"
mG600_mc20e = f"{SAMPLES_PATH}/user.viruelas.HH4b.2022_11_23.514725.MGPy8EG_A14N23LO_RS_G_hh_bbbb_c10_M600.e8470_s3681_r13145_p5440_TREE/user.viruelas.31352535._000001.output-hh4b.root"

invGeV = 1 / 1000


def get_trig_passed(sample):
    trig_passed = sample["trigPassed_HLT_j420_a10t_lcw_jes_35smcINF_L1J100"]
    return trig_passed


def get_total_events(sample):
    total_events = sample["cutbookkeepers"][0].values()[0]
    return total_events


def get_mass_point(sample_name):
    mass_point = int(sample_name.split("_")[0].replace("mG", "")) * invGeV
    return mass_point


def compute_eff(trig_passed, total):
    tot_trig_passed = ak.sum(trig_passed)
    eff = tot_trig_passed / total
    return eff


def draw_trig_eff(largeR_dict):
    fig_trigEff, ax_trigEff = plts.subplots()
    total_events = []
    trig_passed = []
    mass_points = []
    for sample_name, sample in largeR_dict.items():
        total_events += [get_total_events(sample)]
        trig_passed += [get_trig_passed(sample)]
        mass_points += [get_mass_point(sample_name)]

    effs = [compute_eff(trig, tot) for trig, tot in zip(trig_passed, total_events)]
    ax_trigEff.pplot(
        mass_points,
        effs,
        label="Trigger",
        x_label=r"m$\left( \mathit{G*_{KK}} \right)$ [TeV]",
        y_label=r"Acceptance $\times$ Efficiency",
        atlas_sec_tag="Boosted channel, spin-2 signal",
        marker="o",
        enlarge=2,
    )
    ax_trigEff.axhline(y=1.0, color="k", linestyle="--")
    fig_trigEff.savefig("trig_eff.png", bbox_inches="tight")


def run():
    largeR_dict = {}
    samples = {
        "mG600_mc20e": mG600_mc20e,
        "mG700_mc20e": mG700_mc20e,
        "mG800_mc20e": mG800_mc20e,
        "mG900_mc20e": mG900_mc20e,
        "mG1000_mc20e": mG1000_mc20e,
        "mG1500_mc20e": mG1500_mc20e,
        "mG2000_mc20e": mG2000_mc20e,
        "mG2500_mc20e": mG2500_mc20e,
        "mG3000_mc20e": mG3000_mc20e,
    }
    for sname, fname in samples.items():
        with uproot.open(f"{fname}") as f:
            tree = f["AnalysisMiniTree"]
            cbk_key = [key for key in f.keys() if key.startswith("CutBookkeeper")][0]
            branches = tree.arrays(filter_name="/trigPassed_HLT/i")
            branches["cutbookkeepers"] = f[cbk_key]
            largeR_dict[sname] = branches
    draw_trig_eff(largeR_dict)


if __name__ == "__main__":
    run()
