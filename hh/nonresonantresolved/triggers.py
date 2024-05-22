trigs_long = [
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bdl1d77_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25",  # run 3 2022 asymmetric
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_3j20c_020jvt_bdl1d82_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25",  # run 3 2022 asymmetric
    "HLT_2j45_0eta290_020jvt_bdl1d60_2j45_pf_ftf_presel2j25XX2j25b85_L14J15p0ETA25",  # run 3 2022 symmetric
    "HLT_j150_2j55_0eta290_020jvt_bdl1d70_pf_ftf_preselj80XX2j45b90_L1J85_3J30",  # run 3 2022
    "HLT_2j35c_020jvt_bdl1d60_2j35c_020jvt_pf_ftf_presel2j25XX2j25b85_L14J15p0ETA25",  # run 3 2022 symmetric
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bdl1d77_pf_ftf_presel2c20XX2c20b85_L1MU8F_2J15_J20",  # run 3 2022 asymmetric
    "HLT_j75c_020jvt_j50c_020jvt_j25c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bgn177_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25",  # run 3 2023 asymmetric
    "HLT_2j35c_020jvt_bgn160_2j35c_020jvt_pf_ftf_presel2j25XX2j25b85_L14J15p0ETA25",  # run 3 2023 symmetric
    "HLT_j140_2j50_0eta290_020jvt_bgn170_pf_ftf_preselj80XX2j45b90_L1J85_3J30",  # run 3 2023
    "HLT_j225_bmv2c2060_split",  # run2 2016 1b
    "HLT_j100_2j55_bmv2c2060_split",  # run2 2016 2b1j
    "HLT_2j35_bmv2c2060_split_2j35_L14J15p0ETA25",  # run2 2016 2bHT
    "HLT_2j35_bmv2c2060_split_2j35_L14J15p0ETA25",  # run2 2016 2b2j
    "HLT_j225_gsc300_bmv2c1070_split",  # run2 2017 1b
    "HLT_j110_gsc150_boffperf_split_2j35_gsc55_bmv2c1070_split_L1J85_3J30",  # run2 2017 2b1j
    "HLT_2j35_gsc55_bmv2c1050_split_ht300_L1HT190_J15s5pETA21",  # run2 2017 2bHT
    "HLT_2j15_gsc35_bmv2c1040_split_2j15_gsc35_boffperf_split_L14J15p0ETA25",  # run2 2017 2b2j
    "HLT_j225_gsc300_bmv2c1070_split",  # run2 2018 1b
    "HLT_j110_gsc150_boffperf_split_2j45_gsc55_bmv2c1070_split_L1J85_3J30",  # run2 2018 2b1j
    "HLT_2j45_gsc55_bmv2c1050_split_ht300_L1HT190_J15s5pETA21",  # run2 2018 2bHT
    "HLT_2j35_bmv2c1060_split_2j35_L14J15p0ETA25",  # run2 2018 2b2j
]

trigs_short = [
    "assym_2b2j_delayed",
    "assym_3b1j",
    "symm_2b2j_central",
    "2b1j",
    "symm_2b2j",
    "asymm_2b2j_L1mu",
    "assym_2b2j_delayed",
    "symm_2b2j",
    "2b1j",
    "1b",
    "2b1j",
    "2bHT",
    "2b2j",
    "1b",
    "2b1j",
    "2bHT",
    "2b2j",
    "1b",
    "2b1j",
    "2bHT",
    "2b2j",
]

trigs_labels = [
    "Asymm 2b2j DL1d@77% (Delayed)",  # run3 2022, only used when running on delayed stream files
    "Asymm 3b1j DL1d@82% (Main)",  # run3 2022
    "Symm 2b2j DL1d@60% Central (Main)",  # run2 reoptimized
    "2b1j DL1d@70% (Main)",  # run2 2022 reoptimized
    "Symm 2b2j DL1d@60% (Main)",  # run2 reoptimized
    "Asymm 2b2j+L1mu DL1d@77%",  # run3, not studied yet (not used in analysis)
    "Asymm 2b2j GN1d@77% (Delayed)",  # run3 2023, only used when running on delayed stream files
    "Symm 2b2j GN1d@60% (Main)",  # run3 2023
    "2b1j GN1@70% (Main)",  # run3 2023
    "1b",  # run2 2016
    "2b1j",  # run2 2016
    "2bHT",  # run2 2016
    "2b2j",  # run2 2016
    "1b",  # run2 2017
    "2b1j",  # run2 2017
    "2bHT",  # run2 2017
    "2b2j",  # run2 2017
    "1b",  # run2 2018
    "2b1j",  # run2 2018
    "2bHT",  # run2 2018
    "2b2j",  # run2 2018
]

run3_2022_main_stream_idx = [1, 2, 3, 4]
run3_2022_delayed_stream_idx = [0, 5]
run3_2022_asymm_L1_jet_idx = [0, 1]
run3_2022_asymm_L1_all_idx = [0, 1, 5]
run3_2023_main_stream_idx = [7, 8]
run3_2023_delayed_stream_idx = [6]
run2_reoptimized_idx = [3, 4, 5]
run2_2016_idx = [9, 10, 11, 12]
run2_2017_idx = [13, 14, 15, 16]
run2_2018_idx = [17, 18, 19, 20]


def _get_triggers(idx):
    return [(trigs_long[i], trigs_short[i], trigs_labels[i]) for i in idx]


run3_2022_main_stream = _get_triggers(run3_2022_main_stream_idx)
run3_2022_delayed_stream = _get_triggers(run3_2022_delayed_stream_idx)
run3_2022_asymm_L1_jet = _get_triggers(run3_2022_asymm_L1_jet_idx)
run3_2022_asymm_L1_all = _get_triggers(run3_2022_asymm_L1_all_idx)
run3_2023_main_stream = _get_triggers(run3_2023_main_stream_idx)
run3_2023_delayed_stream = _get_triggers(run3_2023_delayed_stream_idx)
run2_reoptimized = _get_triggers(run2_reoptimized_idx)
run2_2016 = _get_triggers(run2_2016_idx)
run2_2017 = _get_triggers(run2_2017_idx)
run2_2018 = _get_triggers(run2_2018_idx)
all = dict(zip(trigs_labels, trigs_long))

trig_sets = {
    "Run 2 2016": run2_2016,
    "Run 2 2017": run2_2017,
    "Run 2 2018": run2_2018,
    "Run 2 reoptimized": run2_reoptimized,
    "Run 3 Main physics stream": run3_2022_main_stream,
    "Run 3 Main + delayed streams": run3_2022_delayed_stream,
    "Run 3 Asymm L1 jet": run3_2022_asymm_L1_jet,
    "Run 3 Asymm L1 all": run3_2022_asymm_L1_all,
    "Run 3 Experimental 2022": run3_2022_main_stream + run3_2022_delayed_stream,
    "Run 3 Experimental 2023": run3_2023_main_stream + run3_2023_delayed_stream,
}
