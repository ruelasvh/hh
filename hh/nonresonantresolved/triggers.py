trigs_long = [
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bdl1d77_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25",  # asymmetric
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_3j20c_020jvt_bdl1d82_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25",  # asymmetric
    "HLT_j150_2j55_0eta290_020jvt_bdl1d70_pf_ftf_preselj80XX2j45b90_L1J85_3J30",  # symmetric
    "HLT_2j35c_020jvt_bdl1d60_2j35c_020jvt_pf_ftf_presel2j25XX2j25b85_L14J15p0ETA25",  # symmetric
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bdl1d77_pf_ftf_presel2c20XX2c20b85_L1MU8F_2J15_J20",  # asymmetric
]

trigs_short = [
    "assym_2b2j_delayed",
    "assym_3b1j",
    "2b1j",
    "symm_2b2j",
    "asymm_2b2j_L1mu",
]

trigs_labels = [
    "Asymm 2b2j DL1d@77% (Delayed)",  # run3, only used when running on delayed stream files
    "Asymm 3b1j DL1d@82% (Main)",  # run3
    "2b1j DL1d@70% (Main)",  # run2 reoptimized
    "Symm 2b2j DL1d@60% (Main)",  # run2 reoptimized
    "Asymm 2b2j+L1mu DL1d@77%",  # run3, not studied yet (not used in analysis)
]

run3_main_stream_idx = [1, 2, 3]
run3_delayed_stream_idx = [0, 1, 2, 3]
run3_asymm_L1_jet_idx = [0, 1]
run3_asymm_L1_all_idx = [0, 1, 4]
run2_reoptimized_idx = [2, 3]


def _get_triggers(idx):
    return [(trigs_long[i], trigs_short[i], trigs_labels[i]) for i in idx]


run3_main_stream = _get_triggers(run3_main_stream_idx)
run3_delayed_stream = _get_triggers(run3_delayed_stream_idx)
run3_asymm_L1_jet = _get_triggers(run3_asymm_L1_jet_idx)
run3_asymm_L1_all = _get_triggers(run3_asymm_L1_all_idx)
run2_reoptimized = _get_triggers(run2_reoptimized_idx)
all = dict(zip(trigs_labels, trigs_long))

trig_sets = {
    "Run 2": run2_reoptimized,
    "Main physics stream": run3_main_stream,
    "Main + delayed streams": run3_delayed_stream,
    "Asymm L1 jet": run3_asymm_L1_jet,
    "Asymm L1 all": run3_asymm_L1_all,
}
