run3_all_short = [
    "Asymm 2b2j DL1d@77% (Delayed)",  # run3
    "Asymm 3b1j DL1d@82% (Main)",  # run3
    "2b1j DL1d@70% (Main)",  # run2 reoptimized
    "Symm 2b2j DL1d@60% (Main)",  # run2 reoptimized
    "Asymm 2b2j+L1mu DL1d@77%",  # run3
]
run3_all = [
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bdl1d77_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25",  # 1 asymm
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_3j20c_020jvt_bdl1d82_pf_ftf_presel2c20XX2c20b85_L1J45p0ETA21_3J15p0ETA25",  # 2 asymm
    "HLT_j150_2j55_0eta290_020jvt_bdl1d70_pf_ftf_preselj80XX2j45b90_L1J85_3J30",  # 3 symm
    "HLT_2j35c_020jvt_bdl1d60_2j35c_020jvt_pf_ftf_presel2j25XX2j25b85_L14J15p0ETA25",  # 4 symm,
    "HLT_j80c_020jvt_j55c_020jvt_j28c_020jvt_j20c_020jvt_SHARED_2j20c_020jvt_bdl1d77_pf_ftf_presel2c20XX2c20b85_L1MU8F_2J15_J20",  # 5 asymm
]
run3_main_stream = run3_all[1:4]
run3_delayed_stream = run3_all[0:4]
run3_asymm_L1_jet = run3_all[0:2]
run3_asymm_L1_all = run3_all[0:2] + run3_all[4:5]
run2 = run3_all[2:4]

sets = {
    "Run 2": run2,
    "Main physics stream": run3_main_stream,
    "Main + delayed streams": run3_delayed_stream,
    "Asymm L1 jet": run3_asymm_L1_jet,
    "Asymm L1 all": run3_asymm_L1_all,
    "OR of all": run3_all,
}
