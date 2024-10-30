sample_types = {
    "ggF_k01": r"$\kappa_{\lambda}=1$ ggF",
    "ggF_k05": r"$\kappa_{\lambda}=5$ ggF",
    "ggF_k10": r"$\kappa_{\lambda}=10$ ggF",
    "ttbar": r"$t \bar t$",
    "multijet_b_filtered": "QCD b-filtered",
    "multijet": "QCD multijet",
}

sample_labels = {
    # mc20 ggF kl=1 samples
    "mc20_ggF_k01": f"{sample_types['ggF_k01']} MC 2016-2018",
    "mc20a_ggF_k01": f"{sample_types['ggF_k01']} MC 2016",
    "mc20d_ggF_k01": f"{sample_types['ggF_k01']} MC 2017",
    "mc20e_ggF_k01": f"{sample_types['ggF_k01']} MC 2018",
    # mc20 ggF kl=10 samples
    "mc20_ggF_k10": f"{sample_types['ggF_k10']} MC 2016-2018",
    "mc20a_ggF_k10": f"{sample_types['ggF_k10']} MC 2016",
    "mc20d_ggF_k10": f"{sample_types['ggF_k10']} MC 2017",
    "mc20e_ggF_k10": f"{sample_types['ggF_k10']} MC 2018",
    # mc20 ttbar samples
    "mc20_ttbar": f"{sample_types['ttbar']} MC 2016-2018",
    "mc20a_ttbar": f"{sample_types['ttbar']} MC 2016",
    "mc20d_ttbar": f"{sample_types['ttbar']} MC 2017",
    "mc20e_ttbar": f"{sample_types['ttbar']} MC 2018",
    # mc20 QCD samples
    "mc20_multijet": f"{sample_types['multijet_b_filtered']} MC 2016-2018",
    "mc20a_multijet": f"{sample_types['multijet_b_filtered']} MC 2016",
    "mc20d_multijet": f"{sample_types['multijet_b_filtered']} MC 2017",
    "mc20e_multijet": f"{sample_types['multijet_b_filtered']} MC 2018",
    # mc23 ggF kl=1 samples
    "mc23_ggF_k01": f"{sample_types['ggF_k01']} MC23",
    "mc23a_ggF_k01": f"{sample_types['ggF_k01']} MC23a",
    "mc23d_ggF_k01": f"{sample_types['ggF_k01']} MC23d",
    # mc23 ggF kl=5 samples
    "mc23_ggF_k05": f"{sample_types['ggF_k05']} MC23",
    "mc23a_ggF_k05": f"{sample_types['ggF_k05']} MC23a",
    "mc23d_ggF_k05": f"{sample_types['ggF_k05']} MC23d",
    # mc23 QCD samples
    "mc23_multijet": f"{sample_types['multijet']} MC23",
    "mc23a_multijet": f"{sample_types['multijet']} MC23a",
    "mc23d_multijet": f"{sample_types['multijet']} MC23d",
    # mc23 ttbar samples
    "mc23_ttbar": f"{sample_types['ttbar']} MC23",
    "mc23a_ttbar": f"{sample_types['ttbar']} MC23a",
    "mc23d_ttbar": f"{sample_types['ttbar']} MC23d",
}

hh_var_labels = {
    "hh_mass": r"$m_{\mathrm{HH}}$ [GeV]",
    "hh_pt": r"$p_{\mathrm{T}}$ (HH) [GeV]",
    "hh_sum_jet_pt": r"$H_{\mathrm{T}}$ $(\Sigma^{\mathrm{jets}} p_{\mathrm{T}})$ [GeV]",
    "hh_delta_eta": r"$\Delta\eta_{\mathrm{HH}}$",
}

selections_labels = {
    "truth_matching": r"$\Delta R < 0.3$ truth-matched jets",
    "central_jets": r"$\geq$ 4 jets with $p_{\mathrm{T}} > 25$ GeV, $|\eta| < 2.5$",
    "btagged_GN277_4_jets": r"$\geq$ 4 b-tags with GN2v01@77%",
    # "truth_matched_4_plus_jets": r"${n_b}_\mathrm{match} \geq$ 4",
    "truth_matched_4_plus_jets": r"$\geq$ 4 truth-matched jets",
}

kin_labels = {"pt": r"$p_T$", "eta": r"$\eta$", "phi": r"$\phi$", "mass": r"$m$"}
