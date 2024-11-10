import numpy as np
from argparse import Namespace
from hh.nonresonantresolved.pairing import pairing_methods
from hh.shared.selection import X_HH
from hh.shared.labels import sample_labels, hh_var_labels, selections_labels, kin_labels
from hh.shared.drawhists import (
    draw_1d_hists,
    draw_kin_hists,
    draw_mHH_plane_2D_hists,
    draw_mHH_plane_3D_hists,
    draw_mHH_plane_projections_hists,
    draw_mH_1D_hists_v2,
    num_events_vs_sample,
    draw_1d_hists_v2,
    draw_efficiency,
    draw_signal_vs_background,
    draw_dijet_slices_hists,
)


def draw_hists(
    hists_group: dict,
    args: Namespace,
) -> None:
    """Draw all the histrograms"""

    lumi = args.luminosity
    energy = args.energy
    output_dir = args.output_dir
    btagging = {"GN2v01_77": 4}
    mc_campaigns = {
        "mc20a": 36.64674,
        "mc20d": 44.6306,
        "mc20e": 58.7916,
        "mc23a": 26.0714,
        "mc23d": 25.7675,
    }

    # ###############################################
    # # Leading jet pt plots for multijet samples
    # ###############################################
    draw_dijet_slices_hists(
        {k: v for k, v in hists_group.items() if "multijet" in k},
        hists_group,
        "leading_truth_jet_1_pt",
        energy,
        luminosity=lumi,
        xlabel=f"Leading truth jet {kin_labels['pt']} [GeV]",
        output_dir=output_dir,
    )
    for btagger, btag_count in btagging.items():
        draw_dijet_slices_hists(
            {k: v for k, v in hists_group.items() if "multijet" in k},
            hists_group,
            f"leading_resolved_{btag_count}btags_{btagger}_reco_jet_1_pt",
            energy,
            luminosity=lumi,
            xlabel=f"Leading resolved jet {kin_labels['pt']} [GeV]",
            third_exp_label="\n"
            + selections_labels["jets"](n_jets=4, pT_cut=20, eta_cut=2.5)
            + "\n"
            + selections_labels["b-tagging"](
                n_btags=btag_count,
                tagger=btagger.split("_")[0],
                eff=btagger.split("_")[1],
            ),
            output_dir=output_dir,
        )

    # ###############################################
    # # Truth H1 and H2 pT plots
    # ###############################################
    # for i in ["1", "2"]:
    #     draw_1d_hists(
    #         hists_group,
    #         f"h{i}_pt_truth",
    #         energy,
    #         xlabel=f"Truth H{i} {kin_labels['pt']} [GeV]",
    #         legend_labels={key: sample_labels[key] for key in hists_group.keys()},
    #         luminosity=lumi,
    #         output_dir=output_dir,
    #     )

    # ###############################################
    # # Truth HH mass plots
    # ###############################################
    # draw_1d_hists(
    #     hists_group,
    #     "hh_mass_truth",
    #     energy,
    #     xlabel=f"Truth {hh_var_labels['hh_mass']}",
    #     legend_labels={key: sample_labels[key] for key in hists_group.keys()},
    #     luminosity=lumi,
    #     xmin=100,
    #     ggFk01_factor=10,
    #     draw_errors=True,
    #     output_dir=output_dir,
    # )

    # ###############################################
    # # Reco truth-matched HH mass plots
    # ###############################################
    # draw_1d_hists(
    #     hists_group,
    #     f"hh_mass_reco_truth_matched",
    #     energy,
    #     xlabel=hh_var_labels["hh_mass"],
    #     ylabel="Events",
    #     legend_labels={key: sample_labels[key] for key in hists_group.keys()},
    #     third_exp_label=f"\n{selections_labels['truth_matching']}",
    #     luminosity=lumi,
    #     xmin=100,
    #     ggFk01_factor=10,
    #     draw_errors=True,
    #     output_dir=output_dir,
    # )

    # ################################################
    # # Plot backgrounds vs signal
    # ################################################
    # for mc, lumi in mc_campaigns.items():
    #     mc_samples = {
    #         sample: hists for sample, hists in hists_group.items() if mc in sample
    #     }
    #     mc_kl1_samples = {
    #         sample: hists for sample, hists in mc_samples.items() if "ggF_k01" in sample
    #     }
    #     mc_background_samples = {
    #         sample: hists
    #         for sample, hists in mc_samples.items()
    #         if any(bkg in sample for bkg in ["ttbar", "multijet"])
    #     }
    #     # for region in ["signal", "control"]:
    #     for region in ["signal"]:
    #         for pairing in pairing_methods:
    #             draw_signal_vs_background(
    #                 f"hh_mass_reco_{region}_4b_GN2v01_77_{pairing}_bins_logscale",
    #                 signal=mc_kl1_samples,
    #                 background=mc_background_samples,
    #                 energy=energy,
    #                 luminosity=lumi,
    #                 xlabel=hh_var_labels["hh_mass"],
    #                 legend_labels={
    #                     **{key: sample_labels[key] for key in mc_kl1_samples.keys()},
    #                     "Background MC": "Background MC",
    #                 },
    #                 third_exp_label="\n".join(
    #                     [
    #                         f"\n4b {region.capitalize()} Region",
    #                         f"{pairing_methods[pairing]['label']}",
    #                     ]
    #                 ),
    #                 plot_name=f"{mc}_sig_vs_bkg_{region}_region_4b_GN2v01_77_{pairing}_bins_logscale",
    #                 output_dir=output_dir,
    #                 show_counts=True,
    #             )

    for sample_type, sample_hists in hists_group.items():
        #### Draw jet flavor distribution histograms ####
        for pairing_id, pairing_info in pairing_methods.items():
            for btagger, btag_count in btagging.items():
                for pairing_id, pairing_info in pairing_methods.items():
                    draw_1d_hists_v2(
                        {sample_type: sample_hists},
                        [f"jet_flavor_signal_{btag_count}btags_{btagger}_{pairing_id}"],
                        energy,
                        luminosity=lumi,
                        xlabel="Jet flavor",
                        ylabel="Counts",
                        legend_labels=[f"{pairing_info['label']}"],
                        legend_options={"loc": "upper right", "fontsize": "small"},
                        third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
                        plot_name=f"jet_flavor_distribution_{pairing_id}",
                        output_dir=output_dir,
                    )
                    draw_1d_hists_v2(
                        {sample_type: sample_hists},
                        [
                            f"bjet_discrim_{flav}_signal_{btag_count}btags_{btagger}_{pairing_id}"
                            for flav in ["b", "c", "u"]
                        ],
                        energy,
                        luminosity=lumi,
                        xlabel="b-jet Discriminant",
                        ylabel="Counts",
                        legend_labels=[
                            f"{flav}-jets ({pairing_info['label']})"
                            for flav in ["b", "c", "u"]
                        ],
                        legend_options={"loc": "upper right", "fontsize": "small"},
                        third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
                        plot_name=f"jet_btag_discriminant_distribution_{pairing_id}",
                        output_dir=output_dir,
                    )

        # lumi = [mc_campaigns[mc] for mc in mc_campaigns if mc in sample_type][0]
        # ########################################################
        # # HH mass plots truth reco-matched vs reco truth-matched
        # ########################################################
        # draw_1d_hists_v2(
        #     {sample_type: sample_hists},
        #     [
        #         "hh_mass_truth_reco_matched",
        #         "hh_mass_reco_truth_matched",
        #     ],
        #     energy,
        #     luminosity=lumi,
        #     xlabel=hh_var_labels["hh_mass"],
        #     ylabel="Events",
        #     legend_labels=["Truth (reco-matched)", "Reco (truth-matched)"],
        #     legend_options={"loc": "center right", "fontsize": "small"},
        #     third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
        #     xmin=100,
        #     draw_errors=True,
        #     plot_name="hh_mass_truth_vs_reco_truth_matched",
        #     output_dir=output_dir,
        # )

        # ########################################################
        # # HH mass plots truth reco-matched vs reco truth-matched
        # # v2 uses HadronConeExclTruthLabelID = 5 to match jets
        # ########################################################
        # draw_1d_hists_v2(
        #     {sample_type: sample_hists},
        #     [
        #         "hh_mass_reco_truth_matched",
        #         "hh_mass_reco_truth_matched_v2",
        #     ],
        #     energy,
        #     luminosity=lumi,
        #     xlabel=hh_var_labels["hh_mass"],
        #     ylabel="Events",
        #     legend_labels=["Reco (truth-matched)", r"$\geq 4$ HadronConeExclTruthLabelID = 5"],
        #     legend_options={"loc": "center right", "fontsize": "small"},
        #     third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
        #     xmin=100,
        #     draw_ratio=True,
        #     ymin_ratio=0.5,
        #     ymax_ratio=2,
        #     plot_name="hh_mass_truth_matching_methods",
        #     output_dir=output_dir,
        # )

        # ###############################################
        # # HH mass response plots reco vs truth
        # ###############################################
        # draw_1d_hists(
        #     {sample_type: sample_hists},
        #     "hh_mass_reco_vs_truth_response",
        #     energy,
        #     luminosity=lumi,
        #     xlabel="HH mass response [%]",
        #     legend_labels={sample_type: sample_labels[sample_type]},
        #     third_exp_label=f"\n{selections_labels['truth_matching']}",
        #     output_dir=output_dir,
        # )

        #### HH mass plane plots for different pairing methods ####
        for pairing_id, pairing_info in pairing_methods.items():
            for btagger, btag_count in btagging.items():
                draw_mHH_plane_3D_hists(
                    sample_hists,
                    sample_type,
                    f"mHH_plane_reco_{btag_count}btags_{btagger}_{pairing_id}",
                    energy,
                    third_exp_label=pairing_info["label"].replace("pairing", ""),
                    output_dir=output_dir,
                )
                draw_mHH_plane_projections_hists(
                    sample_hists,
                    sample_type,
                    f"h1_mass_reco_{btag_count}btags_{btagger}_{pairing_id}",
                    f"h2_mass_reco_{btag_count}btags_{btagger}_{pairing_id}",
                    f"mHH_plane_reco_{btag_count}btags_{btagger}_{pairing_id}",
                    energy,
                    third_exp_label=pairing_info["label"].replace("pairing", ""),
                    output_dir=output_dir,
                )
                draw_mHH_plane_2D_hists(
                    sample_hists,
                    sample_type,
                    f"mHH_plane_reco_{btag_count}btags_{btagger}_{pairing_id}",
                    energy,
                    third_exp_label=pairing_info["label"].replace("pairing", ""),
                    log_z=True,
                    output_dir=output_dir,
                )
                draw_mHH_plane_2D_hists(
                    sample_hists,
                    sample_type,
                    f"mHH_plane_reco_{btag_count}btags_{btagger}_{pairing_id}_wrong_pairs",
                    energy,
                    third_exp_label=pairing_info["label"].replace("pairing", ""),
                    log_z=True,
                    output_dir=output_dir,
                )
                draw_mHH_plane_2D_hists(
                    sample_hists,
                    sample_type,
                    f"mHH_plane_reco_{btag_count}btags_{btagger}_{pairing_id}_lt_370_GeV",
                    energy,
                    third_exp_label=pairing_info["label"].replace("pairing", ""),
                    log_z=True,
                    output_dir=output_dir,
                )
                draw_mHH_plane_projections_hists(
                    sample_hists,
                    sample_type,
                    f"h1_mass_reco_{btag_count}btags_{btagger}_{pairing_id}_lt_370_GeV",
                    f"h2_mass_reco_{btag_count}btags_{btagger}_{pairing_id}_lt_370_GeV",
                    f"mHH_plane_reco_{btag_count}btags_{btagger}_{pairing_id}_lt_370_GeV",
                    energy,
                    third_exp_label=pairing_info["label"].replace("pairing", ""),
                    output_dir=output_dir,
                )
                draw_mHH_plane_2D_hists(
                    sample_hists,
                    sample_type,
                    f"mHH_plane_reco_{btag_count}btags_{btagger}_{pairing_id}_geq_370_GeV",
                    energy,
                    third_exp_label=pairing_info["label"].replace("pairing", ""),
                    log_z=True,
                    output_dir=output_dir,
                )
                # for region in ["signal", "control"]:
                for region in ["signal"]:
                    draw_mHH_plane_2D_hists(
                        sample_hists,
                        sample_type,
                        f"mHH_plane_reco_{region}_{btag_count}btags_{btagger}_{pairing_id}",
                        energy,
                        third_exp_label=pairing_info["label"].replace("pairing", ""),
                        log_z=True,
                        output_dir=output_dir,
                    )
                    draw_mHH_plane_2D_hists(
                        sample_hists,
                        sample_type,
                        f"mHH_plane_reco_{region}_{btag_count}btags_{btagger}_{pairing_id}_wrong_pairs",
                        energy,
                        third_exp_label=pairing_info["label"].replace("pairing", ""),
                        log_z=True,
                        output_dir=output_dir,
                    )
                    draw_mHH_plane_2D_hists(
                        sample_hists,
                        sample_type,
                        f"mHH_plane_reco_{region}_{btag_count}btags_{btagger}_{pairing_id}_lt_370_GeV",
                        energy,
                        third_exp_label=pairing_info["label"].replace("pairing", ""),
                        log_z=True,
                        output_dir=output_dir,
                    )
                    draw_mHH_plane_2D_hists(
                        sample_hists,
                        sample_type,
                        f"mHH_plane_reco_{region}_{btag_count}btags_{btagger}_{pairing_id}_geq_370_GeV",
                        energy,
                        third_exp_label=pairing_info["label"].replace("pairing", ""),
                        log_z=True,
                        output_dir=output_dir,
                    )

        # ##############################################
        # #### Pairing efficiency plots ################
        # ##############################################
        for btagger, btag_count in btagging.items():
            for hh_var, hh_var_label in hh_var_labels.items():
                draw_efficiency(
                    {sample_type: sample_hists},
                    [
                        {
                            "pass": f"{hh_var}_reco_{btag_count}btags_{btagger}_{pairing_id}_correct_pairs",
                            "total": f"{hh_var}_reco_{btag_count}btags_{btagger}_{pairing_id}",
                        }
                        for pairing_id in pairing_methods
                    ],
                    energy,
                    draw_errors=False,
                    luminosity=lumi,
                    xlabel=f"Reco {hh_var_label}",
                    legend_labels=[
                        pairing_methods[pairing_id]["label"]
                        for pairing_id in pairing_methods
                    ],
                    xmin=150 if "hh_mass" == hh_var else None,
                    xmax=600 if "hh_pt" == hh_var else None,
                    legend_options={"loc": "upper right", "fontsize": "small"},
                    third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
                    output_dir=output_dir,
                    plot_name=f"pairing_efficiency_reco_{hh_var}",
                )
                draw_efficiency(
                    {sample_type: sample_hists},
                    [
                        {
                            "pass": f"{hh_var}_reco_truth_matched_{btag_count}btags_{btagger}_{pairing_id}_correct_pairs",
                            "total": f"{hh_var}_reco_truth_matched_{btag_count}btags_{btagger}_{pairing_id}",
                        }
                        for pairing_id in pairing_methods
                    ],
                    energy,
                    luminosity=lumi,
                    draw_errors=False,
                    xlabel=f"Truth {hh_var_label}",
                    legend_labels=[
                        pairing_methods[pairing_id]["label"]
                        for pairing_id in pairing_methods
                    ],
                    xmin=200 if "hh_mass" == hh_var else None,
                    xmax=600 if "hh_pt" == hh_var else None,
                    legend_options={"loc": "upper right", "fontsize": "small"},
                    third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
                    output_dir=output_dir,
                    plot_name=f"pairing_efficiency_truth_{hh_var}",
                )

        # #### Pairing plots fraction of correct pairs vs m_HH ####
        # for pairing_id, pairing_info in pairing_methods.items():
        #     for btagger, btag_count in btagging.items():
        #         for pairing_id, pairing_info in pairing_methods.items():
        #             draw_1d_hists_v2(
        #                 {sample_type: sample_hists},
        #                 [
        #                     f"hh_mass_reco_truth_matched_{btag_count}btags_{btagger}_{pairing_id}",
        #                     f"hh_mass_reco_truth_matched_{btag_count}btags_{btagger}_{pairing_id}_correct_pairs",
        #                 ],
        #                 energy,
        #                 luminosity=lumi,
        #                 xlabel=hh_var_labels["hh_mass"],
        #                 ylabel="Events",
        #                 legend_labels=[pairing_info["label"], f"{pairing_info['label']} and parent ID 25"],
        #                 legend_options={"loc": "center right", "fontsize": "small"},
        #                 third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
        #                 xmin=100,
        #                 draw_ratio=True,
        #                 output_dir=output_dir,
        #                 plot_name=f"hh_mass_reco_truth_matched_pairing_efficiency_{pairing_id}",
        #             )

        # #### X_HH plots for different pairing methods ####
        # # create flat 2D distributions for variables m_H1 and m_H2 using the
        # # bins from the m_HH plane 0-200 GeV and 50x50 bins
        # bins_GeV = np.linspace(50, 200, 100)
        # X, Y = np.meshgrid(bins_GeV, bins_GeV)
        # base_discrim = X_HH(X.flatten(), Y.flatten())
        # for pairing_id, pairing_info in pairing_methods.items():
        #     for btagger, btag_count in btagging.items():
        #         for pairing_id, pairing_info in pairing_methods.items():
        #             for region in ["signal", "control"]:
        #                 draw_1d_hists_v2(
        #                     {sample_type: sample_hists},
        #                     [
        #                         f"hh_mass_discrim_reco_{region}_{btag_count}btags_{btagger}_{pairing}"
        #                         for pairing in pairing_methods
        #                     ],
        #                     energy,
        #                     luminosity=lumi,
        #                     xlabel=r"$\mathrm{X}_{\mathrm{HH}}$",
        #                     baseline=base_discrim,
        #                     normalize=True,
        #                     legend_labels=[
        #                         pairing_info["label"]
        #                         for pairing, pairing_info in pairing_methods.items()
        #                     ],
        #                     xmax=4,
        #                     legend_options={"loc": "upper right", "fontsize": "small"},
        #                     third_exp_label=f"\n{sample_labels[sample_type]}",
        #                     output_dir=output_dir,
        #                     plot_name=f"hh_mass_discrim_reco_{region}",
        #                 )

        # ##############################################
        # # TODO: Refactor for dynamic b-tagging WP
        # # Cutflow plots
        # ##############################################
        # for btagger, btag_count in btagging.items():
        #     draw_1d_hists_v2(
        #         {sample_type: sample_hists},
        #         [
        #             "hh_mass_truth",
        #             "hh_mass_truth_reco_central_jets_selection",
        #             "hh_mass_truth_reco_central_truth_matched_jets_selection",
        #             # "hh_mass_truth_reco_central_btagged_jets_selection",
        #             f"hh_mass_truth_reco_central_{btag_count}btags_{btagger}_jets_selection",
        #             # "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection",
        #             f"hh_mass_truth_reco_central_{btag_count}btags_{btagger}_4_plus_truth_matched_jets_selection",
        #             # "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection_v2",
        #             # "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_correct_min_deltar_pairing_selection",
        #         ],
        #         energy,
        #         luminosity=lumi,
        #         xlabel=f"Truth {hh_var_labels['hh_mass']}",
        #         ylabel="Events",
        #         legend_labels=[
        #             "Truth",
        #             selections_labels["central_jets"],
        #             selections_labels["truth_matched_4_plus_jets"],
        #             # selections_labels["btagged_GN277_4_jets"],
        #             selections_labels["truth_matched_4_plus_jets"],
        #             # r"$\geq$ 4 jets HadronConeExclTruthLabelID = 5",
        #             # f"correct pairs with {pairing_methods['min_deltar_pairing']}",
        #         ],
        #         legend_options={"loc": "center right", "fontsize": "small"},
        #         third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
        #         xmin=200,
        #         xmax=1000,
        #         draw_ratio=True,
        #         ymin_ratio=-0.1,
        #         ymax_ratio=1.2,
        #         plot_name="hh_mass_truth_cutflow",
        #         output_dir=output_dir,
        #     )

        ###############################################
        # Trigger efficiency plots
        ###############################################
        # for i in [1, 2, 3, 4]:
        #     categories = [
        #         "_truth_matched_2b2j_asym",
        #     ]
        #     for cat in categories:
        #         draw_1d_hists_v2(
        #             {sample_type: sample_hists},
        #             [
        #                 f"hh_jet_{i}_pt_truth_matched",
        #                 f"hh_jet_{i}_pt{cat}",
        #                 f"hh_jet_{i}_pt{cat}_n_btags",
        #                 f"hh_jet_{i}_pt{cat}_4_btags",
        #                 # f"hh_jet_{i}_pt_truth_matched_4_btags",
        #             ],
        #             energy,
        #             luminosity=lumi,
        #             xmin=0,
        #             xlabel="HH jet$_" + str(i) + "$ $p_{\mathrm{T}}$ [GeV]",
        #             ylabel="Events",
        #             legend_labels=[
        #                 r"$\geq$ 4 jets with $p_{\mathrm{T}} > 25$ GeV, $|\eta| < 2.5$, JVT",
        #                 "asym 2b2j trigger",
        #                 r"asym 2b2j and $\geq$ 2 jets passing GN2v01@77%",
        #                 r"asym 2b2j and $\geq$ 4 jets passing GN2v01@77%",
        #                 # r"$\geq$ 4 jets passing GN2v01@77%",
        #             ],
        #             legend_options={"loc": "center right", "fontsize": "small"},
        #             third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
        #             draw_ratio=True,
        #             output_dir=output_dir,
        #         )

    # if args.split_jz:
    #     for region in ["signal", "control"]:
    #         num_events_vs_sample(
    #             {
    #                 key: value
    #                 for key, value in sorted(hists_group.items())
    #                 if "multijet" in key
    #             },
    #             f"mH_plane_baseline_{region}_region$",
    #             energy,
    #             ylabel=f"{region.capitalize()} Events",
    #             xlabel="Multijet Samples",
    #             luminosity=lumi,
    #             density=True,
    #             third_exp_label=f"Signal Region",
    #             output_dir=output_dir,
    #         )
