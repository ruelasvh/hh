import numpy as np
from argparse import Namespace
from hh.nonresonantresolved.pairing import pairing_methods
from hh.shared.selection import X_HH
from hh.shared.labels import sample_labels, hh_var_labels, selections_labels
from hh.shared.drawhists import (
    draw_1d_hists,
    draw_kin_hists,
    draw_mH_plane_2D_hists,
    draw_mH_1D_hists_v2,
    num_events_vs_sample,
    draw_1d_hists_v2,
    draw_efficiency,
    draw_signal_vs_background,
    draw_efficiency_scan_2d,
    draw_efficiency_scan_1d,
)


def draw_hists(
    hists_group: dict,
    args: Namespace,
) -> None:
    """Draw all the histrograms"""

    luminosity = args.luminosity
    energy = args.energy
    output_dir = args.output_dir
    btagging = {"GN2v01_77": 4}
    mc_campaigns = {"mc20a": 36.64674, "mc20d": 44.6306, "mc20e": 58.7916}

    ###############################################
    # Truth HH mass plots
    ###############################################
    draw_1d_hists(
        hists_group,
        "hh_mass_truth",
        energy,
        xlabel=f"Truth {hh_var_labels['hh_mass']}",
        legend_labels={key: sample_labels[key] for key in hists_group.keys()},
        luminosity=sum(mc_campaigns.values()),
        xmin=100,
        ggFk01_factor=10,
        draw_errors=True,
        output_dir=output_dir,
    )

    ###############################################
    # Reco truth-matched HH mass plots
    ###############################################
    draw_1d_hists(
        hists_group,
        f"hh_mass_reco_truth_matched",
        energy,
        xlabel=hh_var_labels["hh_mass"],
        ylabel="Events",
        legend_labels={key: sample_labels[key] for key in hists_group.keys()},
        third_exp_label=f"\n{selections_labels['truth_matching']}",
        luminosity=sum(mc_campaigns.values()),
        xmin=100,
        ggFk01_factor=10,
        draw_errors=True,
        output_dir=output_dir,
    )

    ###############################################
    #  Pairing efficiency scan plots
    ###############################################
    for btagger, btag_count in btagging.items():
        for hh_var, hh_var_label in hh_var_labels.items():
            pairing_key = "min_mass_optimized_1d_pairing"
            pairing_info = pairing_methods[pairing_key]
            m_X_range = pairing_info["m_X_range"]
            # if not emtpy do the 1D scan
            if len(m_X_range) == 0:
                continue
            hist_reco_keys = [
                [
                    f"{hh_var}_reco_{btag_count}_btag_{btagger}_{pairing_key}_m_X_{m_X}",
                    f"{hh_var}_reco_{btag_count}_btag_{btagger}_{pairing_key}_m_X_{m_X}_correct",
                ]
                # for m_X in [m_X_range[0], m_X_range[-1]]
                for m_X in m_X_range
            ]
            hist_reco_labels = {
                f"{hh_var}_reco_{btag_count}_btag_{btagger}_{pairing_key}_m_X_{m_X}": f"m_X = {m_X}"
                # for m_X in [m_X_range[0], m_X_range[-1]]
                for m_X in m_X_range
            }
            draw_efficiency_scan_1d(
                hists_group,
                hist_reco_keys,
                energy,
                luminosity=sum(mc_campaigns.values()),
                xlabel=f"Reco {hh_var_label}",
                legend_labels=hist_reco_labels,
                xmin=150 if "hh_mass" == hh_var else None,
                xmax=600 if "hh_pt" == hh_var else None,
                legend_options={"loc": "upper right", "fontsize": "small"},
                third_exp_label=f"\n{selections_labels['truth_matching']}\n{pairing_info['label']}",
                output_dir=output_dir,
                plot_name=f"pairing_efficiency_reco_{hh_var}",
            )
            hist_truth_keys = [
                [
                    f"{hh_var}_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_key}_m_X_{m_X}",
                    f"{hh_var}_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_key}_m_X_{m_X}_correct",
                ]
                # for m_X in [m_X_range[0], m_X_range[-1]]
                for m_X in m_X_range
            ]
            hist_truth_labels = {
                f"{hh_var}_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_key}_m_X_{m_X}": f"m_X = {m_X}"
                # for m_X in [m_X_range[0], m_X_range[-1]]
                for m_X in m_X_range
            }
            draw_efficiency_scan_1d(
                hists_group,
                hist_truth_keys,
                energy,
                luminosity=sum(mc_campaigns.values()),
                xlabel=f"Truth {hh_var_label}",
                legend_labels=hist_truth_labels,
                xmin=200 if "hh_mass" == hh_var else None,
                xmax=600 if "hh_pt" == hh_var else None,
                legend_options={"loc": "upper right", "fontsize": "small"},
                third_exp_label=f"\n{selections_labels['truth_matching']}\n{pairing_info['label']}",
                output_dir=output_dir,
                plot_name=f"pairing_efficiency_truth_{hh_var}",
            )

    for sample_type, sample_hists in hists_group.items():
        for btagger, btag_count in btagging.items():
            for hh_var, hh_var_label in hh_var_labels.items():
                pairing_key = "min_mass_optimized_pairing"
                pairing_info = pairing_methods[pairing_key]
                m_X_lead_range, m_X_sub_range = (
                    pairing_info["m_X_lead_range"],
                    pairing_info["m_X_sub_range"],
                )
                # only do the 2D scan if the ranges are not empty
                if len(m_X_lead_range) == 0 and len(m_X_sub_range) == 0:
                    continue
                hist_reco_keys = [
                    [
                        f"{hh_var}_reco_{btag_count}_btag_{btagger}_{pairing_key}_m_X_lead_{m_X_lead}_m_X_sub_{m_X_sub}",
                        f"{hh_var}_reco_{btag_count}_btag_{btagger}_{pairing_key}_m_X_lead_{m_X_lead}_m_X_sub_{m_X_sub}_correct",
                    ]
                    for m_X_lead in m_X_lead_range
                    for m_X_sub in m_X_sub_range
                ]
                hist_reco_labels = {
                    f"{hh_var}_reco_{btag_count}_btag_{btagger}_{pairing_key}_m_X_lead_{m_X_lead}_m_X_sub_{m_X_sub}": f"m_X_lead = {m_X_lead}, m_X_sub = {m_X_sub}"
                    for m_X_lead in m_X_lead_range
                    for m_X_sub in m_X_sub_range
                }
                draw_efficiency_scan_2d(
                    {sample_type: sample_hists},
                    hist_reco_keys,
                    energy,
                    luminosity=sum(mc_campaigns.values()),
                    xlabel=f"Reco {hh_var_label}",
                    legend_labels=hist_reco_labels,
                    xmin=150 if "hh_mass" == hh_var else None,
                    xmax=600 if "hh_pt" == hh_var else None,
                    legend_options={"loc": "lower right", "fontsize": "small"},
                    third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}\n{pairing_info['label']}",
                    output_dir=output_dir,
                    plot_name=f"pairing_efficiency_reco_{hh_var}",
                )
                hist_truth_keys = [
                    [
                        f"{hh_var}_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_key}_m_X_lead_{m_X_lead}_m_X_sub_{m_X_sub}",
                        f"{hh_var}_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_key}_m_X_lead_{m_X_lead}_m_X_sub_{m_X_sub}_correct",
                    ]
                    for m_X_lead in m_X_lead_range
                    for m_X_sub in m_X_sub_range
                ]
                hist_truth_labels = {
                    f"{hh_var}_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_key}_m_X_lead_{m_X_lead}_m_X_sub_{m_X_sub}": f"m_X_lead = {m_X_lead}, m_X_sub = {m_X_sub}"
                    for m_X_lead in m_X_lead_range
                    for m_X_sub in m_X_sub_range
                }
                draw_efficiency_scan_2d(
                    {sample_type: sample_hists},
                    hist_truth_keys,
                    energy,
                    luminosity=sum(mc_campaigns.values()),
                    xlabel=f"Truth {hh_var_label}",
                    legend_labels=hist_truth_labels,
                    xmin=200 if "hh_mass" == hh_var else None,
                    xmax=600 if "hh_pt" == hh_var else None,
                    legend_options={"loc": "lower right", "fontsize": "small"},
                    third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}\n{pairing_info['label']}",
                    output_dir=output_dir,
                    plot_name=f"pairing_efficiency_truth_{hh_var}",
                )

    ################################################
    # Plot backgrounds vs signal
    ################################################
    for mc, lumi in mc_campaigns.items():
        mc_samples = {
            sample: hists for sample, hists in hists_group.items() if mc in sample
        }
        mc_kl1_samples = {
            sample: hists for sample, hists in mc_samples.items() if "ggF_k01" in sample
        }
        mc_background_samples = {
            sample: hists
            for sample, hists in mc_samples.items()
            if any(bkg in sample for bkg in ["ttbar", "multijet"])
        }
        # for region in ["signal", "control"]:
        # for region in ["signal"]:
        #     for pairing in pairing_methods:
        #         draw_signal_vs_background(
        #             f"hh_mass_reco_{region}_4b_GN2v01_77_{pairing}_bins_logscale",
        #             signal=mc_kl1_samples,
        #             background=mc_background_samples,
        #             energy=energy,
        #             luminosity=lumi,
        #             xlabel=hh_var_labels["hh_mass"],
        #             legend_labels={
        #                 **{key: sample_labels[key] for key in mc_kl1_samples.keys()},
        #                 "Background MC": "Background MC",
        #             },
        #             third_exp_label="\n".join(
        #                 [
        #                     f"\n4b {region.capitalize()} Region",
        #                     f"{pairing_methods[pairing]['label']}",
        #                 ]
        #             ),
        #             plot_name=f"{mc}_sig_vs_bkg_{region}_region_4b_GN2v01_77_{pairing}_bins_logscale",
        #             output_dir=output_dir,
        #             show_counts=True,
        #         )

    for sample_type, sample_hists in hists_group.items():
        # sample_lumi = [mc_campaigns[mc] for mc in mc_campaigns if mc in sample_type][0]
        # rewrite above line to sum the campaings if there is not a match
        sample_lumi = sum(
            [
                mc_campaigns[mc] if mc in sample_type else mc_campaigns[mc]
                for mc in mc_campaigns
            ]
        )
        ########################################################
        # HH mass plots truth reco-matched vs reco truth-matched
        ########################################################
        draw_1d_hists_v2(
            {sample_type: sample_hists},
            [
                "hh_mass_truth_reco_matched",
                "hh_mass_reco_truth_matched",
            ],
            energy,
            luminosity=sample_lumi,
            xlabel=hh_var_labels["hh_mass"],
            ylabel="Events",
            legend_labels={
                "hh_mass_truth_reco_matched": "Truth (reco-matched)",
                "hh_mass_reco_truth_matched": "Reco (truth-matched)",
            },
            legend_options={"loc": "center right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
            xmin=100,
            draw_errors=True,
            plot_name="hh_mass_truth_vs_reco_truth_matched",
            output_dir=output_dir,
        )

        ########################################################
        # HH mass plots truth reco-matched vs reco truth-matched
        # v2 uses HadronConeExclTruthLabelID = 5 to match jets
        ########################################################
        draw_1d_hists_v2(
            {sample_type: sample_hists},
            [
                "hh_mass_reco_truth_matched",
                "hh_mass_reco_truth_matched_v2",
            ],
            energy,
            luminosity=sample_lumi,
            xlabel=hh_var_labels["hh_mass"],
            ylabel="Events",
            legend_labels={
                "hh_mass_reco_truth_matched": "Reco (truth-matched)",
                "hh_mass_reco_truth_matched_v2": r"$\geq 4$ HadronConeExclTruthLabelID = 5",
            },
            legend_options={"loc": "center right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
            xmin=100,
            draw_ratio=True,
            ymin_ratio=0.5,
            ymax_ratio=2,
            plot_name="hh_mass_truth_matching_methods",
            output_dir=output_dir,
        )

        ###############################################
        # HH mass response plots reco vs truth
        ###############################################
        draw_1d_hists(
            {sample_type: sample_hists},
            "hh_mass_reco_vs_truth_response",
            energy,
            luminosity=sample_lumi,
            xlabel="HH mass response [%]",
            legend_labels={sample_type: sample_labels[sample_type]},
            third_exp_label=f"\n{selections_labels['truth_matching']}",
            output_dir=output_dir,
        )

        # #### HH mass plane plots for different pairing methods ####
        # for pairing_id, pairing_info in pairing_methods.items():
        #     for btagger, btag_count in btagging.items():
        #         draw_mH_plane_2D_hists(
        #             sample_hists,
        #             sample_type,
        #             f"mHH_plane_reco_{btag_count}b_{btagger}_{pairing_id}",
        #             energy,
        #             third_exp_label=pairing_info["label"].replace("pairing", "pairs"),
        #             log_z=True,
        #             output_dir=output_dir,
        #         )
        #         draw_mH_plane_2D_hists(
        #             sample_hists,
        #             sample_type,
        #             f"mHH_plane_reco_{btag_count}b_{btagger}_{pairing_id}_lt_300_GeV",
        #             energy,
        #             third_exp_label=pairing_info["label"].replace("pairing", "pairs"),
        #             log_z=False,
        #             output_dir=output_dir,
        #         )
        #         draw_mH_plane_2D_hists(
        #             sample_hists,
        #             sample_type,
        #             f"mHH_plane_reco_{btag_count}b_{btagger}_{pairing_id}_geq_300_GeV",
        #             energy,
        #             third_exp_label=pairing_info["label"].replace("pairing", "pairs"),
        #             log_z=False,
        #             output_dir=output_dir,
        #         )

        # # ##############################################
        # # #### Pairing efficiency plots ################
        # # ##############################################

        # #### Pairing plots vs reco variables ####
        # for btagger, btag_count in btagging.items():
        #     for hh_var, hh_var_label in hh_var_labels.items():
        #         draw_efficiency(
        #             {sample_type: sample_hists},
        #             [
        #                 [
        #                     f"{hh_var}_reco_{btag_count}_btag_{btagger}_{pairing_id}",
        #                     f"{hh_var}_reco_{btag_count}_btag_{btagger}_{pairing_id}_correct",
        #                 ]
        #                 for pairing_id in pairing_methods
        #             ],
        #             energy,
        #             luminosity=sample_lumi,
        #             xlabel=f"Reco {hh_var_label}",
        #             legend_labels={
        #                 f"{hh_var}_reco_{btag_count}_btag_{btagger}_{pairing_id}": pairing_info[
        #                     "label"
        #                 ]
        #                 for pairing_id, pairing_info in pairing_methods.items()
        #             },
        #             xmin=150 if "hh_mass" == hh_var else None,
        #             xmax=600 if "hh_pt" == hh_var else None,
        #             legend_options={"loc": "upper right", "fontsize": "small"},
        #             third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
        #             output_dir=output_dir,
        #             plot_name=f"pairing_efficiency_reco_{hh_var}",
        #         )

        # #### Pairing plots vs truth variables ####
        # for btagger, btag_count in btagging.items():
        #     for hh_var, hh_var_label in hh_var_labels.items():
        #         draw_efficiency(
        #             {sample_type: sample_hists},
        #             [
        #                 [
        #                     f"{hh_var}_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_id}",
        #                     f"{hh_var}_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_id}_correct",
        #                 ]
        #                 for pairing_id in pairing_methods
        #             ],
        #             energy,
        #             luminosity=sample_lumi,
        #             xlabel=f"Truth {hh_var_label}",
        #             legend_labels={
        #                 f"{hh_var}_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_id}": pairing_info[
        #                     "label"
        #                 ]
        #                 for pairing_id, pairing_info in pairing_methods.items()
        #             },
        #             xmin=200 if "hh_mass" == hh_var else None,
        #             xmax=600 if "hh_pt" == hh_var else None,
        #             legend_options={"loc": "upper right", "fontsize": "small"},
        #             third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
        #             output_dir=output_dir,
        #             plot_name=f"pairing_efficiency_truth_{hh_var}",
        #         )

        # #### Pairing plots fraction of correct pairs vs m_HH ####
        # for pairing_id, pairing_info in pairing_methods.items():
        #     for btagger, btag_count in btagging.items():
        #         for pairing_id, pairing_info in pairing_methods.items():
        #             draw_1d_hists_v2(
        #                 {sample_type: sample_hists},
        #                 [
        #                     f"hh_mass_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_id}",
        #                     f"hh_mass_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_id}_correct",
        #                 ],
        #                 energy,
        #                 luminosity=sample_lumi,
        #                 xlabel=hh_var_labels["hh_mass"],
        #                 ylabel="Events",
        #                 legend_labels={
        #                     f"hh_mass_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_id}": pairing_info[
        #                         "label"
        #                     ],
        #                     f"hh_mass_reco_truth_matched_{btag_count}_btag_{btagger}_{pairing_id}_correct": f"{pairing_info['label']} and parent ID 25",
        #                 },
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
        #                         f"hh_mass_discrim_reco_{region}_{btag_count}b_{btagger}_{pairing}"
        #                         for pairing in pairing_methods
        #                     ],
        #                     energy,
        #                     luminosity=sample_lumi,
        #                     xlabel=r"$\mathrm{X}_{\mathrm{HH}}$",
        #                     baseline=base_discrim,
        #                     normalize=True,
        #                     legend_labels={
        #                         f"hh_mass_discrim_reco_{region}_{btag_count}b_{btagger}_{pairing}": pairing_info[
        #                             "label"
        #                         ]
        #                         for pairing, pairing_info in pairing_methods.items()
        #                     },
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
        #             f"hh_mass_truth_reco_central_{btag_count}b_{btagger}_jets_selection",
        #             # "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection",
        #             f"hh_mass_truth_reco_central_{btag_count}b_{btagger}_4_plus_truth_matched_jets_selection",
        #             # "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection_v2",
        #             # "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_correct_min_deltar_pairing_selection",
        #         ],
        #         energy,
        #         luminosity=sample_lumi,
        #         xlabel=f"Truth {hh_var_labels['hh_mass']}",
        #         ylabel="Events",
        #         legend_labels={
        #             "hh_mass_truth": "Truth",
        #             "hh_mass_truth_reco_central_jets_selection": selections_labels[
        #                 "central_jets"
        #             ],
        #             "hh_mass_truth_reco_central_truth_matched_jets_selection": selections_labels[
        #                 "truth_matched_4_plus_jets"
        #             ],
        #             f"hh_mass_truth_reco_central_{btag_count}b_{btagger}_jets_selection": selections_labels[
        #                 "btagged_GN277_4_jets"
        #             ],
        #             f"hh_mass_truth_reco_central_{btag_count}b_{btagger}_4_plus_truth_matched_jets_selection": selections_labels[
        #                 "truth_matched_4_plus_jets"
        #             ],
        #             # "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection_v2": r"$\geq$ 4 jets HadronConeExclTruthLabelID = 5",
        #             # "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_correct_min_deltar_pairing_selection": f"correct pairs with {pairing_methods['min_deltar_pairing']}",
        #         },
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
        #             luminosity=sample_lumi,
        #             xmin=0,
        #             xlabel="HH jet$_" + str(i) + "$ $p_{\mathrm{T}}$ [GeV]",
        #             ylabel="Events",
        #             legend_labels={
        #                 f"hh_jet_{i}_pt_truth_matched": r"$\geq$ 4 jets with $p_{\mathrm{T}} > 25$ GeV, $|\eta| < 2.5$, JVT",
        #                 f"hh_jet_{i}_pt{cat}": "asym 2b2j trigger",
        #                 f"hh_jet_{i}_pt{cat}_n_btags": r"asym 2b2j and $\geq$ 2 jets passing GN2v01@77%",
        #                 f"hh_jet_{i}_pt{cat}_4_btags": r"asym 2b2j and $\geq$ 4 jets passing GN2v01@77%",
        #                 # f"hh_jet_{i}_pt_truth_matched_4_btags": r"$\geq$ 4 jets passing GN2v01@77%",
        #             },
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
    #             luminosity=sample_lumi,
    #             density=True,
    #             third_exp_label=f"Signal Region",
    #             output_dir=output_dir,
    #         )
