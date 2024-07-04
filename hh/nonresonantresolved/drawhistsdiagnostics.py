import numpy as np
from argparse import Namespace
from hh.nonresonantresolved.pairing import pairing_methods
from hh.shared.selection import X_HH
from hh.shared.drawhists import (
    draw_1d_hists,
    draw_kin_hists,
    draw_mH_plane_2D_hists,
    draw_mH_1D_hists_v2,
    num_events_vs_sample,
    draw_1d_hists_v2,
    draw_efficiency,
)


def draw_hists(
    hists_group: dict,
    args: Namespace,
) -> None:
    """Draw all the histrograms"""

    luminosity = args.luminosity
    energy = args.energy
    output_dir = args.output_dir

    sample_labels = {
        "mc23a_ggF_k01": r"$\kappa_{\lambda}=1$ ggF MC23a",
        "mc23d_ggF_k01": r"$\kappa_{\lambda}=1$ ggF MC23d",
        "mc23a_ggF_k05": r"$\kappa_{\lambda}=5$ ggF MC23a",
        "mc23d_ggF_k05": r"$\kappa_{\lambda}=5$ ggF MC23d",
        "mc20a_multijet": "QCD b-filtered",
        "mc20a_ggF_k01": r"$\kappa_{\lambda}=1$ ggF MC20a",
        "mc20a_ggF_k10": r"$\kappa_{\lambda}=10$ ggF MC20a",
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

    ###############################################
    # Truth HH mass plots
    ###############################################
    draw_1d_hists(
        hists_group,
        "hh_mass_truth",
        energy,
        xlabel=f"Truth {hh_var_labels['hh_mass']}",
        legend_labels={key: sample_labels[key] for key in hists_group.keys()},
        luminosity=luminosity,
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
        luminosity=luminosity,
        xmin=100,
        ggFk01_factor=10,
        draw_errors=True,
        output_dir=output_dir,
    )

    for sample_type, sample_hists in hists_group.items():
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
            luminosity=luminosity,
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
            luminosity=luminosity,
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
            luminosity=luminosity,
            xlabel="HH mass response [%]",
            legend_labels={sample_type: sample_labels[sample_type]},
            third_exp_label=f"\n{selections_labels['truth_matching']}",
            output_dir=output_dir,
        )

        ###############################################
        # Cutflow plots
        ###############################################
        draw_1d_hists_v2(
            {sample_type: sample_hists},
            [
                "hh_mass_truth",
                "hh_mass_truth_reco_central_jets_selection",
                "hh_mass_truth_reco_central_truth_matched_jets_selection",
                "hh_mass_truth_reco_central_btagged_jets_selection",
                "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection",
                # "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection_v2",
                # "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_correct_min_deltar_pairing_selection",
            ],
            energy,
            luminosity=luminosity,
            xlabel=f"Truth {hh_var_labels['hh_mass']}",
            ylabel="Events",
            legend_labels={
                "hh_mass_truth": "Truth",
                "hh_mass_truth_reco_central_jets_selection": selections_labels[
                    "central_jets"
                ],
                "hh_mass_truth_reco_central_truth_matched_jets_selection": selections_labels[
                    "truth_matched_4_plus_jets"
                ],
                "hh_mass_truth_reco_central_btagged_jets_selection": selections_labels[
                    "btagged_GN277_4_jets"
                ],
                "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection": selections_labels[
                    "truth_matched_4_plus_jets"
                ],
                # "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection_v2": r"$\geq$ 4 jets HadronConeExclTruthLabelID = 5",
                # "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_correct_min_deltar_pairing_selection": f"correct pairs with {pairing_methods['min_deltar_pairing']}",
            },
            legend_options={"loc": "center right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
            xmin=200,
            xmax=1000,
            draw_ratio=True,
            ymin_ratio=-0.1,
            ymax_ratio=1.2,
            plot_name="hh_mass_truth_cutflow",
            output_dir=output_dir,
        )

        draw_1d_hists_v2(
            {sample_type: sample_hists},
            [
                "hh_mass_truth_unweighted",
                "hh_mass_truth_reco_central_jets_selection_unweighted",
                "hh_mass_truth_reco_central_truth_matched_jets_selection_unweighted",
                "hh_mass_truth_reco_central_btagged_jets_selection_unweighted",
                "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection_unweighted",
                "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection_v2_unweighted",
            ],
            energy,
            luminosity=luminosity,
            xlabel=f"Truth {hh_var_labels['hh_mass']}",
            ylabel="Events",
            legend_labels={
                "hh_mass_truth_unweighted": "Truth",
                "hh_mass_truth_reco_central_jets_selection_unweighted": selections_labels[
                    "central_jets"
                ],
                "hh_mass_truth_reco_central_truth_matched_jets_selection_unweighted": selections_labels[
                    "truth_matched_4_plus_jets"
                ],
                "hh_mass_truth_reco_central_btagged_jets_selection_unweighted": selections_labels[
                    "btagged_GN277_4_jets"
                ],
                "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection_unweighted": selections_labels[
                    "truth_matched_4_plus_jets"
                ],
                "hh_mass_truth_reco_central_btagged_4_plus_truth_matched_jets_selection_v2_unweighted": r"$\geq$ 4 jets HadronConeExclTruthLabelID = 5",
            },
            legend_options={"loc": "center right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]} (unweighted)\n{selections_labels['truth_matching']}",
            xmin=200,
            xmax=1000,
            draw_ratio=True,
            ymin_ratio=0,
            plot_name="hh_mass_truth_cutflow_unweighted",
            output_dir=output_dir,
        )

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
        #             luminosity=luminosity,
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

        ###############################################
        # Pairing efficiency plots
        ###############################################

        #### Pairing plots vs reco variables ####
        for hh_var, hh_var_label in hh_var_labels.items():
            draw_efficiency(
                {sample_type: sample_hists},
                [
                    [
                        f"{hh_var}_reco_{pairing_id}",
                        f"{hh_var}_reco_{pairing_id}_correct",
                    ]
                    for pairing_id in pairing_methods
                ],
                energy,
                luminosity=luminosity,
                xlabel=f"Reco {hh_var_label}",
                legend_labels={
                    f"{hh_var}_reco_{pairing_id}": pairing_info["label"]
                    for pairing_id, pairing_info in pairing_methods.items()
                },
                xmin=150 if "hh_mass" == hh_var else None,
                xmax=600 if "hh_pt" == hh_var else None,
                legend_options={"loc": "upper right", "fontsize": "small"},
                third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
                output_dir=output_dir,
                plot_name=f"pairing_efficiency_reco_{hh_var}",
            )

        #### Pairing plots vs truth variables ####
        for hh_var, hh_var_label in hh_var_labels.items():
            draw_efficiency(
                {sample_type: sample_hists},
                [
                    [
                        f"{hh_var}_reco_truth_matched_{pairing_id}",
                        f"{hh_var}_reco_truth_matched_{pairing_id}_correct",
                    ]
                    for pairing_id in pairing_methods
                ],
                energy,
                luminosity=luminosity,
                xlabel=f"Truth {hh_var_label}",
                legend_labels={
                    f"{hh_var}_reco_truth_matched_{pairing_id}": pairing_info["label"]
                    for pairing_id, pairing_info in pairing_methods.items()
                },
                xmin=200 if "hh_mass" == hh_var else None,
                xmax=600 if "hh_pt" == hh_var else None,
                legend_options={"loc": "upper right", "fontsize": "small"},
                third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
                output_dir=output_dir,
                plot_name=f"pairing_efficiency_truth_{hh_var}",
            )

        #### Pairing plots fraction of correct pairs vs m_HH ####
        for pairing_id, pairing_info in pairing_methods.items():
            draw_1d_hists_v2(
                {sample_type: sample_hists},
                [
                    f"hh_mass_reco_truth_matched_{pairing_id}",
                    f"hh_mass_reco_truth_matched_{pairing_id}_correct",
                ],
                energy,
                luminosity=luminosity,
                xlabel=hh_var_labels["hh_mass"],
                ylabel="Events",
                legend_labels={
                    f"hh_mass_reco_truth_matched_{pairing_id}": pairing_info["label"],
                    f"hh_mass_reco_truth_matched_{pairing_id}_correct": f"{pairing_info['label']} and parent ID 25",
                },
                legend_options={"loc": "center right", "fontsize": "small"},
                third_exp_label=f"\n{sample_labels[sample_type]}\n{selections_labels['truth_matching']}",
                xmin=100,
                draw_ratio=True,
                output_dir=output_dir,
                plot_name=f"hh_mass_reco_truth_matched_pairing_efficiency_{pairing_id}",
            )

        #### HH mass plane plots for different pairing methods ####
        for pairing_id, pairing_info in pairing_methods.items():
            draw_mH_plane_2D_hists(
                sample_hists,
                sample_type,
                f"mHH_plane_reco_{pairing_id}",
                energy,
                third_exp_label=pairing_info["label"].replace("pairing", "pairs"),
                log_z=True,
                output_dir=output_dir,
            )
            draw_mH_plane_2D_hists(
                sample_hists,
                sample_type,
                f"mHH_plane_reco_{pairing_id}_lt_300_GeV",
                energy,
                third_exp_label=pairing_info["label"].replace("pairing", "pairs"),
                log_z=False,
                output_dir=output_dir,
            )
            draw_mH_plane_2D_hists(
                sample_hists,
                sample_type,
                f"mHH_plane_reco_{pairing_id}_geq_300_GeV",
                energy,
                third_exp_label=pairing_info["label"].replace("pairing", "pairs"),
                log_z=False,
                output_dir=output_dir,
            )

        #### X_HH plots for different pairing methods ####
        # create flat 2D distributions for variables m_H1 and m_H2 using the
        # bins from the m_HH plane 0-200 GeV and 50x50 bins
        bins_GeV = np.linspace(50, 200, 100)
        X, Y = np.meshgrid(bins_GeV, bins_GeV)
        base_discrim = X_HH(X.flatten(), Y.flatten())
        for region in ["signal", "control"]:
            draw_1d_hists_v2(
                {sample_type: sample_hists},
                [
                    f"hh_mass_discrim_reco_{region}_{pairing}"
                    for pairing in pairing_methods
                ],
                energy,
                luminosity=luminosity,
                xlabel=r"$\mathrm{X}_{\mathrm{HH}}$",
                baseline=base_discrim,
                normalize=True,
                legend_labels={
                    f"hh_mass_discrim_reco_{region}_{pairing}": pairing_info["label"]
                    for pairing, pairing_info in pairing_methods.items()
                },
                xmax=4,
                legend_options={"loc": "upper right", "fontsize": "small"},
                third_exp_label=f"\n{sample_labels[sample_type]}",
                output_dir=output_dir,
                plot_name=f"hh_mass_discrim_reco_{region}",
            )

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
    #             luminosity=luminosity,
    #             density=True,
    #             third_exp_label=f"Signal Region",
    #             output_dir=output_dir,
    #         )
