from argparse import Namespace
from hh.shared.utils import logger
from hh.shared.drawhists import (
    draw_1d_hists,
    draw_kin_hists,
    draw_mH_plane_2D_hists,
    draw_mH_1D_hists_v2,
    num_events_vs_sample,
    draw_truth_vs_reco_truth_matched,
    draw_efficiency,
)


def draw_hists(
    hists_group: dict,
    args: Namespace,
) -> None:
    """Draw all the histrograms"""

    btag = args.btag or ""
    luminosity = args.luminosity
    energy = args.energy
    output_dir = args.output_dir

    sample_labels = {
        "mc23a_ggF_k01": "kl=1 ggF MC23a",
        "mc23a_ggF_k05": "kl=5 ggF MC23a",
        # "mc23d_ggF_k01": "kl=1 ggF MC23d",
        # "mc23d_ggF_k05": "kl=5 ggF MC23d",
    }

    legend_labels = {
        "min_deltar_pairing": r"min $\Delta R_{\mathrm{jj}}^{\mathrm{HC1}}$ pairing",
        "max_deltar_pairing": r"max $\Delta R_{\mathrm{jj}}^{\mathrm{HC1}}$ pairing",
        "min_mass_true_pairing": r"min $\Sigma(m_{jj}-m_H)^2$ pairing",
        "min_mass_pairing": r"min $(m_{jj}-m_{jj})^2$ pairing",
        "truth_matching": r"Truth-matched $\Delta R < 0.3$ jets",
    }

    draw_1d_hists(
        hists_group,
        "hh_mass_truth",
        energy,
        xlabel="Truth HH mass [GeV]",
        ylabel="Events",
        legend_labels=sample_labels,
        luminosity=luminosity,
        xmin=100,
        ggFk01_factor=10,
        draw_errors=True,
        output_dir=output_dir,
    )

    draw_1d_hists(
        hists_group,
        f"hh_mass_reco_truth_matched",
        energy,
        xlabel="Reconstructed Truth-Matched HH Mass [GeV]",
        ylabel="Events",
        legend_labels=sample_labels,
        third_exp_label=f"\n{legend_labels['truth_matching']}",
        luminosity=luminosity,
        xmin=100,
        ggFk01_factor=10,
        draw_errors=True,
        output_dir=output_dir,
    )

    for sample_type, sample_hists in hists_group.items():
        draw_truth_vs_reco_truth_matched(
            {sample_type: sample_hists},
            [
                "hh_mass_truth_reco_matched",
                "hh_mass_reco_truth_matched",
            ],
            energy,
            luminosity=luminosity,
            xlabel="$m_{\mathrm{HH}}$ [GeV]",
            ylabel="Events",
            legend_labels={
                "hh_mass_truth_reco_matched": "Truth (reco-matched)",
                "hh_mass_reco_truth_matched": "Reco (truth-matched)",
            },
            legend_options={"loc": "center right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{legend_labels['truth_matching']}",
            xmin=100,
            draw_errors=True,
            output_dir=output_dir,
        )

        draw_truth_vs_reco_truth_matched(
            {sample_type: sample_hists},
            [
                "hh_mass_reco_truth_matched",
                "hh_mass_reco_truth_matched_v2",
            ],
            energy,
            luminosity=luminosity,
            xlabel="$m_{\mathrm{HH}}$ [GeV]",
            ylabel="Events",
            legend_labels={
                "hh_mass_reco_truth_matched": "Reco (truth-matched)",
                "hh_mass_reco_truth_matched_v2": r"$\geq 4$ HadronConeExclTruthLabelID = 5",
            },
            legend_options={"loc": "center right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{legend_labels['truth_matching']}",
            xmin=100,
            draw_ratio=True,
            ymin_ratio=0.5,
            ymax_ratio=2,
            output_dir=output_dir,
        )

        draw_1d_hists(
            {sample_type: sample_hists},
            "hh_mass_reco_vs_truth_response",
            energy,
            luminosity=luminosity,
            xlabel="HH mass response [%]",
            legend_labels={sample_type: sample_labels[sample_type]},
            third_exp_label=f"\n{legend_labels['truth_matching']}",
            output_dir=output_dir,
        )

        for i in [1, 2, 3, 4]:
            categories = [
                "_truth_matched_2b2j_asym",
            ]
            for cat in categories:
                draw_truth_vs_reco_truth_matched(
                    {sample_type: sample_hists},
                    [
                        f"hh_jet_{i}_pt_truth_matched",
                        f"hh_jet_{i}_pt{cat}",
                        f"hh_jet_{i}_pt{cat}_n_btags",
                        f"hh_jet_{i}_pt{cat}_4_btags",
                        # f"hh_jet_{i}_pt_truth_matched_4_btags",
                    ],
                    energy,
                    luminosity=luminosity,
                    xmin=0,
                    xlabel="HH jet$_" + str(i) + "$ $p_{\mathrm{T}}$ [GeV]",
                    ylabel="Events",
                    legend_labels={
                        f"hh_jet_{i}_pt_truth_matched": r"$\geq$ 4 jets with $p_{\mathrm{T}} > 25$ GeV, $|\eta| < 2.5$, JVT",
                        f"hh_jet_{i}_pt{cat}": "asym 2b2j trigger",
                        f"hh_jet_{i}_pt{cat}_n_btags": r"asym 2b2j and $\geq$ 2 jets passing GN2v01@77%",
                        f"hh_jet_{i}_pt{cat}_4_btags": r"asym 2b2j and $\geq$ 4 jets passing GN2v01@77%",
                        # f"hh_jet_{i}_pt_truth_matched_4_btags": r"$\geq$ 4 jets passing GN2v01@77%",
                    },
                    legend_options={"loc": "center right", "fontsize": "small"},
                    third_exp_label=f"\n{sample_labels[sample_type]}\n{legend_labels['truth_matching']}",
                    draw_ratio=True,
                    output_dir=output_dir,
                )

        #### Pairing plots vs reco variables ####
        draw_efficiency(
            {sample_type: sample_hists},
            [
                [
                    "hh_mass_reco_min_deltar_pairing",
                    "hh_mass_reco_min_deltar_pairing_correct",
                ],
                [
                    "hh_mass_reco_max_deltar_pairing",
                    "hh_mass_reco_max_deltar_pairing_correct",
                ],
                [
                    "hh_mass_reco_min_mass_true_pairing",
                    "hh_mass_reco_min_mass_true_pairing_correct",
                ],
                [
                    "hh_mass_reco_min_mass_pairing",
                    "hh_mass_reco_min_mass_pairing_correct",
                ],
            ],
            energy,
            luminosity=luminosity,
            xlabel="Reco $m_{\mathrm{HH}}$ [GeV]",
            legend_labels={
                "hh_mass_reco_min_deltar_pairing": legend_labels["min_deltar_pairing"],
                "hh_mass_reco_max_deltar_pairing": legend_labels["max_deltar_pairing"],
                "hh_mass_reco_min_mass_true_pairing": legend_labels[
                    "min_mass_true_pairing"
                ],
                "hh_mass_reco_min_mass_pairing": legend_labels["min_mass_pairing"],
            },
            legend_options={"loc": "upper right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{legend_labels['truth_matching']}",
            xmin=200,
            xmax=1000,
            output_dir=output_dir,
            plot_name="pairing_efficiency_reco_mHH",
        )

        draw_efficiency(
            {sample_type: sample_hists},
            [
                [
                    "hh_pt_reco_min_deltar_pairing",
                    "hh_pt_reco_min_deltar_pairing_correct",
                ],
                [
                    "hh_pt_reco_max_deltar_pairing",
                    "hh_pt_reco_max_deltar_pairing_correct",
                ],
                [
                    "hh_pt_reco_min_mass_true_pairing",
                    "hh_pt_reco_min_mass_true_pairing_correct",
                ],
                [
                    "hh_pt_reco_min_mass_pairing",
                    "hh_pt_reco_min_mass_pairing_correct",
                ],
            ],
            energy,
            luminosity=luminosity,
            xlabel="Reco $p_{\mathrm{T}}$ (HH) [GeV]",
            legend_labels={
                "hh_pt_reco_min_deltar_pairing": legend_labels["min_deltar_pairing"],
                "hh_pt_reco_max_deltar_pairing": legend_labels["max_deltar_pairing"],
                "hh_pt_reco_min_mass_true_pairing": legend_labels[
                    "min_mass_true_pairing"
                ],
                "hh_pt_reco_min_mass_pairing": legend_labels["min_mass_pairing"],
            },
            legend_options={"loc": "upper right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{legend_labels['truth_matching']}",
            # xmin=200,
            # xmax=1000,
            output_dir=output_dir,
            plot_name="pairing_efficiency_reco_pt_HH",
        )

        #### Pairing plots vs truth variables ####
        draw_efficiency(
            {sample_type: sample_hists},
            [
                [
                    "hh_mass_reco_truth_matched_min_deltar_pairing",
                    "hh_mass_reco_truth_matched_min_deltar_pairing_correct",
                ],
                [
                    "hh_mass_reco_truth_matched_max_deltar_pairing",
                    "hh_mass_reco_truth_matched_max_deltar_pairing_correct",
                ],
                [
                    "hh_mass_reco_truth_matched_min_mass_true_pairing",
                    "hh_mass_reco_truth_matched_min_mass_true_pairing_correct",
                ],
                [
                    "hh_mass_reco_truth_matched_min_mass_pairing",
                    "hh_mass_reco_truth_matched_min_mass_pairing_correct",
                ],
            ],
            energy,
            luminosity=luminosity,
            xlabel="Truth $m_{\mathrm{HH}}$ [GeV]",
            legend_labels={
                "hh_mass_reco_truth_matched_min_deltar_pairing": legend_labels[
                    "min_deltar_pairing"
                ],
                "hh_mass_reco_truth_matched_max_deltar_pairing": legend_labels[
                    "max_deltar_pairing"
                ],
                "hh_mass_reco_truth_matched_min_mass_true_pairing": legend_labels[
                    "min_mass_true_pairing"
                ],
                "hh_mass_reco_truth_matched_min_mass_pairing": legend_labels[
                    "min_mass_pairing"
                ],
            },
            legend_options={"loc": "upper right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{legend_labels['truth_matching']}",
            xmin=200,
            xmax=1000,
            output_dir=output_dir,
            plot_name="pairing_efficiency_truth_mHH",
        )

        draw_efficiency(
            {sample_type: sample_hists},
            [
                [
                    "hh_pt_reco_truth_matched_min_deltar_pairing",
                    "hh_pt_reco_truth_matched_min_deltar_pairing_correct",
                ],
                [
                    "hh_pt_reco_truth_matched_max_deltar_pairing",
                    "hh_pt_reco_truth_matched_max_deltar_pairing_correct",
                ],
                [
                    "hh_pt_reco_truth_matched_min_mass_true_pairing",
                    "hh_pt_reco_truth_matched_min_mass_true_pairing_correct",
                ],
                [
                    "hh_pt_reco_truth_matched_min_mass_pairing",
                    "hh_pt_reco_truth_matched_min_mass_pairing_correct",
                ],
            ],
            energy,
            luminosity=luminosity,
            xlabel="Truth $p_{\mathrm{T}}$ (HH) [GeV]",
            legend_labels={
                "hh_pt_reco_truth_matched_min_deltar_pairing": legend_labels[
                    "min_deltar_pairing"
                ],
                "hh_pt_reco_truth_matched_max_deltar_pairing": legend_labels[
                    "max_deltar_pairing"
                ],
                "hh_pt_reco_truth_matched_min_mass_true_pairing": legend_labels[
                    "min_mass_true_pairing"
                ],
                "hh_pt_reco_truth_matched_min_mass_pairing": legend_labels[
                    "min_mass_pairing"
                ],
            },
            legend_options={"loc": "upper right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{legend_labels['truth_matching']}",
            # xmin=200,
            # xmax=1000,
            output_dir=output_dir,
            plot_name="pairing_efficiency_truth_pt_HH",
        )

        draw_truth_vs_reco_truth_matched(
            {sample_type: sample_hists},
            [
                "hh_mass_reco_truth_matched_min_deltar_pairing",
                "hh_mass_reco_truth_matched_min_deltar_pairing_correct",
            ],
            energy,
            luminosity=luminosity,
            xlabel="$m_{\mathrm{HH}}$ [GeV]",
            ylabel="Events",
            legend_labels={
                "hh_mass_reco_truth_matched_min_deltar_pairing": legend_labels[
                    "min_deltar_pairing"
                ],
                "hh_mass_reco_truth_matched_min_deltar_pairing_correct": f"{legend_labels['min_deltar_pairing']} and parent ID 25",
            },
            legend_options={"loc": "center right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{legend_labels['truth_matching']}",
            xmin=100,
            # draw_errors=True,
            draw_ratio=True,
            output_dir=output_dir,
        )

        draw_truth_vs_reco_truth_matched(
            {sample_type: sample_hists},
            [
                "hh_mass_reco_truth_matched_max_deltar_pairing",
                "hh_mass_reco_truth_matched_max_deltar_pairing_correct",
            ],
            energy,
            luminosity=luminosity,
            xlabel="$m_{\mathrm{HH}}$ [GeV]",
            ylabel="Events",
            legend_labels={
                "hh_mass_reco_truth_matched_max_deltar_pairing": legend_labels[
                    "max_deltar_pairing"
                ],
                "hh_mass_reco_truth_matched_max_deltar_pairing_correct": f"{legend_labels['max_deltar_pairing']} and parent ID 25",
            },
            legend_options={"loc": "center right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{legend_labels['truth_matching']}",
            xmin=100,
            # draw_errors=True,
            draw_ratio=True,
            output_dir=output_dir,
        )

        draw_truth_vs_reco_truth_matched(
            {sample_type: sample_hists},
            [
                "hh_mass_reco_truth_matched_min_mass_true_pairing",
                "hh_mass_reco_truth_matched_min_mass_true_pairing_correct",
            ],
            energy,
            luminosity=luminosity,
            xlabel="$m_{\mathrm{HH}}$ [GeV]",
            ylabel="Events",
            legend_labels={
                "hh_mass_reco_truth_matched_min_mass_true_pairing": legend_labels[
                    "min_mass_true_pairing"
                ],
                "hh_mass_reco_truth_matched_min_mass_true_pairing_correct": f"{legend_labels['min_mass_true_pairing']} and parent ID 25",
            },
            legend_options={"loc": "center right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{legend_labels['truth_matching']}",
            xmin=100,
            draw_ratio=True,
            output_dir=output_dir,
        )

        draw_truth_vs_reco_truth_matched(
            {sample_type: sample_hists},
            [
                "hh_mass_reco_truth_matched_min_mass_pairing",
                "hh_mass_reco_truth_matched_min_mass_pairing_correct",
            ],
            energy,
            luminosity=luminosity,
            xlabel="$m_{\mathrm{HH}}$ [GeV]",
            ylabel="Events",
            legend_labels={
                "hh_mass_reco_truth_matched_min_mass_pairing": legend_labels[
                    "min_mass_pairing"
                ],
                "hh_mass_reco_truth_matched_min_mass_pairing_correct": f"{legend_labels['min_mass_pairing']} and parent ID 25",
            },
            legend_options={"loc": "center right", "fontsize": "small"},
            third_exp_label=f"\n{sample_labels[sample_type]}\n{legend_labels['truth_matching']}",
            xmin=100,
            draw_ratio=True,
            output_dir=output_dir,
        )

        draw_mH_plane_2D_hists(
            sample_hists,
            sample_type,
            "mHH_plane_reco_min_deltar_pairing",
            energy,
            luminosity=luminosity,
            output_dir=output_dir,
        )
        draw_mH_plane_2D_hists(
            sample_hists,
            sample_type,
            "mHH_plane_reco_max_deltar_pairing",
            energy,
            luminosity=luminosity,
            output_dir=output_dir,
        )
        draw_mH_plane_2D_hists(
            sample_hists,
            sample_type,
            "mHH_plane_reco_min_mass_true_pairing",
            energy,
            luminosity=luminosity,
            output_dir=output_dir,
        )
        draw_mH_plane_2D_hists(
            sample_hists,
            sample_type,
            "mHH_plane_reco_min_mass_pairing",
            energy,
            luminosity=luminosity,
            output_dir=output_dir,
        )

    #### Old plots ####

    # draw_1d_hists(
    #     hists_group,
    #     f"leading_jet_{1}_pt",
    #     energy,
    #     xlabel="Leading jet $p_{\mathrm{T}}$ [GeV]",
    #     ylabel="Events",
    #     luminosity=luminosity,
    #     yscale="log",
    #     output_dir=output_dir,
    # )
    # draw_1d_hists(
    #     hists_group,
    #     "hh_deltaeta",
    #     energy,
    #     xlabel="$\Delta\eta_{HH}$",
    #     ylabel="Events Normalized",
    #     luminosity=luminosity,
    #     density=True,
    #     xcut=1.5,
    #     third_exp_label="\n" + btag + " events, no $\mathrm{X}_{\mathrm{Wt}}$ cut",
    #     output_dir=output_dir,
    # )
    # draw_1d_hists(
    #     hists_group,
    #     "top_veto",
    #     energy,
    #     xlabel="$\mathrm{X}_{\mathrm{Wt}}$",
    #     ylabel="Events Normalized",
    #     third_exp_label=f"\n{btag} events",
    #     density=True,
    #     luminosity=luminosity,
    #     xcut=1.5,
    #     output_dir=output_dir,
    # )
    # draw_1d_hists(
    #     hists_group,
    #     "hh_mass_discrim",
    #     energy,
    #     xlabel="$\mathrm{X}_{\mathrm{HH}}$",
    #     ylabel="Events Normalized",
    #     density=True,
    #     luminosity=luminosity,
    #     third_exp_label=f"\n{btag} events",
    #     xcut=1.6,
    #     output_dir=output_dir,
    # )
    # draw_mH_1D_hists_v2(
    #     hists_group,
    #     "h[12]_mass_baseline$",
    #     energy,
    #     luminosity=luminosity,
    #     yscale="log",
    #     ylabel="Events",
    #     xlims=[60, 200],
    #     ylims=[1e-1, 1e8],
    #     third_exp_label=f"\n{btag} SR/CR Inclusive",
    #     output_dir=output_dir,
    # )
    # draw_mH_1D_hists_v2(
    #     hists_group,
    #     "h[12]_mass_baseline_signal_region$",
    #     energy,
    #     luminosity=luminosity,
    #     yscale="log",
    #     ylabel="Events",
    #     xlims=[60, 200],
    #     ylims=[1e-1, 1e5],
    #     third_exp_label=f"\n{btag} Signal Region",
    #     output_dir=output_dir,
    # )
    # draw_mH_1D_hists_v2(
    #     hists_group,
    #     "h[12]_mass_baseline_control_region$",
    #     energy,
    #     luminosity=luminosity,
    #     yscale="log",
    #     ylabel="Events",
    #     xlims=[60, 200],
    #     ylims=[1e-1, 1e7],
    #     third_exp_label=f"\n{btag} Control Region",
    #     output_dir=output_dir,
    # )

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

    # for ith_h, label in enumerate(["leading", "subleading"]):
    #     draw_1d_hists(
    #         {key: value for key, value in hists_group.items() if "data" not in key},
    #         f"h{ith_h + 1}_truth_jet_baseline_signal_region",
    #         energy,
    #         luminosity=luminosity,
    #         xlabel="$\mathrm{H}_{\mathrm{" + label + "}}$ jet truth ID",
    #         third_exp_label=f"\n{btag} Signal Region \n 0: light, 4: charm, 5: bottom",
    #         density=True,
    #         draw_errors=True,
    #         output_dir=output_dir,
    #     )

    # for sample_type, sample_hists in hists_group.items():
    #     draw_kin_hists(
    #         sample_hists,
    #         sample_type,
    #         energy,
    #         object="jet",
    #         luminosity=luminosity,
    #         yscale="log",
    #         output_dir=output_dir,
    #     )
    #     draw_kin_hists(
    #         sample_hists,
    #         sample_type,
    #         energy,
    #         object="H1",
    #         luminosity=luminosity,
    #         yscale="log",
    #         output_dir=output_dir,
    #     )
    #     draw_kin_hists(
    #         sample_hists,
    #         sample_type,
    #         energy,
    #         object="H2",
    #         luminosity=luminosity,
    #         yscale="log",
    #         output_dir=output_dir,
    #     )
    #     draw_kin_hists(
    #         sample_hists,
    #         sample_type,
    #         energy,
    #         object="HH",
    #         luminosity=luminosity,
    #         yscale="log",
    #         output_dir=output_dir,
    #     )
    #     draw_mH_plane_2D_hists(
    #         sample_hists,
    #         sample_type,
    #         "mH_plane_baseline$",
    #         energy,
    #         luminosity=luminosity,
    #         output_dir=output_dir,
    #     )
    #     draw_mH_plane_2D_hists(
    #         sample_hists,
    #         sample_type,
    #         "mH_plane_baseline_control_region$",
    #         energy,
    #         luminosity=luminosity,
    #         output_dir=output_dir,
    #     )
    #     draw_mH_plane_2D_hists(
    #         sample_hists,
    #         sample_type,
    #         "mH_plane_baseline_signal_region$",
    #         energy,
    #         luminosity=luminosity,
    #         output_dir=output_dir,
    #     )
    #     draw_1d_hists(
    #         {sample_type: sample_hists},
    #         "mc_event_weight",
    #         energy,
    #         luminosity=luminosity,
    #         yscale="log",
    #         xlabel="mc_event_weight",
    #         output_dir=output_dir,
    #     )
    #     draw_1d_hists(
    #         {sample_type: sample_hists},
    #         "total_event_weight",
    #         energy,
    #         luminosity=luminosity,
    #         yscale="log",
    #         xlabel="total_event_weight",
    #         output_dir=output_dir,
    #     )
