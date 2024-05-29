from argparse import Namespace
from hh.shared.utils import logger
from hh.shared.drawhists import (
    draw_1d_hists,
    draw_kin_hists,
    draw_mH_plane_2D_hists,
    draw_mH_1D_hists_v2,
    num_events_vs_sample,
    draw_truth_vs_reco_truth_matched,
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
        "mc23d_ggF_k05": "kl=5 ggF MC23d",
        "mc23a_ggF_k01": "kl=1 ggF MC23a",
        "mc23a_ggF_k05": "kl=5 ggF MC23a",
        "mc23d_ggF_k01": "kl=1 ggF MC23d",
    }

    draw_1d_hists(
        hists_group,
        f"hh_mass_truth",
        energy,
        xlabel="Truth HH mass [GeV]",
        ylabel="Events",
        legend_labels=sample_labels,
        luminosity=luminosity,
        xmin=0,
        ggFk01_factor=10,
        draw_errors=True,
        output_dir=output_dir,
    )

    draw_1d_hists(
        hists_group,
        f"hh_mass_reco_truth_matched",
        energy,
        xlabel="Reco Truth Matched HH mass [GeV]",
        ylabel="Events",
        legend_labels=sample_labels,
        third_exp_label="\nReco Truth-matched $\Delta R < 0.3$",
        luminosity=luminosity,
        xmin=0,
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
                "hh_mass_reco_truth_matched_v2",
            ],
            energy,
            luminosity=luminosity,
            xlabel="$m_{\mathrm{HH}}$ [GeV]",
            ylabel="Events",
            legend_labels={
                "hh_mass_truth_reco_matched": f"Truth reco-matched",
                "hh_mass_reco_truth_matched": f"Reco truth-matched",
                "hh_mass_reco_truth_matched_v2": r"$\geq 4 \mathrm{HadronConeExclTruthLabelID} = 5$",
            },
            legend_options={"loc": "upper right", "fontsize": "x-small"},
            third_exp_label=f"\n{sample_labels[sample_type]}"
            + "\nReco Truth-matched $\Delta R < 0.3$",
            xmin=0,
            draw_errors=True,
            output_dir=output_dir,
        )
        draw_1d_hists(
            {sample_type: sample_hists},
            "hh_mass_reco_vs_truth_response",
            energy,
            luminosity=luminosity,
            # yscale="log",
            xlabel="HH mass response [%]",
            legend_labels={sample_type: sample_labels[sample_type]},
            third_exp_label="\nReco Truth-matched $\Delta R < 0.3$",
            output_dir=output_dir,
        )
        for i in [1, 2, 3, 4]:
            categories = [
                "_truth_matched_2b2j_asym",
                # "_truth_matched_2b2j_sym",
                # "_truth_matched_2b1j",
            ]
            for cat in categories:
                draw_truth_vs_reco_truth_matched(
                    {sample_type: sample_hists},
                    [
                        f"hh_jet_{i}_pt_truth_matched",
                        f"hh_jet_{i}_pt{cat}",
                        f"hh_jet_{i}_pt{cat}_n_btags",
                        f"hh_jet_{i}_pt{cat}_4_btags",
                        f"hh_jet_{i}_pt_truth_matched_4_btags",
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
                        f"hh_jet_{i}_pt_truth_matched_4_btags": r"$\geq$ 4 jets passing GN2v01@77%",
                    },
                    legend_options={"loc": "center right", "fontsize": "small"},
                    third_exp_label=f"\n{sample_labels[sample_type]}"
                    + "\nReco Truth-matched $\Delta R < 0.3$",
                    # scale_factors=[1, 10],
                    # draw_errors=True,
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
