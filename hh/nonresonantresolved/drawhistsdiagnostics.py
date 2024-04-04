from argparse import Namespace
from hh.shared.utils import logger
from hh.shared.drawhists import (
    draw_1d_hists,
    draw_kin_hists,
    draw_mH_plane_2D_hists,
    draw_mH_1D_hists_v2,
)


def draw_hists(
    hists_group: dict,
    args: Namespace,
) -> None:
    """Draw all the histrograms"""

    btag = args.btag
    luminosity = args.luminosity
    energy = args.energy
    output_dir = args.output_dir

    # draw_1d_hists(
    #     {
    #         key: value
    #         for key, value in hists_group.items()
    #         if "multijet" in key or "QCD" in key
    #     },
    #     f"leading_jet_{1}_pt",
    #     energy,
    #     xlabel="Leading jet $p_{\mathrm{T}}$ [GeV]",
    #     ylabel="Events",
    #     luminosity=luminosity,
    #     yscale="log",
    #     output_dir=output_dir,
    # )
    # draw_1d_hists(
    #     {
    #         key: value
    #         for key, value in hists_group.items()
    #         if "ggF" in key or "data" in key
    #     },
    #     "hh_deltaeta",
    #     energy,
    #     xlabel="$\Delta\eta_{HH}$",
    #     ylabel="Events",
    #     luminosity=luminosity,
    #     ynorm_binwidth=True,
    #     xcut=1.5,
    #     third_exp_label="\n" + btag + ", no $\mathrm{X}_{\mathrm{Wt}}$ cut",
    #     ggFk01_factor=500,
    #     ggFk10_factor=50,
    #     output_dir=output_dir,
    # )
    # draw_1d_hists(
    #     {
    #         key: value
    #         for key, value in hists_group.items()
    #         if "ggF_k01" in key or "ttbar" in key
    #     },
    #     "top_veto",
    #     energy,
    #     xlabel="$\mathrm{X}_{\mathrm{Wt}}$",
    #     ylabel="Events",
    #     third_exp_label=f"\n{btag}",
    #     ynorm_binwidth=True,
    #     luminosity=luminosity,
    #     ggFk01_factor=250,
    #     xcut=1.5,
    #     output_dir=output_dir,
    # )
    # draw_1d_hists(
    #     {
    #         key: value
    #         for key, value in hists_group.items()
    #         if "ggF" in key or "multijet" in key or "ttbar" in key
    #     },
    #     "hh_mass_discrim",
    #     energy,
    #     xlabel="$\mathrm{X}_{\mathrm{HH}}$",
    #     ylabel="Events",
    #     ynorm_binwidth=True,
    #     luminosity=luminosity,
    #     third_exp_label=f"\n{btag}",
    #     ggFk01_factor=5000,
    #     ggFk10_factor=500,
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

    # draw_1d_hists(
    #     {key: value for key, value in hists_group.items() if "data" not in key},
    #     "mc_event_weight",
    #     energy,
    #     luminosity=luminosity,
    #     yscale="log",
    #     xlabel="mc_event_weight",
    #     output_dir=output_dir,
    # )

    # draw_1d_hists(
    #     {key: value for key, value in hists_group.items() if "data" not in key},
    #     "mc_event_weight_baseline_signal_region",
    #     energy,
    #     luminosity=luminosity,
    #     yscale="log",
    #     xlabel="mc_event_weight_baseline_signal_region",
    #     output_dir=output_dir,
    # )

    # draw_1d_hists(
    #     {key: value for key, value in hists_group.items() if "data" not in key},
    #     "total_event_weight",
    #     energy,
    #     luminosity=luminosity,
    #     yscale="log",
    #     xlabel="total_event_weight",
    #     output_dir=output_dir,
    # )

    # draw_1d_hists(
    #     {key: value for key, value in hists_group.items() if "data" not in key},
    #     "total_event_weight_baseline_signal_region",
    #     energy,
    #     luminosity=luminosity,
    #     yscale="log",
    #     xlabel="total_event_weight_baseline_signal_region",
    #     output_dir=output_dir,
    # )

    # for ith_h in [1, 2]:
    #     draw_1d_hists(
    #         {key: value for key, value in hists_group.items() if "data" not in key},
    #         f"h{ith_h}_truth_jet_baseline_signal_region",
    #         energy,
    #         luminosity=luminosity,
    #         xlabel=f"H{ith_h} jet truth ID",
    #         third_exp_label=f"\n{btag} Signal Region \n 0: light, 4: c, 5: b",
    #         density=True,
    #         output_dir=output_dir,
    #     )

    for sample_type, sample_hists in hists_group.items():
        is_data = "data" in sample_type
        draw_kin_hists(
            sample_hists,
            sample_type,
            energy,
            object="jet",
            luminosity=luminosity,
            yscale="log",
            output_dir=output_dir,
        )
        draw_1d_hists(
            {sample_type: sample_hists},
            f"leading_jet_{1}_pt",
            energy,
            xlabel="Leading jet $p_{\mathrm{T}}$ [GeV]",
            ylabel="Events",
            luminosity=luminosity,
            yscale="log",
            plot_name=f"leading_jet_{1}_pt_{sample_type}",
            output_dir=output_dir,
        )
        draw_1d_hists(
            {sample_type: sample_hists},
            "mc_event_weight",
            energy,
            luminosity=luminosity,
            yscale="log",
            xlabel="mc_event_weight",
            plot_name=f"mc_event_weight_{sample_type}",
            output_dir=output_dir,
        )
        draw_1d_hists(
            {sample_type: sample_hists},
            "total_event_weight",
            energy,
            luminosity=luminosity,
            yscale="log",
            xlabel="total_event_weight",
            plot_name=f"total_event_weight_{sample_type}",
            output_dir=output_dir,
        )
        # draw_kin_hists(
        #     sample_hists,
        #     sample_type,
        #     energy,
        #     object="jet",
        #     region="_baseline_signal_region",
        #     luminosity=luminosity,
        #     yscale="log",
        #     output_dir=output_dir,
        # )
        # draw_kin_hists(
        #     sample_hists,
        #     sample_type,
        #     energy,
        #     object="jet",
        #     region="_baseline_control_region",
        #     luminosity=luminosity,
        #     yscale="log",
        #     output_dir=output_dir,
        # )

        # draw_kin_hists(
        #     sample_hists,
        #     sample_type,
        #     energy,
        #     object="h1",
        #     luminosity=luminosity,
        #     yscale="log",
        #     output_dir=output_dir,
        # )
        # draw_kin_hists(
        #     sample_hists,
        #     sample_type,
        #     energy,
        #     object="h2",
        #     luminosity=luminosity,
        #     yscale="log",
        #     output_dir=output_dir,
        # )
        # draw_kin_hists(
        #     sample_hists,
        #     sample_type,
        #     energy,
        #     object="hh",
        #     luminosity=luminosity,
        #     yscale="log",
        #     output_dir=output_dir,
        # )

        # draw_mH_plane_2D_hists(
        #     sample_hists,
        #     sample_type,
        #     "mH_plane_baseline$",
        #     energy,
        #     luminosity=luminosity,
        #     output_dir=output_dir,
        # )
        # draw_mH_plane_2D_hists(
        #     sample_hists,
        #     sample_type,
        #     "mH_plane_baseline_control_region$",
        #     energy,
        #     luminosity=luminosity,
        #     output_dir=output_dir,
        # )
        # if not is_data:
        #     draw_mH_plane_2D_hists(
        #         sample_hists,
        #         sample_type,
        #         "mH_plane_baseline_signal_region$",
        #         energy,
        #         luminosity=luminosity,
        #         output_dir=output_dir,
        #     )
