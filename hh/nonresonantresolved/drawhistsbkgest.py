from argparse import Namespace
from hh.shared.utils import logger
from hh.shared.drawhists import (
    draw_1d_hists,
    draw_kin_hists,
    draw_mH_1D_hists,
    draw_mHH_plane_2D_hists,
    draw_mH_1D_hists_v2,
)


def draw_hists(
    hists_group: dict,
    args: Namespace,
) -> None:
    """Draw all the histrograms"""

    luminosity = args.luminosity
    output_dir = args.output_dir
    bkg_weight = args.bkg_weight

    # draw_1d_hists(
    #     hists_group={
    #         key: value
    #         for key, value in hists_group.items()
    #         if "ggF_k01" in key or "data" in key
    #     },
    #     hist_prefix="top_veto",
    #     xlabel="$\mathrm{X}_{\mathrm{Wt}}$",
    #     ylabel="Events",
    #     third_exp_label=f"\n Control Region",
    #     ynorm_binwidth=True,
    #     luminosity=luminosity,
    #     # density=True,
    #     postfix="before_reweighting",
    #     output_dir=output_dir,
    # )
    # draw_1d_hists(
    #     hists_group={
    #         key: value
    #         for key, value in hists_group.items()
    #         if "ggF_k01" in key or "data" in key
    #     },
    #     hist_prefix="top_veto",
    #     xlabel="$\mathrm{X}_{\mathrm{Wt}}$",
    #     ylabel="Events",
    #     third_exp_label=f"\n Control Region",
    #     ynorm_binwidth=True,
    #     # density=True,
    #     luminosity=luminosity,
    #     data2b_factor=bkg_weight,
    #     postfix="after_reweighting",
    #     output_dir=output_dir,
    # )

    draw_1d_hists(
        hists_group={
            key: value
            for key, value in hists_group.items()
            if "ggF_k01" in key or "data" in key
        },
        hist_prefix="h1_mass_baseline_control_region",
        xlabel="$\mathrm{m}_{\mathrm{H1}}$ [GeV]",
        ylabel="Events",
        third_exp_label=f"\n Control Region",
        ynorm_binwidth=True,
        # density=True,
        luminosity=luminosity,
        postfix="before_reweighting",
        output_dir=output_dir,
    )
    draw_1d_hists(
        hists_group={
            key: value
            for key, value in hists_group.items()
            if "ggF_k01" in key or "data" in key
        },
        hist_prefix="h1_mass_baseline_control_region",
        xlabel="$\mathrm{m}_{\mathrm{H1}}$ [GeV]",
        ylabel="Events",
        third_exp_label=f"\n Control Region",
        ynorm_binwidth=True,
        luminosity=luminosity,
        data2b_factor=bkg_weight,
        postfix="after_reweighting",
        output_dir=output_dir,
    )
    draw_1d_hists(
        hists_group={
            key: value
            for key, value in hists_group.items()
            if "ggF_k01" in key or "data" in key
        },
        hist_prefix="h2_mass_baseline_control_region",
        xlabel="$\mathrm{m}_{\mathrm{H2}}$ [GeV]",
        ylabel="Events",
        third_exp_label=f"\n Control Region",
        ynorm_binwidth=True,
        # density=True,
        luminosity=luminosity,
        postfix="before_reweighting",
        output_dir=output_dir,
    )
    draw_1d_hists(
        hists_group={
            key: value
            for key, value in hists_group.items()
            if "ggF_k01" in key or "data" in key
        },
        hist_prefix="h2_mass_baseline_control_region",
        xlabel="$\mathrm{m}_{\mathrm{H2}}$ [GeV]",
        ylabel="Events",
        third_exp_label=f"\n Control Region",
        ynorm_binwidth=True,
        luminosity=luminosity,
        data2b_factor=bkg_weight,
        postfix="after_reweighting",
        output_dir=output_dir,
    )
