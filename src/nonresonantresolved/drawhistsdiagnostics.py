from argparse import Namespace
from src.shared.utils import logger
from src.shared.drawhists import (
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
    output_dir = args.output_dir

    draw_1d_hists(
        hists_group={
            "ggF (signal)": hists_group["mc21_ggF_k01"],
            "QCD": hists_group["mc21_multijet"],
            "ttbar": hists_group["mc21_ttbar"],
        },
        hist_prefix="jet_pt",
        xlabel="jet $p_{\mathrm{T}}$ [GeV]",
        ylabel="Events",
        luminosity=luminosity,
        yscale="log",
        output_dir=output_dir,
    )

    draw_1d_hists(
        hists_group={
            key: value
            for key, value in hists_group.items()
            if "ggF" in key or "data" in key
        },
        hist_prefix="hh_deltaeta",
        xlabel="$\Delta\eta_{HH}$",
        ylabel="Events",
        luminosity=luminosity,
        ynorm_binwidth=True,
        xcut=1.5,
        third_exp_label="\n" + btag + ", no $\mathrm{X}_{\mathrm{Wt}}$ cut",
        ggFk01_factor=500,
        ggFk10_factor=50,
        output_dir=output_dir,
    )
    draw_1d_hists(
        hists_group={
            key: value
            for key, value in hists_group.items()
            if "ggF_k01" in key or "ttbar" in key
        },
        hist_prefix="top_veto",
        xlabel="$\mathrm{X}_{\mathrm{Wt}}$",
        ylabel="Events",
        third_exp_label=f"\n{btag}",
        ynorm_binwidth=True,
        luminosity=luminosity,
        ggFk01_factor=250,
        xcut=1.5,
        output_dir=output_dir,
    )
    draw_1d_hists(
        hists_group={
            key: value
            for key, value in hists_group.items()
            if "ggF" in key or "multijet" in key or "ttbar" in key
        },
        hist_prefix="hh_mass_discrim",
        xlabel="$\mathrm{X}_{\mathrm{HH}}$",
        ylabel="Events",
        ynorm_binwidth=True,
        luminosity=luminosity,
        third_exp_label=f"\n{btag}",
        ggFk01_factor=5000,
        ggFk10_factor=500,
        xcut=1.6,
        output_dir=output_dir,
    )
    draw_mH_1D_hists_v2(
        hists_group,
        hist_prefix="h[12]_mass_baseline$",
        luminosity=luminosity,
        yscale="log",
        ylabel="Events",
        xlims=[60, 200],
        ylims=[1e-1, 1e8],
        third_exp_label=f"\n{btag} SR/CR Inclusive",
        output_dir=output_dir,
    )
    draw_mH_1D_hists_v2(
        hists_group,
        hist_prefix="h[12]_mass_baseline_signal_region$",
        region="SR",
        luminosity=luminosity,
        yscale="log",
        ylabel="Events",
        xlims=[60, 200],
        ylims=[1e-1, 1e5],
        third_exp_label=f"\n{btag} Signal Region",
        output_dir=output_dir,
    )
    draw_mH_1D_hists_v2(
        hists_group,
        hist_prefix="h[12]_mass_baseline_control_region$",
        region="CR",
        luminosity=luminosity,
        yscale="log",
        ylabel="Events",
        xlims=[60, 200],
        ylims=[1e-1, 1e7],
        third_exp_label=f"\n{btag} Control Region",
        output_dir=output_dir,
    )

    for sample_type, sample_hists in hists_group.items():
        is_data = "data" in sample_type
        draw_kin_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            object="jet",
            luminosity=luminosity,
            yscale="log",
            output_dir=output_dir,
        )
        draw_kin_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            object="h1",
            luminosity=luminosity,
            yscale="log",
            output_dir=output_dir,
        )
        draw_kin_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            object="h2",
            luminosity=luminosity,
            yscale="log",
            output_dir=output_dir,
        )
        draw_kin_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            object="hh",
            luminosity=luminosity,
            yscale="log",
            output_dir=output_dir,
        )
        draw_mH_plane_2D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            hist_prefix="mH_plane_baseline$",
            luminosity=luminosity,
            output_dir=output_dir,
        )
        draw_mH_plane_2D_hists(
            sample_hists=sample_hists,
            sample_name=sample_type,
            region="CR",
            hist_prefix="mH_plane_baseline_control_region$",
            luminosity=luminosity,
            output_dir=output_dir,
        )
        if not is_data:
            draw_mH_plane_2D_hists(
                sample_hists=sample_hists,
                sample_name=sample_type,
                region="SR",
                hist_prefix="mH_plane_baseline_signal_region$",
                luminosity=luminosity,
                output_dir=output_dir,
            )
