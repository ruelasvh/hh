from argparse import Namespace
from hh.shared.utils import logger
from hh.shared.drawhists import (
    draw_1d_hists,
    draw_kin_hists,
    draw_mH_plane_2D_hists,
    draw_mH_1D_hists_v2,
    num_events_vs_sample,
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

    draw_1d_hists(
        {key: value for key, value in hists_group.items() if "multijet" in key},
        f"leading_jet_{1}_pt",
        energy,
        xlabel="Leading jet $p_{\mathrm{T}}$ [GeV]",
        ylabel="Events",
        luminosity=luminosity,
        yscale="log",
        output_dir=output_dir,
    )
    draw_1d_hists(
        hists_group,
        "hh_deltaeta",
        energy,
        xlabel="$\Delta\eta_{HH}$",
        ylabel="Events Normalized",
        luminosity=luminosity,
        density=True,
        xcut=1.5,
        third_exp_label="\n" + btag + " events, no $\mathrm{X}_{\mathrm{Wt}}$ cut",
        output_dir=output_dir,
    )
    draw_1d_hists(
        hists_group,
        "top_veto",
        energy,
        xlabel="$\mathrm{X}_{\mathrm{Wt}}$",
        ylabel="Events Normalized",
        third_exp_label=f"\n{btag} events",
        density=True,
        luminosity=luminosity,
        xcut=1.5,
        output_dir=output_dir,
    )
    draw_1d_hists(
        hists_group,
        "hh_mass_discrim",
        energy,
        xlabel="$\mathrm{X}_{\mathrm{HH}}$",
        ylabel="Events Normalized",
        density=True,
        luminosity=luminosity,
        third_exp_label=f"\n{btag} events",
        xcut=1.6,
        output_dir=output_dir,
    )
    draw_mH_1D_hists_v2(
        hists_group,
        "h[12]_mass_baseline$",
        energy,
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
        "h[12]_mass_baseline_signal_region$",
        energy,
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
        "h[12]_mass_baseline_control_region$",
        energy,
        luminosity=luminosity,
        yscale="log",
        ylabel="Events",
        xlims=[60, 200],
        ylims=[1e-1, 1e7],
        third_exp_label=f"\n{btag} Control Region",
        output_dir=output_dir,
    )

    if args.split_jz:
        for region in ["signal", "control"]:
            num_events_vs_sample(
                {
                    key: value
                    for key, value in sorted(hists_group.items())
                    if "multijet" in key
                },
                f"mH_plane_baseline_{region}_region$",
                energy,
                ylabel=f"{region.capitalize()} Events",
                xlabel="Multijet Samples",
                luminosity=luminosity,
                density=True,
                third_exp_label=f"Signal Region",
                output_dir=output_dir,
            )

    for ith_h, label in enumerate(["leading", "subleading"]):
        draw_1d_hists(
            {key: value for key, value in hists_group.items() if "data" not in key},
            f"h{ith_h + 1}_truth_jet_baseline_signal_region",
            energy,
            luminosity=luminosity,
            xlabel="$\mathrm{H}_{\mathrm{" + label + "}}$ jet truth ID",
            third_exp_label=f"\n{btag} Signal Region \n 0: light, 4: charm, 5: bottom",
            density=True,
            output_dir=output_dir,
        )

    for sample_type, sample_hists in hists_group.items():
        draw_kin_hists(
            sample_hists,
            sample_type,
            energy,
            object="jet",
            luminosity=luminosity,
            yscale="log",
            output_dir=output_dir,
        )
        draw_kin_hists(
            sample_hists,
            sample_type,
            energy,
            object="h1",
            luminosity=luminosity,
            yscale="log",
            output_dir=output_dir,
        )
        draw_kin_hists(
            sample_hists,
            sample_type,
            energy,
            object="h2",
            luminosity=luminosity,
            yscale="log",
            output_dir=output_dir,
        )
        draw_kin_hists(
            sample_hists,
            sample_type,
            energy,
            object="hh",
            luminosity=luminosity,
            yscale="log",
            output_dir=output_dir,
        )
        draw_mH_plane_2D_hists(
            sample_hists,
            sample_type,
            "mH_plane_baseline$",
            energy,
            luminosity=luminosity,
            output_dir=output_dir,
        )
        draw_mH_plane_2D_hists(
            sample_hists,
            sample_type,
            "mH_plane_baseline_control_region$",
            energy,
            luminosity=luminosity,
            output_dir=output_dir,
        )
        draw_mH_plane_2D_hists(
            sample_hists,
            sample_type,
            "mH_plane_baseline_signal_region$",
            energy,
            luminosity=luminosity,
            output_dir=output_dir,
        )
        draw_1d_hists(
            {sample_type: sample_hists},
            "mc_event_weight",
            energy,
            luminosity=luminosity,
            yscale="log",
            xlabel="mc_event_weight",
            output_dir=output_dir,
        )
        draw_1d_hists(
            {sample_type: sample_hists},
            "total_event_weight",
            energy,
            luminosity=luminosity,
            yscale="log",
            xlabel="total_event_weight",
            output_dir=output_dir,
        )
