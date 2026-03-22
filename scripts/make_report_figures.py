from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from cerium_hydride import (
    ONE_ATM,
    CeriumHydrideModel,
    H2ArTransport,
    SarussiSolidCeriumKinetics,
    plot_limitation_map,
    plot_limitation_maps_grid,
    plot_regional_radial_profiles,
    save_figure,
)

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        # Match the LaTeX document typography so the figures feel native to the
        # report rather than imported from a separate plotting style.
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.prop_cycle": matplotlib.cycler(color=["#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]),
    }
)


OUTPUT_DIR = Path("report/figures")

# ---------------------------------------------------------------------------
# Base model settings used across the report figures
# ---------------------------------------------------------------------------
PURE_H2_CUTOFF = 0.999
SHELL_THICKNESS_M = 100e-9
D_SHELL_SOLID_M2_S = 4.0e-8
T_SHELL_MELT_K = 1690.0
D_SHELL_MOLTEN_MULTIPLIER = 10.0
RHO_CE_KG_M3 = 6680.0

# ---------------------------------------------------------------------------
# Sweep settings for quasi-steady bottleneck maps
# ---------------------------------------------------------------------------
P_TOTAL_ATM = 1.0
P_TOTAL_PA = P_TOTAL_ATM * ONE_ATM
# Use a denser sweep for the final figures so category boundaries are resolved
# more cleanly and the maps do not show obvious stair-stepping.
T_VALUES_K = np.linspace(1100.0, 1800.0, 151)
Y_VALUES = np.linspace(1.0e-4, 1.0, 301)
PARTICLE_DIAMETERS_UM = [10.0, 50.0, 100.0]
PRESSURE_MAPS_ATM = [0.25, 1.0, 5.0]

# ---------------------------------------------------------------------------
# Detailed radial-profile comparison settings
# ---------------------------------------------------------------------------
PROFILE_DIAMETER_UM = 50.0
PROFILE_T_K = 1450.0
PROFILE_Y_VALUES = [0.002, 0.02, 0.2]


def make_model(initial_diameter_um: float) -> CeriumHydrideModel:
    """Construct one baseline quasi-steady model instance."""
    r0_m = 0.5 * initial_diameter_um * 1e-6
    return CeriumHydrideModel(
        r0_m=r0_m,
        R_far_m=1000.0 * r0_m,
        transport=H2ArTransport("gri30.yaml"),
        kinetics=SarussiSolidCeriumKinetics(rho_ce_kg_m3=RHO_CE_KG_M3),
        geometry_mode="constant_shell",
        delta_shell_m=SHELL_THICKNESS_M,
        rho_ce_kg_m3=RHO_CE_KG_M3,
        D_shell_solid_m2_s=D_SHELL_SOLID_M2_S,
        T_shell_melt_K=T_SHELL_MELT_K,
        D_shell_molten_multiplier=D_SHELL_MOLTEN_MULTIPLIER,
        pure_h2_cutoff=PURE_H2_CUTOFF,
    )


def save_limitation_map_single() -> None:
    model = make_model(PROFILE_DIAMETER_UM)
    limitation_map = model.build_limitation_map(
        T_values_K=T_VALUES_K,
        y_values=Y_VALUES,
        P_total_Pa=P_TOTAL_PA,
    )
    fig, _ = plot_limitation_map(
        limitation_map,
        title=None,
        show_colorbar=True,
        show_cutoff=True,
    )
    save_figure(fig, OUTPUT_DIR / "limitation_map_single.png")


def save_limitation_maps_by_size() -> None:
    limitation_maps = []
    panel_titles = []

    for diameter_um in PARTICLE_DIAMETERS_UM:
        model = make_model(diameter_um)
        limitation_maps.append(
            model.build_limitation_map(
                T_values_K=T_VALUES_K,
                y_values=Y_VALUES,
                P_total_Pa=P_TOTAL_PA,
            )
        )
        panel_titles.append(rf"$d = {diameter_um:.0f}\,\mu\mathrm{{m}}$")

    fig, _ = plot_limitation_maps_grid(
        limitation_maps,
        panel_titles,
        suptitle=None,
    )
    save_figure(fig, OUTPUT_DIR / "limitation_maps_by_size.png")


def save_limitation_maps_by_pressure() -> None:
    limitation_maps = []
    panel_titles = []
    model = make_model(PROFILE_DIAMETER_UM)

    for pressure_atm in PRESSURE_MAPS_ATM:
        limitation_maps.append(
            model.build_limitation_map(
                T_values_K=T_VALUES_K,
                y_values=Y_VALUES,
                P_total_Pa=pressure_atm * ONE_ATM,
            )
        )
        panel_titles.append(rf"$P = {pressure_atm:.2f}\,\mathrm{{atm}}$")

    fig, _ = plot_limitation_maps_grid(
        limitation_maps,
        panel_titles,
        suptitle=None,
    )
    save_figure(fig, OUTPUT_DIR / "limitation_maps_by_pressure.png")


def save_regional_radial_profiles() -> None:
    """
    Compare several far-field hydrogen conditions at one temperature and pressure.

    This figure is intentionally quasi-steady: each profile is an independent
    steady operating point rather than a snapshot from a time-dependent run.
    """
    model = make_model(PROFILE_DIAMETER_UM)
    profiles = []
    labels = []

    for y_inf in PROFILE_Y_VALUES:
        profiles.append(
            model.steady_state_radial_profile(
                a_m=model.r0_m,
                T_K=PROFILE_T_K,
                P_Pa=P_TOTAL_PA,
                y_inf=y_inf,
                n_shell_points=120,
                n_gas_points=320,
            )
        )
        labels.append(f"{y_inf:.3f}")

    fig, _ = plot_regional_radial_profiles(
        profiles,
        labels,
        title=None,
    )
    save_figure(fig, OUTPUT_DIR / "radial_profiles_by_yinf.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_limitation_map_single()
    save_limitation_maps_by_size()
    save_limitation_maps_by_pressure()
    save_regional_radial_profiles()
    print(f"Saved figures to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
