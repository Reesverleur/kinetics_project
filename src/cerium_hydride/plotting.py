from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .constants import (
    LIMITATION_DISPLAY_LABELS,
    LIMITATION_GAS_DIFFUSION,
    LIMITATION_KINETICS,
    LIMITATION_SHELL_DIFFUSION,
)


def _ensure_parent(path: str | Path | None) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _limitation_colormap():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    # Okabe-Ito style colors keep the categories distinguishable for many
    # color-vision deficiencies while remaining visually distinct in print.
    cmap = ListedColormap(["#0072B2", "#E69F00", "#D55E00"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    return cmap, norm


def plot_limitation_map(
    limitation_map: dict,
    *,
    ax=None,
    title: str | None = None,
    show_colorbar: bool = True,
    show_cutoff: bool = True,
):
    """
    Plot the dominant bottleneck over temperature and far-field H2 fraction.

    The colors correspond to the process with the largest resistance share:
    gas diffusion, shell diffusion, or interface kinetics.
    """
    import matplotlib.pyplot as plt

    T_values_K = limitation_map["T_values_K"]
    y_values = limitation_map["y_values"]
    limiting_process_code = limitation_map["limiting_process_code"]

    cmap, norm = _limitation_colormap()

    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        created_figure = True
    else:
        fig = ax.figure

    mesh = ax.contourf(
        y_values,
        T_values_K,
        limiting_process_code,
        levels=[-0.5, 0.5, 1.5, 2.5],
        cmap=cmap,
        norm=norm,
        antialiased=False,
    )
    # Draw explicit category boundaries so the maps remain interpretable even if
    # color differences are harder to distinguish on screen or in print.
    ax.contour(
        y_values,
        T_values_K,
        limiting_process_code,
        levels=[0.5, 1.5],
        colors="k",
        linewidths=0.45,
    )
    ax.set_xlabel("Far-field H2 mole fraction")
    ax.set_ylabel("Temperature [K]")
    ax.set_xlim(float(np.min(y_values)), float(np.max(y_values)))
    if title is not None:
        ax.set_title(title)

    if show_colorbar:
        colorbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2])
        colorbar.ax.set_yticklabels(
            [
                LIMITATION_DISPLAY_LABELS[LIMITATION_GAS_DIFFUSION],
                LIMITATION_DISPLAY_LABELS[LIMITATION_SHELL_DIFFUSION],
                LIMITATION_DISPLAY_LABELS[LIMITATION_KINETICS],
            ]
        )

    cutoff = limitation_map.get("pure_h2_cutoff")
    if show_cutoff and cutoff is not None and cutoff < 1.0:
        ax.axvline(cutoff, color="k", linestyle="--", linewidth=0.8)

    if created_figure:
        fig.tight_layout()
    return fig, ax


def plot_limitation_maps_grid(
    limitation_maps: Sequence[dict],
    panel_titles: Sequence[str],
    *,
    ncols: int = 3,
    suptitle: str | None = None,
):
    """Plot several limitation maps on one figure with a shared colorbar."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    n_maps = len(limitation_maps)
    ncols = max(1, min(ncols, n_maps))
    nrows = int(np.ceil(n_maps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols + 0.9, 3.35 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for index, (limitation_map, panel_title) in enumerate(zip(limitation_maps, panel_titles)):
        plot_limitation_map(
            limitation_map,
            ax=axes_flat[index],
            title=panel_title,
            show_colorbar=False,
            show_cutoff=(index == 0),
        )

    for index in range(n_maps, len(axes_flat)):
        axes_flat[index].axis("off")

    cmap, norm = _limitation_colormap()
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.13, top=0.90 if suptitle else 0.95, wspace=0.28, hspace=0.25)
    colorbar_axis = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    colorbar = fig.colorbar(sm, cax=colorbar_axis, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(
        [
            LIMITATION_DISPLAY_LABELS[LIMITATION_GAS_DIFFUSION],
            LIMITATION_DISPLAY_LABELS[LIMITATION_SHELL_DIFFUSION],
            LIMITATION_DISPLAY_LABELS[LIMITATION_KINETICS],
        ]
    )

    if suptitle:
        fig.suptitle(suptitle)
    return fig, axes


def plot_regime_map(
    regime_map: dict,
    *,
    ax=None,
    title: str | None = None,
    show_colorbar: bool = True,
    show_cutoff: bool = True,
):
    """Compatibility wrapper around :func:`plot_limitation_map`."""
    return plot_limitation_map(
        regime_map,
        ax=ax,
        title=title,
        show_colorbar=show_colorbar,
        show_cutoff=show_cutoff,
    )


def plot_regime_maps_by_size(
    regime_maps: Sequence[dict],
    diameters_um: Sequence[float],
    *,
    ncols: int = 3,
    suptitle: str | None = None,
):
    """Compatibility wrapper used by older scripts."""
    titles = [f"d = {diameter_um:.1f} um" for diameter_um in diameters_um]
    return plot_limitation_maps_grid(regime_maps, titles, ncols=ncols, suptitle=suptitle)


def plot_radial_profile(
    profile: dict,
    *,
    ax=None,
    shell_as_effective_variable: bool = True,
    title: str | None = None,
    x_scale: str = "linear",
):
    """
    Plot the steady radial hydrogen profile.

    ``x_scale="log"`` is especially useful because the gas domain usually spans
    several orders of magnitude in radius while the shell occupies only a very
    narrow band near the particle surface.
    """
    import matplotlib.pyplot as plt

    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.7, 4.3))
        created_figure = True
    else:
        fig = ax.figure

    r_core_m = profile["r_core_m"]
    r_outer_m = profile["r_outer_m"]
    r_shell_m = profile["r_shell_m"]
    y_shell = profile["y_shell_effective"]
    r_gas_m = profile["r_gas_m"]
    y_gas = profile["y_gas"]
    y_inf = profile["y_inf"]
    state = profile["state"]

    if r_shell_m.size > 0:
        ax.axvspan(r_core_m * 1e6, r_outer_m * 1e6, color="0.92", zorder=0)
        if shell_as_effective_variable:
            ax.plot(
                r_shell_m * 1e6,
                y_shell,
                "--",
                linewidth=2.0,
                label="Shell-side effective H2 variable",
            )

    ax.plot(r_gas_m * 1e6, y_gas, linewidth=2.2, label="Gas-phase H2 mole fraction")
    ax.axhline(y_inf, color="0.4", linestyle=":", linewidth=1.0, label="Far-field H2 mole fraction")

    ax.set_xscale(x_scale)
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("Hydrogen mole fraction / effective shell variable")
    ax.set_ylim(bottom=0.0)
    if title is not None:
        ax.set_title(title)
    ax.legend(fontsize=8)

    if created_figure:
        fig.tight_layout()
    return fig, ax


def plot_diameters_vs_time(simulation: dict, *, ax=None, title: str | None = None):
    """Plot the Ce-core diameter and total particle diameter versus time."""
    import matplotlib.pyplot as plt

    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        created_figure = True
    else:
        fig = ax.figure

    t_ms = simulation["t_s"] * 1e3
    d_core_um = 2.0 * simulation["a_m"] * 1e6
    d_outer_um = 2.0 * simulation["b_m"] * 1e6

    ax.plot(t_ms, d_core_um, linewidth=2.2, label="Ce core diameter")
    ax.plot(t_ms, d_outer_um, linewidth=2.2, label="Outer particle diameter")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Diameter [um]")
    if title is not None:
        ax.set_title(title)
    ax.legend()

    if created_figure:
        fig.tight_layout()
    return fig, ax


def plot_characteristic_time_families(
    families: Sequence[dict],
    *,
    x_key: str = "p_h2_atm",
    x_label: str = "H2 partial pressure [atm]",
    target_keys: Sequence[str] = ("t_50_s", "t_90_s"),
    target_labels: Sequence[str] = ("t50", "t90"),
    title: str | None = None,
):
    """
    Plot characteristic hydriding times for several parameter families.

    Each family dictionary should contain:
    - ``label`` for the legend,
    - an x-array under ``x_key``,
    - one y-array for each key listed in ``target_keys``.
    """
    import matplotlib.pyplot as plt

    if len(target_keys) != len(target_labels):
        raise ValueError("target_keys and target_labels must have the same length.")

    fig, axes = plt.subplots(1, len(target_keys), figsize=(6.0 * len(target_keys), 4.1), squeeze=False)
    axes_flat = axes.ravel()

    for axis, target_key, target_label in zip(axes_flat, target_keys, target_labels):
        for family in families:
            axis.plot(
                family[x_key],
                family[target_key],
                marker="o",
                linewidth=2.0,
                markersize=4.0,
                label=family["label"],
            )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(x_label)
        axis.set_ylabel(f"{target_label} [s]")
        axis.set_title(target_label)
        axis.grid(True, which="both", alpha=0.25)

    axes_flat[0].legend()

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    return fig, axes


def plot_regional_radial_profiles(
    profiles: Sequence[dict],
    condition_labels: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    title: str | None = None,
):
    """
    Plot multiple steady profiles using separate x-scales for metal, shell, and gas.

    This figure is designed specifically for the report. It emphasizes the
    different spatial scales of the metal core, hydride shell, and external gas
    without collapsing the shell and near-surface structure into a tiny sliver.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if len(profiles) != len(condition_labels):
        raise ValueError("profiles and condition_labels must have the same length.")
    if len(profiles) == 0:
        raise ValueError("At least one profile is required.")

    if colors is None:
        colors = ["#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7"][: len(profiles)]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 3.95),
        sharey=True,
        gridspec_kw={"width_ratios": [1.2, 1.4, 3.0]},
    )

    metal_ax, shell_ax, gas_ax = axes
    metal_ax.set_facecolor("#f0f0f0")
    shell_ax.set_facecolor("#fff1e6")
    gas_ax.set_facecolor("#eef5ff")

    y_max = 0.0
    for profile in profiles:
        if profile["y_shell_effective"].size > 0:
            y_max = max(y_max, float(np.max(profile["y_shell_effective"])))
        if profile["y_gas"].size > 0:
            y_max = max(y_max, float(np.max(profile["y_gas"])))
    y_max = max(y_max, 1.0e-6)
    y_min = -0.03 * y_max

    # The metal region is identically zero in the present model, so one black
    # line is sufficient regardless of how many conditions are compared.
    core_radius_um = float(profiles[0]["r_core_m"]) * 1e6
    metal_ax.plot([0.0, core_radius_um], [0.0, 0.0], color="k", linewidth=2.0)

    legend_handles = []
    for profile, condition_label, color in zip(profiles, condition_labels, colors):
        r_shell_um = profile["r_shell_m"] * 1e6
        y_shell = profile["y_shell_effective"]
        r_gas_um = profile["r_gas_m"] * 1e6
        y_gas = profile["y_gas"]

        if r_shell_um.size > 0:
            shell_ax.plot(r_shell_um, y_shell, linestyle="--", color=color, linewidth=2.0)
        gas_ax.plot(r_gas_um, y_gas, linestyle="-", color=color, linewidth=2.2)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2.2, label=condition_label))

    metal_ax.set_xlim(0.0, core_radius_um)
    if profiles[0]["r_shell_m"].size > 0:
        shell_ax.set_xlim(float(profiles[0]["r_shell_m"][0]) * 1e6, float(profiles[0]["r_shell_m"][-1]) * 1e6)
    gas_ax.set_xlim(float(profiles[0]["r_gas_m"][0]) * 1e6, float(profiles[0]["r_gas_m"][-1]) * 1e6)
    gas_ax.set_xscale("log")

    for axis in axes:
        x_min, x_max = axis.get_xlim()
        axis.axvline(x_min, color="0.35", linestyle="--", linewidth=1.0)
        axis.axvline(x_max, color="0.35", linestyle="--", linewidth=1.0)
        axis.set_ylim(y_min, 1.02 * y_max)
        axis.grid(True, alpha=0.20)

    metal_ax.text(0.5, 0.92, "Metal\n($y_{H_2}=0$)", transform=metal_ax.transAxes, ha="center", va="top")
    shell_ax.text(0.5, 0.92, "Hydride shell", transform=shell_ax.transAxes, ha="center", va="top")
    gas_ax.text(0.5, 0.92, "Gas", transform=gas_ax.transAxes, ha="center", va="top")

    metal_ax.set_xlabel("Metal radius [um]")
    shell_ax.set_xlabel("Shell radius [um]")
    gas_ax.set_xlabel("Gas radius [um] (log)")
    metal_ax.set_ylabel("Hydrogen mole fraction / effective shell variable")
    gas_ax.legend(handles=legend_handles, title="Far-field $y_{H_2}$", loc="upper right", fontsize=9)

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig, axes


def save_figure(fig, output_path: str | Path, dpi: int = 450) -> None:
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
