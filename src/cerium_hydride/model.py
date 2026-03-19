from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from .constants import (
    LIMITATION_DISPLAY_LABELS,
    LIMITATION_GAS_DIFFUSION,
    LIMITATION_KINETICS,
    LIMITATION_NAME_BY_CODE,
    LIMITATION_SHELL_DIFFUSION,
    M_CE,
    M_CEH2,
    ONE_ATM,
    R_UNIVERSAL,
)
from .kinetics import SarussiSolidCeriumKinetics
from .transport import H2ArTransport

GeometryMode = Literal["constant_shell", "growing_shell", "shedding_cap"]


@dataclass
class CeriumHydrideModel:
    """
    Reduced cerium hydriding model for a single spherical particle.

    The model treats hydriding as three serial processes:

    1. H2 transport through the surrounding H2/Ar gas.
    2. H2 transport through an already-formed hydride shell.
    3. Interfacial consumption of H2 at the Ce / CeH2 reaction front.

    The state variable tracked in time is the radius of the remaining Ce core.
    At every time step the model solves a steady, coupled resistance problem to
    obtain the instantaneous H2 uptake rate. The resulting rate is then used to
    move the reaction front inward.

    Notes for the report
    --------------------
    - This is an isothermal model.
    - Gas transport is treated with a spherical Stefan-corrected expression for
      H2 diffusing inward through stagnant Ar.
    - Shell transport is treated with a simple spherical diffusion resistance.
    - The interfacial rate law is a placeholder fit derived for solid cerium,
      so it should be presented as an effective kinetics model rather than as a
      definitive liquid-droplet mechanism.
    """

    r0_m: float
    R_far_m: float
    transport: H2ArTransport
    kinetics: SarussiSolidCeriumKinetics

    geometry_mode: GeometryMode = "constant_shell"
    delta_shell_m: float = 100.0e-9

    rho_ce_kg_m3: float = 6680.0
    rho_ceh2_kg_m3: Optional[float] = None

    D_shell_solid_m2_s: float = 4.0e-8
    T_shell_melt_K: float = 1690.0
    D_shell_molten_multiplier: float = 10.0

    pure_h2_cutoff: float = 0.999
    shell_tol_m: float = 1.0e-12
    mixed_band: float = 3.0

    def __post_init__(self) -> None:
        if self.r0_m <= 0.0:
            raise ValueError("r0_m must be positive.")
        if self.R_far_m <= self.r0_m:
            raise ValueError("R_far_m must be greater than r0_m.")
        if not (0.0 < self.pure_h2_cutoff <= 1.0):
            raise ValueError("pure_h2_cutoff must be in (0, 1].")
        if self.geometry_mode in ("growing_shell", "shedding_cap") and self.rho_ceh2_kg_m3 is None:
            raise ValueError(
                "rho_ceh2_kg_m3 must be provided for 'growing_shell' or 'shedding_cap'."
            )

    # ------------------------------------------------------------------
    # Geometry and bookkeeping helpers
    # ------------------------------------------------------------------
    def initial_ce_moles(self) -> float:
        """Return the initial amount of cerium available to react."""
        return self.ce_moles_from_core_radius(self.r0_m)

    def ce_moles_from_core_radius(self, a_m: float) -> float:
        """Convert a Ce core radius into the corresponding remaining Ce moles."""
        a_m = max(float(a_m), 0.0)
        return (4.0 / 3.0) * np.pi * a_m**3 * self.rho_ce_kg_m3 / M_CE

    def conversion_fraction_from_core_radius(self, a_m: float) -> float:
        """
        Return the fraction of the initial cerium that has reacted.

        The definition is based on the change in Ce core volume, which is the
        natural progress variable for the front-tracking formulation.
        """
        initial_moles = self.initial_ce_moles()
        if initial_moles <= 0.0:
            return 0.0
        remaining_moles = self.ce_moles_from_core_radius(a_m)
        conversion = 1.0 - remaining_moles / initial_moles
        return float(np.clip(conversion, 0.0, 1.0))

    def core_radius_from_conversion_fraction(self, conversion_fraction: float) -> float:
        """Invert the conversion definition to recover the Ce core radius."""
        conversion_fraction = float(np.clip(conversion_fraction, 0.0, 1.0))
        return self.r0_m * (1.0 - conversion_fraction) ** (1.0 / 3.0)

    def natural_outer_radius_m(self, a_m: float) -> float:
        """
        Compute the outer radius that follows from mass conservation alone.

        This helper is only used for geometry modes in which the hydride shell
        is allowed to expand naturally as more Ce is converted to CeH2.
        """
        if self.rho_ceh2_kg_m3 is None:
            raise ValueError("rho_ceh2_kg_m3 is required for natural shell growth.")

        a_m = max(float(a_m), 0.0)
        n_ce_initial = self.initial_ce_moles()
        n_ce_remaining = self.ce_moles_from_core_radius(a_m)
        n_ce_reacted = max(n_ce_initial - n_ce_remaining, 0.0)

        V_core = n_ce_remaining * M_CE / self.rho_ce_kg_m3
        V_hydride = n_ce_reacted * M_CEH2 / self.rho_ceh2_kg_m3
        V_total = V_core + V_hydride
        return (3.0 * V_total / (4.0 * np.pi)) ** (1.0 / 3.0)

    def outer_radius_m(self, a_m: float) -> float:
        """
        Return the particle outer radius for the selected geometry model.

        The current report figures use ``constant_shell`` because the hydride
        density needed for a reliable natural-growth model is not yet pinned
        down from the literature.
        """
        if self.geometry_mode == "constant_shell":
            return max(float(a_m), 0.0) + self.delta_shell_m

        natural_radius = self.natural_outer_radius_m(a_m)

        if self.geometry_mode == "growing_shell":
            return natural_radius
        if self.geometry_mode == "shedding_cap":
            return min(natural_radius, max(float(a_m), 0.0) + self.delta_shell_m)

        raise ValueError(f"Unknown geometry_mode: {self.geometry_mode}")

    # ------------------------------------------------------------------
    # Transport and kinetics coefficients
    # ------------------------------------------------------------------
    def D_shell_m2_s(self, T_K: float) -> float:
        """
        Return the effective shell diffusivity.

        The discontinuous jump at ``T_shell_melt_K`` is a simple placeholder
        meant to capture the expectation that hydrogen transport becomes faster
        once the shell is molten or partially molten.
        """
        D_shell = self.D_shell_solid_m2_s
        if T_K >= self.T_shell_melt_K:
            D_shell *= self.D_shell_molten_multiplier
        return D_shell

    def has_gas_side_resistance(self, y_inf: float) -> bool:
        """
        Decide whether the external gas-side diffusion model is active.

        Near pure hydrogen the stagnant-carrier Stefan formulation becomes
        singular, so the model switches to a negligible gas-resistance limit.
        """
        return float(y_inf) < self.pure_h2_cutoff

    # ------------------------------------------------------------------
    # Gas-side transport
    # ------------------------------------------------------------------
    def gas_h2_molar_rate_mol_s(
        self,
        b_m: float,
        y_surface: float,
        y_inf: float,
        T_K: float,
        P_Pa: float,
    ) -> float:
        """
        Return the inward H2 molar rate through the surrounding gas.

        ``y_surface`` is the H2 mole fraction right outside the particle and
        ``y_inf`` is the far-field H2 mole fraction. The rate is positive for
        inward transport. When the pure-H2 limit is active, the gas-side
        resistance is bypassed and the function returns ``np.inf`` provided the
        requested surface state is compatible with ``y_surface <= y_inf``.
        """
        y_inf = float(np.clip(y_inf, 0.0, 1.0))
        y_surface = float(np.clip(y_surface, 0.0, y_inf))

        if y_inf <= 1.0e-16:
            return 0.0

        if not self.has_gas_side_resistance(y_inf):
            return np.inf if y_surface <= y_inf + 1.0e-15 else 0.0

        D_gas = self.transport.binary_diffusivity_h2_ar(T_K, P_Pa)
        c_total = P_Pa / (R_UNIVERSAL * T_K)
        radial_factor = (1.0 / b_m) - (0.0 if np.isinf(self.R_far_m) else 1.0 / self.R_far_m)

        # The logarithmic driving force is the standard Stefan correction for a
        # binary species diffusing through a stagnant carrier gas.
        y_inf_eff = min(max(y_inf, 1.0e-15), 1.0 - 1.0e-12)
        y_surface_eff = min(max(y_surface, 0.0), y_inf_eff)
        stefan_log_term = np.log((1.0 - y_surface_eff) / (1.0 - y_inf_eff))

        return 4.0 * np.pi * c_total * D_gas * stefan_log_term / radial_factor

    def gas_profile(
        self,
        r_values_m: np.ndarray,
        b_m: float,
        y_surface: float,
        y_inf: float,
        T_K: float,
        P_Pa: float,
    ) -> np.ndarray:
        """
        Evaluate the steady gas-phase H2 mole-fraction profile between ``b`` and
        ``R_far`` for a previously solved operating point.
        """
        r_values_m = np.asarray(r_values_m, dtype=float)
        y_inf = float(np.clip(y_inf, 0.0, 1.0))
        y_surface = float(np.clip(y_surface, 0.0, y_inf))

        if not self.has_gas_side_resistance(y_inf):
            return np.full_like(r_values_m, y_inf, dtype=float)

        D_gas = self.transport.binary_diffusivity_h2_ar(T_K, P_Pa)
        c_total = P_Pa / (R_UNIVERSAL * T_K)
        n_dot = self.gas_h2_molar_rate_mol_s(
            b_m=b_m,
            y_surface=y_surface,
            y_inf=y_inf,
            T_K=T_K,
            P_Pa=P_Pa,
        )
        exponent = n_dot * ((1.0 / r_values_m) - (1.0 / self.R_far_m)) / (4.0 * np.pi * c_total * D_gas)
        y_gas = 1.0 - (1.0 - y_inf) * np.exp(exponent)
        return np.clip(y_gas, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Shell transport
    # ------------------------------------------------------------------
    def shell_h2_molar_rate_mol_s(
        self,
        a_m: float,
        b_m: float,
        y_surface: float,
        y_interface: float,
        T_K: float,
        P_Pa: float,
    ) -> float:
        """
        Return the H2 molar rate through the hydride shell.

        The shell-side driving variable is currently represented as an effective
        normalized hydrogen variable written in a mole-fraction-like form. This
        makes the resistance bookkeeping transparent, but it should be explained
        clearly in the report as an approximate constitutive choice.
        """
        thickness = b_m - a_m
        if thickness <= self.shell_tol_m:
            return np.inf

        y_surface = float(np.clip(y_surface, 0.0, 1.0))
        y_interface = float(np.clip(y_interface, 0.0, y_surface))

        c_reference = P_Pa / (R_UNIVERSAL * T_K)
        D_shell = self.D_shell_m2_s(T_K)
        geometric_factor = a_m * b_m / thickness
        return 4.0 * np.pi * D_shell * geometric_factor * c_reference * (y_surface - y_interface)

    def shell_profile(
        self,
        r_values_m: np.ndarray,
        a_m: float,
        b_m: float,
        y_surface: float,
        y_interface: float,
    ) -> np.ndarray:
        """Return the spherical shell profile of the effective shell variable."""
        r_values_m = np.asarray(r_values_m, dtype=float)
        thickness = b_m - a_m
        if thickness <= self.shell_tol_m:
            return np.full_like(r_values_m, y_surface, dtype=float)

        denominator = (1.0 / a_m) - (1.0 / b_m)
        numerator = (1.0 / r_values_m) - (1.0 / b_m)
        y_shell = y_surface + (y_interface - y_surface) * numerator / denominator
        return np.clip(y_shell, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Interfacial kinetics
    # ------------------------------------------------------------------
    def kinetic_h2_molar_rate_mol_s(self, a_m: float, y_interface: float, T_K: float, P_Pa: float) -> float:
        """Convert the interfacial hydrogen level into an H2 uptake rate."""
        p_h2_atm = float(y_interface) * P_Pa / ONE_ATM
        return self.kinetics.h2_molar_rate_mol_s(a_m=a_m, p_h2_atm=p_h2_atm, T_K=T_K)

    # ------------------------------------------------------------------
    # One-process capacity calculations
    # ------------------------------------------------------------------
    def gas_only_capacity_mol_s(self, a_m: float, T_K: float, P_Pa: float, y_inf: float) -> float:
        """
        Capacity of the gas-side resistance acting alone.

        The particle surface is treated as a perfect sink for hydrogen so that
        the gas transport process sees the largest possible driving force.
        """
        a_m = max(float(a_m), 0.0)
        y_inf = float(np.clip(y_inf, 0.0, 1.0))
        if a_m <= 0.0 or y_inf <= 1.0e-16:
            return 0.0
        b_m = self.outer_radius_m(a_m)
        return self.gas_h2_molar_rate_mol_s(
            b_m=b_m,
            y_surface=0.0,
            y_inf=y_inf,
            T_K=T_K,
            P_Pa=P_Pa,
        )

    def shell_only_capacity_mol_s(self, a_m: float, T_K: float, P_Pa: float, y_inf: float) -> float:
        """
        Capacity of the hydride shell acting alone.

        The shell sees the full upstream hydrogen level at its outer surface and
        a perfect sink at the reaction front.
        """
        a_m = max(float(a_m), 0.0)
        y_inf = float(np.clip(y_inf, 0.0, 1.0))
        if a_m <= 0.0 or y_inf <= 1.0e-16:
            return 0.0

        b_m = self.outer_radius_m(a_m)
        if (b_m - a_m) <= self.shell_tol_m:
            return np.inf

        return self.shell_h2_molar_rate_mol_s(
            a_m=a_m,
            b_m=b_m,
            y_surface=y_inf,
            y_interface=0.0,
            T_K=T_K,
            P_Pa=P_Pa,
        )

    def kinetic_only_capacity_mol_s(self, a_m: float, T_K: float, P_Pa: float, y_inf: float) -> float:
        """
        Capacity of the interfacial kinetics acting alone.

        The interface is given access to the full far-field hydrogen mole
        fraction so that only the kinetics law limits the rate.
        """
        return self.kinetics.h2_molar_rate_from_mole_fraction(
            a_m=a_m,
            y_h2=y_inf,
            T_K=T_K,
            P_Pa=P_Pa,
        )

    # ------------------------------------------------------------------
    # Steady coupled solve
    # ------------------------------------------------------------------
    def solve_coupled_flux(self, a_m: float, T_K: float, P_Pa: float, y_inf: float) -> dict:
        """
        Solve the steady serial-resistance problem for a fixed Ce core radius.

        The unknowns are the hydrogen level at the outer shell surface and the
        hydrogen level at the reaction front. The solution is obtained by
        matching the gas-side, shell-side, and kinetic uptake rates.
        """
        a_m = max(float(a_m), 0.0)
        y_inf = float(np.clip(y_inf, 0.0, 1.0))
        b_m = self.outer_radius_m(a_m)

        if a_m <= 0.0 or y_inf <= 1.0e-16:
            return {
                "a_m": a_m,
                "b_m": b_m,
                "y_surface": 0.0,
                "y_interface": 0.0,
                "n_dot_h2_mol_s": 0.0,
                "gas_side_mode": "inactive" if y_inf <= 1.0e-16 else "pure_h2",
            }

        gas_side_active = self.has_gas_side_resistance(y_inf)
        upper_bound = max(min(y_inf, 1.0 - 1.0e-12), 1.0e-16)

        # If no shell exists, the outer-surface and interface states collapse to
        # a single unknown. This keeps the root solve especially simple.
        if (b_m - a_m) <= self.shell_tol_m:
            if not gas_side_active:
                y_interface = y_inf
                n_dot = self.kinetic_h2_molar_rate_mol_s(a_m, y_interface, T_K, P_Pa)
                return {
                    "a_m": a_m,
                    "b_m": b_m,
                    "y_surface": y_interface,
                    "y_interface": y_interface,
                    "n_dot_h2_mol_s": n_dot,
                    "gas_side_mode": "pure_h2",
                }

            def no_shell_residual(y_interface: float) -> float:
                n_gas = self.gas_h2_molar_rate_mol_s(
                    b_m=b_m,
                    y_surface=y_interface,
                    y_inf=y_inf,
                    T_K=T_K,
                    P_Pa=P_Pa,
                )
                n_kin = self.kinetic_h2_molar_rate_mol_s(
                    a_m=a_m,
                    y_interface=y_interface,
                    T_K=T_K,
                    P_Pa=P_Pa,
                )
                return n_gas - n_kin

            y_interface = brentq(no_shell_residual, 0.0, upper_bound)
            n_dot = self.kinetic_h2_molar_rate_mol_s(a_m, y_interface, T_K, P_Pa)
            return {
                "a_m": a_m,
                "b_m": b_m,
                "y_surface": y_interface,
                "y_interface": y_interface,
                "n_dot_h2_mol_s": n_dot,
                "gas_side_mode": "stefan",
            }

        def solve_interface_for_surface(y_surface: float) -> float:
            """
            For a fixed outer-surface state, solve the shell/kinetics balance.
            """

            def shell_vs_kinetics(y_interface: float) -> float:
                n_shell = self.shell_h2_molar_rate_mol_s(
                    a_m=a_m,
                    b_m=b_m,
                    y_surface=y_surface,
                    y_interface=y_interface,
                    T_K=T_K,
                    P_Pa=P_Pa,
                )
                n_kin = self.kinetic_h2_molar_rate_mol_s(
                    a_m=a_m,
                    y_interface=y_interface,
                    T_K=T_K,
                    P_Pa=P_Pa,
                )
                return n_shell - n_kin

            return brentq(shell_vs_kinetics, 0.0, max(np.nextafter(y_surface, 0.0), 1.0e-16))

        # In the pure-H2 limit the gas-side resistance disappears, so only the
        # shell/kinetics balance remains.
        if not gas_side_active:
            y_surface = y_inf
            y_interface = solve_interface_for_surface(y_surface)
            n_dot = self.shell_h2_molar_rate_mol_s(
                a_m=a_m,
                b_m=b_m,
                y_surface=y_surface,
                y_interface=y_interface,
                T_K=T_K,
                P_Pa=P_Pa,
            )
            return {
                "a_m": a_m,
                "b_m": b_m,
                "y_surface": y_surface,
                "y_interface": y_interface,
                "n_dot_h2_mol_s": n_dot,
                "gas_side_mode": "pure_h2",
            }

        # General gas + shell + kinetics case. The outer root solve adjusts the
        # outer-surface hydrogen level until gas supply matches shell transport.
        def gas_vs_shell(y_surface: float) -> float:
            y_interface = solve_interface_for_surface(y_surface)
            n_gas = self.gas_h2_molar_rate_mol_s(
                b_m=b_m,
                y_surface=y_surface,
                y_inf=y_inf,
                T_K=T_K,
                P_Pa=P_Pa,
            )
            n_shell = self.shell_h2_molar_rate_mol_s(
                a_m=a_m,
                b_m=b_m,
                y_surface=y_surface,
                y_interface=y_interface,
                T_K=T_K,
                P_Pa=P_Pa,
            )
            return n_gas - n_shell

        y_surface = brentq(gas_vs_shell, 0.0, upper_bound)
        y_interface = solve_interface_for_surface(y_surface)
        n_dot = self.shell_h2_molar_rate_mol_s(
            a_m=a_m,
            b_m=b_m,
            y_surface=y_surface,
            y_interface=y_interface,
            T_K=T_K,
            P_Pa=P_Pa,
        )
        return {
            "a_m": a_m,
            "b_m": b_m,
            "y_surface": y_surface,
            "y_interface": y_interface,
            "n_dot_h2_mol_s": n_dot,
            "gas_side_mode": "stefan",
        }

    def steady_state_radial_profile(
        self,
        a_m: float,
        T_K: float,
        P_Pa: float,
        y_inf: float,
        n_shell_points: int = 80,
        n_gas_points: int = 200,
    ) -> dict:
        """
        Assemble a report-ready radial profile for one operating point.

        The gas portion returns the actual gas-phase H2 mole fraction. The shell
        portion returns the effective shell-side variable used in the reduced
        shell-diffusion model.
        """
        state = self.solve_coupled_flux(a_m=a_m, T_K=T_K, P_Pa=P_Pa, y_inf=y_inf)
        a_m = float(state["a_m"])
        b_m = float(state["b_m"])
        y_surface = float(state["y_surface"])
        y_interface = float(state["y_interface"])

        r_shell = np.array([], dtype=float)
        y_shell = np.array([], dtype=float)
        if b_m - a_m > self.shell_tol_m:
            r_shell = np.linspace(a_m, b_m, n_shell_points)
            y_shell = self.shell_profile(
                r_values_m=r_shell,
                a_m=a_m,
                b_m=b_m,
                y_surface=y_surface,
                y_interface=y_interface,
            )

        r_gas = np.linspace(b_m, self.R_far_m, n_gas_points)
        y_gas = self.gas_profile(
            r_values_m=r_gas,
            b_m=b_m,
            y_surface=y_surface,
            y_inf=float(y_inf),
            T_K=T_K,
            P_Pa=P_Pa,
        )

        return {
            "state": state,
            "r_core_m": a_m,
            "r_outer_m": b_m,
            "r_shell_m": r_shell,
            "y_shell_effective": y_shell,
            "r_gas_m": r_gas,
            "y_gas": y_gas,
            "y_inf": float(y_inf),
            "T_K": float(T_K),
            "P_Pa": float(P_Pa),
        }

    # ------------------------------------------------------------------
    # Limitation diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _resistance_from_capacity(capacity_mol_s: float) -> float:
        """
        Convert a one-process capacity into a resistance-like scalar.

        A larger capacity means a smaller resistance. Infinite capacity
        therefore maps to zero resistance, which is exactly what we want when
        the corresponding process is effectively absent.
        """
        if np.isinf(capacity_mol_s):
            return 0.0
        if capacity_mol_s <= 1.0e-300:
            return np.inf
        return 1.0 / capacity_mol_s

    def limitation_summary(self, a_m: float, T_K: float, P_Pa: float, y_inf: float) -> dict:
        """
        Summarize which process is currently the dominant bottleneck.

        The classification is based on one-process capacities for:

        - gas diffusion,
        - shell diffusion,
        - interface kinetics.

        Their reciprocals are treated as resistance-like quantities. The
        process with the largest resistance share is labeled as the dominant
        bottleneck. A mixed flag is also returned when the two strongest
        resistance shares are within ``mixed_band`` of one another.
        """
        state = self.solve_coupled_flux(a_m=a_m, T_K=T_K, P_Pa=P_Pa, y_inf=y_inf)

        capacities = np.array(
            [
                self.gas_only_capacity_mol_s(a_m=a_m, T_K=T_K, P_Pa=P_Pa, y_inf=y_inf),
                self.shell_only_capacity_mol_s(a_m=a_m, T_K=T_K, P_Pa=P_Pa, y_inf=y_inf),
                self.kinetic_only_capacity_mol_s(a_m=a_m, T_K=T_K, P_Pa=P_Pa, y_inf=y_inf),
            ],
            dtype=float,
        )
        resistances = np.array([self._resistance_from_capacity(value) for value in capacities], dtype=float)

        finite_resistance_sum = float(np.sum(resistances[np.isfinite(resistances)]))
        if np.isinf(resistances).any():
            shares = np.zeros_like(resistances)
            dominant_indices = np.where(np.isinf(resistances))[0]
            shares[dominant_indices] = 1.0 / len(dominant_indices)
        elif finite_resistance_sum > 0.0:
            shares = resistances / finite_resistance_sum
        else:
            shares = np.zeros_like(resistances)

        ranking = np.argsort(-shares)
        primary_index = int(ranking[0])
        secondary_index = int(ranking[1])
        primary_share = float(shares[primary_index])
        secondary_share = float(shares[secondary_index])

        if secondary_share <= 1.0e-300:
            control_ratio = np.inf
        else:
            control_ratio = primary_share / secondary_share

        mixed_with = LIMITATION_NAME_BY_CODE[int(secondary_index)] if control_ratio < self.mixed_band else "none"

        return {
            "limiting_process_code": primary_index,
            "limiting_process_name": LIMITATION_NAME_BY_CODE[primary_index],
            "limiting_process_label": LIMITATION_DISPLAY_LABELS[primary_index],
            "gas_capacity_mol_s": float(capacities[LIMITATION_GAS_DIFFUSION]),
            "shell_capacity_mol_s": float(capacities[LIMITATION_SHELL_DIFFUSION]),
            "kinetics_capacity_mol_s": float(capacities[LIMITATION_KINETICS]),
            "gas_resistance_share": float(shares[LIMITATION_GAS_DIFFUSION]),
            "shell_resistance_share": float(shares[LIMITATION_SHELL_DIFFUSION]),
            "kinetics_resistance_share": float(shares[LIMITATION_KINETICS]),
            "control_ratio": float(control_ratio),
            "is_mixed": bool(control_ratio < self.mixed_band),
            "mixed_with": mixed_with,
            "overall_rate_mol_s": float(state["n_dot_h2_mol_s"]),
            "gas_side_mode": state["gas_side_mode"],
            "a_m": float(state["a_m"]),
            "b_m": float(state["b_m"]),
        }

    # Keep the original method name as a compatibility alias for the scripts.
    def classify_regime(self, a_m: float, T_K: float, P_Pa: float, y_inf: float) -> dict:
        return self.limitation_summary(a_m=a_m, T_K=T_K, P_Pa=P_Pa, y_inf=y_inf)

    def classify_limitation(self, a_m: float, T_K: float, P_Pa: float, y_inf: float) -> dict:
        return self.limitation_summary(a_m=a_m, T_K=T_K, P_Pa=P_Pa, y_inf=y_inf)

    def build_limitation_map(
        self,
        T_values_K: Sequence[float],
        y_values: Sequence[float],
        P_total_Pa: float,
        a_ref_m: Optional[float] = None,
    ) -> dict:
        """
        Build a 2D map of the dominant bottleneck over temperature/composition.
        """
        a_reference = self.r0_m if a_ref_m is None else float(a_ref_m)
        T_values_K = np.asarray(T_values_K, dtype=float)
        y_values = np.asarray(y_values, dtype=float)

        limiting_code = np.zeros((len(T_values_K), len(y_values)), dtype=int)
        mixed_mask = np.zeros_like(limiting_code, dtype=bool)
        gas_share = np.zeros_like(limiting_code, dtype=float)
        shell_share = np.zeros_like(limiting_code, dtype=float)
        kinetics_share = np.zeros_like(limiting_code, dtype=float)
        overall_rate = np.zeros_like(limiting_code, dtype=float)
        gas_mode = np.empty(limiting_code.shape, dtype=object)

        for i, T_K in enumerate(T_values_K):
            for j, y_inf in enumerate(y_values):
                summary = self.limitation_summary(
                    a_m=a_reference,
                    T_K=float(T_K),
                    P_Pa=P_total_Pa,
                    y_inf=float(y_inf),
                )
                limiting_code[i, j] = summary["limiting_process_code"]
                mixed_mask[i, j] = summary["is_mixed"]
                gas_share[i, j] = summary["gas_resistance_share"]
                shell_share[i, j] = summary["shell_resistance_share"]
                kinetics_share[i, j] = summary["kinetics_resistance_share"]
                overall_rate[i, j] = summary["overall_rate_mol_s"]
                gas_mode[i, j] = summary["gas_side_mode"]

        return {
            "T_values_K": T_values_K,
            "y_values": y_values,
            "limiting_process_code": limiting_code,
            "mixed_mask": mixed_mask,
            "gas_resistance_share": gas_share,
            "shell_resistance_share": shell_share,
            "kinetics_resistance_share": kinetics_share,
            "overall_rate_mol_s": overall_rate,
            "a_ref_m": a_reference,
            "P_total_Pa": float(P_total_Pa),
            "pure_h2_cutoff": self.pure_h2_cutoff,
            "gas_side_mode": gas_mode,
        }

    # Keep the previous name so older scripts still work.
    def build_regime_map(
        self,
        T_values_K: Sequence[float],
        y_values: Sequence[float],
        P_total_Pa: float,
        a_ref_m: Optional[float] = None,
    ) -> dict:
        return self.build_limitation_map(
            T_values_K=T_values_K,
            y_values=y_values,
            P_total_Pa=P_total_Pa,
            a_ref_m=a_ref_m,
        )

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def da_dt_m_s(self, a_m: float, n_dot_h2_mol_s: float) -> float:
        """
        Convert the instantaneous H2 uptake rate into front motion.

        Because Ce + H2 -> CeH2 consumes one mole of H2 per mole of Ce, the
        H2 molar rate directly determines how fast the Ce core shrinks.
        """
        if a_m <= 0.0:
            return 0.0
        return -M_CE * n_dot_h2_mol_s / (4.0 * np.pi * a_m**2 * self.rho_ce_kg_m3)

    def simulate(
        self,
        T_K: float,
        P_Pa: float,
        y_inf: float,
        t_final_s: float,
        dt_s: float,
        a0_m: Optional[float] = None,
    ) -> dict:
        """
        Integrate the shrinking-core model forward in time with explicit Euler.

        The time integrator is intentionally simple because the focus of the
        course project is on trends and characteristic times rather than on
        stiff numerical integration.
        """
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")

        a_m = self.r0_m if a0_m is None else float(a0_m)
        time_values = np.arange(0.0, t_final_s + dt_s, dt_s)

        output = {
            "t_s": [],
            "a_m": [],
            "b_m": [],
            "shell_thickness_m": [],
            "n_dot_h2_mol_s": [],
            "da_dt_m_s": [],
            "y_surface": [],
            "y_interface": [],
            "conversion_fraction": [],
            "remaining_ce_fraction": [],
        }

        for time_s in time_values:
            state = self.solve_coupled_flux(a_m=a_m, T_K=T_K, P_Pa=P_Pa, y_inf=y_inf)
            da_dt = self.da_dt_m_s(a_m=a_m, n_dot_h2_mol_s=state["n_dot_h2_mol_s"])
            conversion_fraction = self.conversion_fraction_from_core_radius(state["a_m"])

            output["t_s"].append(time_s)
            output["a_m"].append(state["a_m"])
            output["b_m"].append(state["b_m"])
            output["shell_thickness_m"].append(state["b_m"] - state["a_m"])
            output["n_dot_h2_mol_s"].append(state["n_dot_h2_mol_s"])
            output["da_dt_m_s"].append(da_dt)
            output["y_surface"].append(state["y_surface"])
            output["y_interface"].append(state["y_interface"])
            output["conversion_fraction"].append(conversion_fraction)
            output["remaining_ce_fraction"].append(1.0 - conversion_fraction)

            a_m = max(a_m + dt_s * da_dt, 0.0)
            if a_m <= 0.0:
                break

        for key, value in output.items():
            output[key] = np.asarray(value, dtype=float)
        return output

    def estimate_initial_timescale_s(self, T_K: float, P_Pa: float, y_inf: float, a0_m: Optional[float] = None) -> float:
        """
        Estimate a characteristic completion time from the initial uptake rate.

        This is used only to choose a reasonable integration window for the
        characteristic-time calculations.
        """
        a0_m = self.r0_m if a0_m is None else float(a0_m)
        state = self.solve_coupled_flux(a_m=a0_m, T_K=T_K, P_Pa=P_Pa, y_inf=y_inf)
        initial_rate = float(state["n_dot_h2_mol_s"])
        if initial_rate <= 1.0e-300:
            return np.inf
        return self.ce_moles_from_core_radius(a0_m) / initial_rate

    @staticmethod
    def _interpolate_time_at_target(times_s: np.ndarray, values: np.ndarray, target: float) -> float:
        """Linearly interpolate the time at which a monotonic profile hits a target."""
        if values.size == 0 or target > values[-1]:
            return np.nan
        if target <= values[0]:
            return float(times_s[0])

        index = int(np.searchsorted(values, target, side="left"))
        if index == 0:
            return float(times_s[0])

        left_time = float(times_s[index - 1])
        right_time = float(times_s[index])
        left_value = float(values[index - 1])
        right_value = float(values[index])

        if right_value <= left_value + 1.0e-300:
            return right_time

        fraction = (target - left_value) / (right_value - left_value)
        return left_time + fraction * (right_time - left_time)

    def characteristic_times(
        self,
        T_K: float,
        P_Pa: float,
        y_inf: float,
        conversion_targets: Sequence[float] = (0.1, 0.5, 0.9),
        dt_s: Optional[float] = None,
        t_final_s: Optional[float] = None,
        a0_m: Optional[float] = None,
        auto_extend: bool = True,
        max_time_s: float = 100.0,
        return_simulation: bool = False,
    ) -> dict:
        """
        Estimate report-friendly characteristic hydriding times.

        The default targets correspond to 10%, 50%, and 90% conversion of the
        initial cerium inventory. The integration horizon is chosen from the
        initial uptake rate and is automatically extended until the requested
        targets are reached or ``max_time_s`` is hit.
        """
        targets = np.asarray(sorted(float(np.clip(value, 0.0, 0.999999)) for value in conversion_targets), dtype=float)
        if targets.size == 0:
            raise ValueError("At least one conversion target must be supplied.")

        initial_timescale_s = self.estimate_initial_timescale_s(T_K=T_K, P_Pa=P_Pa, y_inf=y_inf, a0_m=a0_m)
        if t_final_s is None:
            if np.isfinite(initial_timescale_s):
                t_final_s = min(max(5.0 * initial_timescale_s, 1.0e-6), max_time_s)
            else:
                t_final_s = max_time_s
        if dt_s is None:
            if np.isfinite(initial_timescale_s):
                dt_s = float(np.clip(initial_timescale_s / 600.0, 1.0e-7, 5.0e-4))
            else:
                dt_s = 1.0e-4

        current_t_final_s = min(float(t_final_s), max_time_s)

        while True:
            simulation = self.simulate(
                T_K=T_K,
                P_Pa=P_Pa,
                y_inf=y_inf,
                t_final_s=current_t_final_s,
                dt_s=dt_s,
                a0_m=a0_m,
            )
            conversion_fraction = simulation["conversion_fraction"]
            if (
                conversion_fraction.size > 0
                and conversion_fraction[-1] >= targets[-1] - 1.0e-9
            ) or not auto_extend or current_t_final_s >= max_time_s:
                break
            current_t_final_s = min(2.0 * current_t_final_s, max_time_s)

        target_times_s = np.array(
            [self._interpolate_time_at_target(simulation["t_s"], simulation["conversion_fraction"], target) for target in targets],
            dtype=float,
        )

        results = {
            "conversion_targets": targets,
            "target_times_s": target_times_s,
            "integration_dt_s": float(dt_s),
            "integration_t_final_s": float(current_t_final_s),
            "initial_timescale_s": float(initial_timescale_s),
            "reached_highest_target": bool(
                simulation["conversion_fraction"].size > 0
                and simulation["conversion_fraction"][-1] >= targets[-1] - 1.0e-9
            ),
        }

        for target, time_s in zip(targets, target_times_s):
            label = int(round(100.0 * target))
            results[f"t_{label:d}_s"] = float(time_s)

        if return_simulation:
            results["simulation"] = simulation
        return results
