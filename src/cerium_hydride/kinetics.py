from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import M_CE, ONE_ATM, R_UNIVERSAL


@dataclass
class SarussiSolidCeriumKinetics:
    """
    Pressure/temperature fit used here as an effective placeholder kinetics law.

    U(P,T) = A'(T) * sqrt(P) / (sqrt(P) + sqrt(B(T)))

    with P in atm and U in m/s after unit conversion.

    Important caveat
    ----------------
    This relation comes from solid-cerium hydriding data. In the current course
    project it is being used as a reduced interfacial model, not as a fully
    validated liquid-droplet reaction mechanism.
    """

    A0_cm_s: float = 0.80
    B0_sqrtT_atm: float = 88.7
    E_in_J_per_mol_H: float = 5.54 * 4184.0
    E_des_J_per_mol_H: float = 2.02 * 4184.0
    rho_ce_kg_m3: float = 6680.0

    def front_velocity_m_s(self, p_h2_atm: float, T_K: float) -> float:
        """
        Return the inward front velocity associated with the fitted law.

        The Arrhenius-like temperature dependence enters through ``A'(T)`` and
        ``B(T)``, while the square-root pressure dependence comes directly from
        the original fit form.
        """
        p_h2_atm = max(float(p_h2_atm), 0.0)
        A_prime = (self.A0_cm_s * 1.0e-2) * np.exp(
            -self.E_in_J_per_mol_H / (R_UNIVERSAL * T_K)
        )
        B_T = self.B0_sqrtT_atm * np.sqrt(T_K) * np.exp(
            -self.E_des_J_per_mol_H / (R_UNIVERSAL * T_K)
        )
        sqrt_p = np.sqrt(p_h2_atm)
        sqrt_B = np.sqrt(B_T)
        return A_prime * sqrt_p / (sqrt_p + sqrt_B + 1.0e-300)

    def h2_molar_rate_mol_s(self, a_m: float, p_h2_atm: float, T_K: float) -> float:
        """
        Convert front speed to total H2 molar rate for:
            Ce + H2 -> CeH2

        The factor ``rho_ce / M_ce`` converts a geometric front speed into a
        molar consumption rate of cerium, which is equal to the H2 molar uptake
        rate for the 1:1 reaction stoichiometry.
        """
        if a_m <= 0.0:
            return 0.0
        U = self.front_velocity_m_s(p_h2_atm, T_K)
        area = 4.0 * np.pi * a_m**2
        return area * (self.rho_ce_kg_m3 / M_CE) * U

    def h2_molar_rate_from_mole_fraction(self, a_m: float, y_h2: float, T_K: float, P_Pa: float) -> float:
        """Convenience wrapper that converts a mole fraction into partial pressure."""
        p_h2_atm = max(float(y_h2), 0.0) * P_Pa / ONE_ATM
        return self.h2_molar_rate_mol_s(a_m=a_m, p_h2_atm=p_h2_atm, T_K=T_K)
