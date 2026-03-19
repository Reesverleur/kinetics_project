from __future__ import annotations

import numpy as np

from cerium_hydride import (
    ONE_ATM,
    CeriumHydrideModel,
    H2ArTransport,
    SarussiSolidCeriumKinetics,
    plot_limitation_map,
)


def main() -> None:
    """Build and display one bottleneck map for a reference particle size."""
    transport = H2ArTransport(mechanism="gri30.yaml", reference_x_h2=0.5)
    kinetics = SarussiSolidCeriumKinetics(rho_ce_kg_m3=6680.0)

    model = CeriumHydrideModel(
        r0_m=25.0e-6,  # 50 um diameter particle
        R_far_m=2.0e-3,
        transport=transport,
        kinetics=kinetics,
        geometry_mode="constant_shell",
        delta_shell_m=100.0e-9,
        rho_ce_kg_m3=6680.0,
        D_shell_solid_m2_s=4.0e-8,
        T_shell_melt_K=1690.0,
        D_shell_molten_multiplier=10.0,
        pure_h2_cutoff=0.999,
        mixed_band=3.0,
    )

    T_values_K = np.linspace(1100.0, 1750.0, 60)
    y_values = np.linspace(1.0e-4, 1.0, 120)

    limitation_map = model.build_limitation_map(
        T_values_K=T_values_K,
        y_values=y_values,
        P_total_Pa=ONE_ATM,
        a_ref_m=model.r0_m,
    )

    plot_limitation_map(limitation_map)


if __name__ == "__main__":
    main()
