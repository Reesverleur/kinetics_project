from __future__ import annotations

from cerium_hydride import ONE_ATM, CeriumHydrideModel, H2ArTransport, SarussiSolidCeriumKinetics


def main() -> None:
    """
    Run one representative operating point and print the main diagnostics.

    This script is meant to be a quick smoke test for the model and a simple
    way to inspect how the three candidate bottlenecks compare for one case.
    """
    transport = H2ArTransport(mechanism="gri30.yaml", reference_x_h2=0.5)
    kinetics = SarussiSolidCeriumKinetics(rho_ce_kg_m3=6680.0)

    model = CeriumHydrideModel(
        r0_m=25.0e-6,
        R_far_m=2.0e-3,
        transport=transport,
        kinetics=kinetics,
        geometry_mode="constant_shell",
        delta_shell_m=100.0e-9,
        D_shell_solid_m2_s=4.0e-8,
        pure_h2_cutoff=0.999,
    )

    T_K = 1450.0
    y_inf = 0.02

    state = model.solve_coupled_flux(a_m=model.r0_m, T_K=T_K, P_Pa=ONE_ATM, y_inf=y_inf)
    limitation = model.classify_limitation(a_m=model.r0_m, T_K=T_K, P_Pa=ONE_ATM, y_inf=y_inf)
    print("Steady coupled state")
    for key, value in state.items():
        print(f"{key:24s}: {value}")

    print("\nDominant bottleneck summary")
    for key, value in limitation.items():
        print(f"{key:24s}: {value}")


if __name__ == "__main__":
    main()
