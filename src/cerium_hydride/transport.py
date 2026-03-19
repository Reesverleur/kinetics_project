from __future__ import annotations

from dataclasses import dataclass

try:
    import cantera as ct
except ImportError:  # pragma: no cover
    ct = None


@dataclass
class H2ArTransport:
    """
    Binary H2-Ar transport helper.

    Notes
    -----
    For an ideal binary pair, the binary diffusion coefficient is primarily a
    function of temperature and pressure rather than mixture composition.
    To avoid numerical pathologies near x_H2 -> 0 or 1, this helper evaluates
    D_H2-Ar at a fixed reference composition by default.
    """

    mechanism: str = "gri30.yaml"
    reference_x_h2: float = 0.5

    def __post_init__(self) -> None:
        if ct is None:
            raise ImportError(
                "Cantera is not installed. Install it first, e.g. `pip install cantera`."
            )
        # The mechanism file is used only as a thermodynamic/transport database
        # here. No gas-phase reaction network is integrated in this project.
        self.gas = ct.Solution(self.mechanism)
        self.i_h2 = self.gas.species_index("H2")
        self.i_ar = self.gas.species_index("AR")

    def binary_diffusivity_h2_ar(self, T_K: float, P_Pa: float) -> float:
        """
        Return the H2-Ar binary diffusion coefficient from Cantera.

        The reference composition is held fixed to keep the binary coefficient
        evaluation numerically well behaved near pure Ar and pure H2 limits.
        """
        x_h2 = min(max(float(self.reference_x_h2), 1.0e-9), 1.0 - 1.0e-9)
        self.gas.TPX = T_K, P_Pa, {"H2": x_h2, "AR": 1.0 - x_h2}
        return float(self.gas.binary_diff_coeffs[self.i_h2, self.i_ar])
