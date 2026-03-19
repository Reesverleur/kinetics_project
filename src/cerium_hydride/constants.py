from __future__ import annotations

R_UNIVERSAL = 8.31446261815324  # J/mol/K
ONE_ATM = 101325.0              # Pa

M_CE = 140.116e-3               # kg/mol
M_H2 = 2.01588e-3               # kg/mol
M_CEH2 = M_CE + M_H2            # kg/mol

LIMITATION_GAS_DIFFUSION = 0
LIMITATION_SHELL_DIFFUSION = 1
LIMITATION_KINETICS = 2

LIMITATION_NAME_BY_CODE = {
    LIMITATION_GAS_DIFFUSION: "gas-diffusion-limited",
    LIMITATION_SHELL_DIFFUSION: "shell-diffusion-limited",
    LIMITATION_KINETICS: "kinetic-limited",
}

LIMITATION_DISPLAY_LABELS = {
    LIMITATION_GAS_DIFFUSION: "Gas diffusion",
    LIMITATION_SHELL_DIFFUSION: "Shell diffusion",
    LIMITATION_KINETICS: "Kinetics",
}
