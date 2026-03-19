# Cerium Hydriding Model

This repository contains a reduced theoretical/computational model for hydriding
of a single spherical cerium particle exposed to an H2/Ar atmosphere.

The current model treats three serial processes:

1. gas-side diffusion of H2 through the surrounding H2/Ar mixture,
2. diffusion of hydrogen through a hydride shell, and
3. finite kinetics at the Ce / CeH2 reaction front.

The main purpose of the code is to answer a quasi-steady regime question:
which of those three processes is the dominant bottleneck under a given set of
conditions?

## Repository layout

- `src/cerium_hydride/constants.py`
  Physical constants and the three bottleneck labels used throughout the code.
- `src/cerium_hydride/transport.py`
  Cantera-backed helper for H2/Ar binary diffusivity.
- `src/cerium_hydride/kinetics.py`
  Effective placeholder kinetics law for hydrogen uptake by cerium.
- `src/cerium_hydride/model.py`
  Core solver plus bottleneck diagnostics for the quasi-steady model.
- `src/cerium_hydride/plotting.py`
  Plotting helpers for bottleneck maps and radial-profile comparisons.
- `scripts/run_single_case.py`
  Smoke test for one operating point.
- `scripts/run_regime_map.py`
  Example driver that builds one bottleneck map.
- `scripts/make_report_figures.py`
  Generates the figures used by the LaTeX report scaffold.
- `report/main.tex`
  Starting report in LaTeX.
- `report/references.bib`
  Bibliography file in BibTeX format.

## Installation

From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Running the model

Single-case diagnostic:

```bash
PYTHONPATH=src python scripts/run_single_case.py
```

One bottleneck map:

```bash
PYTHONPATH=src python scripts/run_regime_map.py
```

Report figures:

```bash
PYTHONPATH=src python scripts/make_report_figures.py
```

The figures are written to `report/figures/`.

## Building the report

From the repository root:

```bash
cd report
latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
```

The compiled PDF will be written to `report/build/main.pdf`.

## Modeling notes

- The gas-side model uses a Stefan-corrected spherical diffusion expression for
  H2 moving inward through stagnant Ar.
- The shell-diffusion model uses a reduced spherical resistance with an
  effective shell-side hydrogen variable.
- The current kinetics law is explicitly a placeholder fit. It is useful for
  exploring whether kinetics can compete with transport, but it should not be
  presented as a validated liquid-cerium surface mechanism.
- The current report workflow is intentionally quasi-steady. It compares gas
  diffusion, shell diffusion, and kinetics at fixed temperature rather than
  trying to interpret an isothermal transient conversion history.

## Suggested first sweeps

- particle diameter: 10, 50, 100 um
- temperature: 1100 to 1800 K
- total pressure: 0.25, 1.0, 5.0 atm
- far-field H2 mole fraction: 1e-4 to 1.0
- shell thickness: 20, 100, 500 nm
- shell diffusivity: `4e-9`, `4e-8`, `4e-7 m^2/s`
