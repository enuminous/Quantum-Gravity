# QST-001 reproducibility record

**Executed:** 2026-10-27  
**Toolchain:** Python 3.13.5, Node.js v22.16.0, standard library only.  
**Scope:** This records checks of the package's toy mathematics and source syntax; it is **not** external peer review or experimental physics.

## Automated tests

Command: `python3 -m unittest -v test_qst001.py`

- `test_scalar_stress_pressure_double_frequency` — PASS (checks closed-form quadratic pressure relation under a free flat oscillator).
- `test_minkowski_oscillator_and_energy` — PASS (numerical flat-field ODE and conserved scalar energy).
- `test_flrw_constraint_and_friction` — PASS (independently integrated $H$ versus Friedmann constraint; damping and expansion).
- `test_occupancy_invariant_interval` — PASS (finite-rate occupancy remains between 0 and 1).
- `test_888_unitary_norm` — PASS (numerical preservation of 888-state Hilbert norm in Fourier basis).
- `test_linear_pressure_explicitly_distinct` — PASS (model distinguishes assumed EFMW linear pressure from stress-derived pressure).

**Result:** 6 tests passed.

## Classical FLRW run

Command: `python3 gravity_lab.py --mode flrw --duration 60 --out qst001-results.json`

- 6,001 samples with `dt=0.01` in dimensionless units.
- Maximum absolute **independently monitored** Friedmann constraint residual: `1.690606422216634e-12` (toy dimensionless units).
- Final scale factor: `15.978197433827516` (toy parameters, **not cosmological calibration**).

## Minkowski run

Command: `python3 gravity_lab.py --mode minkowski --duration 20 --format csv --out qst001-flat.csv`

- 2,001 samples.
- Scale factor stays 1 by construction.
- Einstein backreaction is intentionally excluded; a nonzero homogeneous scalar in exact Minkowski space is generally **not** a solution of the fully coupled Einstein equations at finite $G$.

## Base888 demonstration

Command: `python3 base888_toy.py --time 2.5`

- Finite Hilbert-space dimension: 888.
- Squared norm: `1.0`.
- Return probability at `t=2.5`, nearest-neighbor hopping `J=1`: `0.03154061318127732`.
- Geometry used: *assumed one-dimensional ring*; does not imply four-dimensional spacetime or quantum gravity.

## Browser verification

The inline JavaScript in `index.html` passed Node.js syntax compilation (`new Function(script)`). A Chromium headless visual preview could not be completed in this execution environment, so no rendered layout verification is claimed. The app is self-contained (no network assets or external dependencies).

## Formal proof status

No Lean toolchain was available during package creation, so **no Lean build, theorem verification, or formal proof result is claimed**. Analytic equations in `QST-001.md` require independent checking before citation as certified results.
