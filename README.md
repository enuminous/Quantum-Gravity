# QST-001 · Standard Model + Einstein Gravity + EFMW Pressure-Cycle Research Laboratory

<!-- ENUMINOUS-UNIFIED-NAV:START -->
## Repository navigation

**eNuminous network:** [All public repositories](https://github.com/enuminous/EFMW/blob/main/REPOSITORY_NAVIGATION.md) · [EFMW](https://github.com/enuminous/EFMW) · [102 equations](https://github.com/enuminous/Monolithic_102_EFMW) · [Frozen Lean](https://github.com/enuminous/Monolithic-102-Zoo-Lean4) · [Zoo](https://github.com/enuminous/Tortoise) · [Papers](https://github.com/enuminous/papers) · [Quantum geometry](https://github.com/enuminous/Base888-Quantum-Geometry)

**This project:** [Repository home](https://github.com/enuminous/Quantum-Gravity) · [Subject index](https://github.com/enuminous/EFMW/blob/main/REPOSITORY_NAVIGATION.md#fundamental-physics-cosmology-and-quantum-geometry)

<details open>
<summary><strong>On this page</strong></summary>

- [Start here](#start-here)
- [Files — all at repository root](#files--all-at-repository-root)
- [Reproduce](#reproduce)
- [Physical content / provenance](#physical-content--provenance)
  - [Links](#links)
- [Publication](#publication)
- [Connected physics research units](#connected-physics-research-units)


</details>

**Related repositories:** [QST](https://github.com/enuminous/QST) · [Base888-Quantum-Geometry](https://github.com/enuminous/Base888-Quantum-Geometry) · [Standard-Model-Reduction](https://github.com/enuminous/Standard-Model-Reduction) · [EFMW-Physics-Lab](https://github.com/enuminous/EFMW-Physics-Lab) · [papers](https://github.com/enuminous/papers) · [Cosmology](https://github.com/enuminous/Cosmology) · [EFMW](https://github.com/enuminous/EFMW)

> **Evidence boundary:** A link between projects is navigation, not an assertion of proven equivalence, experimental validation, or shared authorship. See each repository’s own status and review notes.
<!-- ENUMINOUS-UNIFIED-NAV:END -->


**Flat-stacked GitHub ZIP — release candidate v0.1 — 2026-10-27**

> **Boundary:** This is a classical, low-energy benchmark plus a *separate* 888-state unitary toy. It is **not** a derived complete quantum theory of spacetime, not a quantum gravity simulation, and not a complete Standard Model implementation.

## Start here

**Double-click `index.html`** for a standalone, offline desktop explorer. Use the controls to compare a free homogeneous scalar oscillator against classical FLRW expansion, and contrast the stress-energy pressure law with the *hypothesized* linear pressure mapping in the EFMW–Rahasya pressure-cycle note.

Read [`QST-001.md`](QST-001.md) for the action, field variations, pressure-frequency derivation, Friedmann equation, finite-state quantum illustration, and the missing spacetime-emergence theorem. Read [`CLAIMS.md`](CLAIMS.md) before making scientific claims about the outputs.

## Files — all at repository root

| File | Purpose |
|---|---|
| `index.html` | Standalone interactive research explorer; CSS and JavaScript embedded |
| `QST-001.md` | Mathematical paper, conventions, limitations, links |
| `CLAIMS.md` | PASS vs UNRESOLVED boundaries and provenance |
| `gravity_lab.py` | Standard-library classical scalar/FLRW RK4 and optional illustrative occupancy |
| `base888_toy.py` | 888-level exactly unitary *ring* illustration, no emergent gravity |
| `test_qst001.py` | Six standard-library unit tests |
| `README.md` | This GitHub front door |
| `MANIFEST.sha256` | Cryptographic hashes of payload files at packaging time |

No installation or web connection is necessary for the browser demo. Python scripts require Python 3.10+ and no pip packages.

## Reproduce

```bash
python3 -m unittest -v test_qst001.py
python3 gravity_lab.py --mode flrw --out qst001-results.json
python3 gravity_lab.py --mode minkowski --duration 20 --format csv --out qst001-flat.csv
python3 base888_toy.py --time 2.5
```

The JSON/CSV outputs are generated next to the scripts by default and are not frozen scientific observations. The test command assumes the extracted package root is the current working directory.

## Physical content / provenance

- The Standard Model Lagrangian and Einstein–Hilbert gravity are **assumed established inputs**, not results derived by the package.
- The canonical additional scalar and its mapping to EFMW symbols are **candidate assumptions**.
- The homogeneous scalar's stress-energy pressure has double the oscillator's frequency in the flat test-field limit. This holds only under the stated assumptions.
- The pressure threshold and cavity-occupancy kinetics are illustrative. Scalar cosmological pressure is **not** identical to fluid pressure in Rahasya's proposed medium without another constitutive derivation.
- Base888 is represented by a finite 888-level *toy* with explicit unitary dynamics. No continuum spacetime or graviton theorem follows from choosing 888 levels.

### Links

- [RHC-001: Wright's EFMW–Rahasya pressure cycle formalization](https://github.com/enuminous/papers/blob/main/RHC-001-EFMW-Rahasya-Pressure-Cycle-Formalization.md)
- [EFMW canonical review entry](https://github.com/enuminous/EFMW)
- [Base888 companion exploration](https://enuminous.github.io/Base888-Quantum-Geometry/) (external site; not part of this archive, independently unverified here)
- [Donoghue, gravity as effective field theory (1994)](https://doi.org/10.1103/PhysRevD.50.3874)

## Publication

This ZIP is GitHub-ready and *flat-stacked*: upload its contents directly to the root of a GitHub repository, preserving filenames. **Creation of this ZIP does not itself push changes to a remote repository.** A public repository release and scientific peer review are separate actions.

**Licensing:** No license has been chosen for the new package. This does not alter license or attribution requirements of linked external sources. The user/rights holder should select licensing terms before accepting outside contributions.

<!-- ENUMINOUS-RESEARCH-CHAIN:START -->
## Connected physics research units

**Established-theory reconstructions and baseline laboratories:** [Standard-Model-Reduction](https://github.com/enuminous/Standard-Model-Reduction) · [Higgs-Gauge-Lab](https://github.com/enuminous/Higgs-Gauge-Lab) · [EFMW-Physics-Lab](https://github.com/enuminous/EFMW-Physics-Lab)

**Proposed quantum-geometry bridge:** [Base888-Quantum-Geometry](https://github.com/enuminous/Base888-Quantum-Geometry) · [QST-001](https://github.com/enuminous/Quantum-Gravity/blob/main/QST-001.md) · [QST-002](https://github.com/enuminous/QST/blob/main/QST-002.md)

**EFMW definitions, pressure-cycle model, and spacetime investigations:** [Canonical 102 equations](https://github.com/enuminous/Monolithic_102_EFMW) · [Frozen Lean review](https://github.com/enuminous/Monolithic-102-Zoo-Lean4) · [RHC-001: Rahasya pressure-cycle formalization](https://github.com/enuminous/papers/blob/main/RHC-001-EFMW-Rahasya-Pressure-Cycle-Formalization.md) · [EFMW-Lagrangian](https://github.com/enuminous/EFMW-Lagrangian) · [Cosmology](https://github.com/enuminous/Cosmology)

> **Interpretation:** Cross-links represent research dependencies and comparisons. The Standard Model and continuum gravity are inputs in current QST calculations; their emergence from Base888 and the physical status of the pressure cycle remain **UNRESOLVED**.
<!-- ENUMINOUS-RESEARCH-CHAIN:END -->
