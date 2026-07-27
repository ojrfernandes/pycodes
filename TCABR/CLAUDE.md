# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A collection of standalone Python scripts for post-processing and plotting M3D-C1 (fusion-io/PUMI ecosystem)
and FLARE (RMP field-line-tracing code) output for the TCABR tokamak. There is no package, no build system,
no test suite, and no CI. Each `.py` file at the repo root is an independent CLI tool (argparse-driven) that
is also importable as a module — the module-level function shares the file's name
(e.g. `flare_surfmn.py` defines `flare_surfmn()`).

This directory is one project inside the larger `~/pycodes` git repository; sibling directories
(e.g. `AUG/`) are analogous per-machine tool collections and are *not* covered by this file.

## Running scripts

Scripts are executable (`chmod +x`) with a shebang pointing at a specific venv
(`#!/home/jfernandes/.venv/bin/python`), so they run directly, e.g.:

```
./flare_model_gen.py I ./models 3 30 ./coil_sets 1
./m3dc1_units.py resistivity 7.4e-7
```

or via `python3 <script>.py ...` / `from <script> import <script_fn>`. `phase_grid.py` is the one
non-executable file at the root — it's a pure helper module (`phase_grid_size`), not a CLI.

External dependencies of note (not on PyPI — come from the M3D-C1/FLARE installs, must already be on
`PYTHONPATH`/importable in the venv): `fpy`, `m3dc1` (M3D-C1 Python bindings: `eval_field`,
`flux_coordinates`, `eigenfunction`), `flare` (FLARE's `model`/`analysis` API). `shapely`, `scipy`, `numpy`,
`matplotlib` are regular PyPI deps.

## Architecture

### RMP phase-scan pipeline (flare_*, plot_phase_map/plot_flare_surfmn/plot_flare_harmonic/plot_flare_chirikov)

This is the core, multi-file pipeline in the repo. Stages, in order:

1. **`flare_model_gen.py`** — for a given toroidal mode `n_tor` and phase step `d_phase`, builds one
   FLARE model directory (`.bfield` + `.boundary` files) per (phase_L, phase_U) combination on the grid,
   pointing at pre-existing M3D-C1 coil-set outputs (`IL_set_*/C1.h5`, `IM_set_000/C1.h5`, `IU_set_*/C1.h5`,
   or the `CP*` equivalents). Two coil families are supported (`'I'` vs `'CP'`), each with lower/middle/upper
   sets. Two phasing strategies: `flare_phase=True` (default) reuses the same `*_set_000` M3D-C1 run for every
   grid point and encodes the phase shift in the `.bfield` file's `phase:` field (fast — no rerun of M3D-C1
   needed); `flare_phase=False` instead expects a separate M3D-C1 run per phase (`*_set_<phase>/C1.h5`) with
   `phase: 0.0` always.
2. **`flare_phase_map.py`** — iterates the same (phase_L, phase_U) grid, and for each model directory spawns
   `flare_surfmn.py` as a subprocess (via `ProcessPoolExecutor`, one FLARE model loaded per worker) writing
   one `.npz` per grid point plus a per-point log file.
3. **`flare_surfmn.py`** — loads one FLARE model and computes the poloidal Fourier spectrum
   `db_matrix(psiN, m)` of the perturbed field, plus its value interpolated onto the resonant surfaces
   (q = m/n_tor) via `fluxsurf_params()`. Normalization is
   `Bmn [G] = 2*(2*pi)^2*|Phi_mn(psiN)|/area(psiN)*1e4` — **this formula was reverse-derived and validated
   against IDL's `plot_br.pro`/`schaffer_plot.pro`** (see `validate_surfmn.py`'s module docstring for the
   full derivation, including the PEST-Jacobian cancellation and the factor-of-2 from IDL's complex-harmonic
   convention). Don't "simplify" it without reading that docstring first. `n_pol` (FFT resolution) and
   `m_max` (displayed/kept mode range) are intentionally decoupled, mirroring IDL's `bins` keyword.
4. **`plot_phase_map.py`** — reads all the `.npz` files back over the same grid and plots
   `|Bmn|` at one fixed `m_pol` as a 2D map over (phase_L, phase_U). `plot_flare_surfmn.py` plots a single
   `.npz`'s full (m, psiN) spectrum; `plot_flare_harmonic.py` overlays `|Bmn(psiN)|` on resonant surfaces
   from multiple `.npz` files for comparison.
5. **`flare_chirikov.py`** — pure post-processing on top of a `flare_surfmn.py` `.npz` (no FLARE/M3D-C1
   reload): computes the island half-width at each resonant surface,
   `width = (2/pi)*sqrt((area*Bmn[T]/|m*psiprime|)*|q/qprime|)`, and the Chirikov overlap parameter between
   adjacent resonant surfaces, `chi[j] = (width[j]+width[j+1])/2/|psiN_res[j+1]-psiN_res[j]|` — a direct port
   of IDL's `island_widths.pro`/`chirikov.pro`. Because it only reads the `.npz`, it is completely agnostic
   to how that `.npz`'s field was built: running it over `flare_phase_map.py`'s per-grid-point `.npz` outputs
   gives a Chirikov phase map "for free", inheriting the same multi-coil-set amplitude/phase superposition
   (done natively at the FLARE Fortran field-evaluation level) with zero new code. `plot_flare_chirikov.py`
   plots `chirikov` vs `psimid` (or `width_res` vs `psiN_res`), linear-axis line/marker plots matching IDL's
   `plot_bmn.pro` (`/chi`/`/width` modes) — unlike the surfmn contour plots.

**Cross-file invariants to preserve:**
- `phase_grid_size(n_tor, d_phase)` (in `phase_grid.py`) is the single source of truth for grid size and is
  called identically in `flare_model_gen.py`, `flare_phase_map.py`, and `plot_phase_map.py`. `d_phase` must
  evenly divide `360/n_tor` or the sweep falls short of a full period (the function warns but doesn't raise).
- Directory/file naming is a de facto protocol between stages:
  `dephase_IL_{phase_L:03d}_IU_{phase_U:03d}/` (or `CPL`/`CPU`) for model dirs, same stem `.npz` for phase-map
  outputs. Phase values are always `{:03d}`-zero-padded absolute values; sign is applied separately via
  `phase_signal` (default `[-1, 1]` for L/U respectively) at plot time, not baked into the filename.
- `validate_surfmn.py` is a standalone one-off validation script (not run in CI) that recomputes the same
  spectrum independently and diffs it against an IDL-generated reference NetCDF (`bmn_vac.nc`/`bmn_res.nc`).
  Keep it in sync if `flare_surfmn.py`'s normalization ever changes, since it's the only record of *why*
  the current formula is correct.
- `flare_surfmn.py`'s `.npz` also carries `area_res`, `qprime_res` (both per resonant surface, from
  `fluxsurf_params()`) and `psiprime` (a single scalar per model — `psiN` is linear in poloidal flux, so
  `d(poloidal flux)/dpsiN` is constant — fetched as `equi2d.poloidal_flux`, **not** divided by `2*pi`: IDL's
  own `island_widths.pro` divides by `2*pi` only because its `flux_pol` array is itself pre-multiplied by
  `2*pi` in `flux_coordinates.pro`, which FLARE's `poloidal_flux` is not; dividing again here would double
  that cancellation — confirmed empirically against a real IDL run, see `validate_chirikov.py`). These three
  keys are what `flare_chirikov.py` needs and are the only reason `flare_surfmn.py`'s `.npz` schema and
  `fluxsurf_params()`'s return signature grew past the original 6-tuple/8-key set.
- `validate_chirikov.py` is `flare_chirikov.py`'s counterpart to `validate_surfmn.py`: it diffs the ported
  island-width/Chirikov formula against real IDL `island_widths.pro` output (`chirikov.pro` itself cannot be
  called directly under modern IDL — its `if(width eq 0)` error-sentinel check errors out on a real,
  multi-element success array; reproduce its documented formula from `island_widths.pro`'s output instead,
  as the validation driver script does). Validated on `shot0009/n3/single_fluid/coil_sets_1kAt/IM_set_000`
  (n_tor=3): once the FLARE model's `timeslice` matches the M3D-C1 timeslice IDL's `plot_br.pro` actually
  read (its `slice` keyword defaults to the *last* available slice, not slice 0 — a real footgun when
  hand-building a `.bfield` for validation, since `flare_model_gen.py` always takes `timeslice` explicitly
  from the caller so this doesn't bite production phase-map runs), geometric quantities (`q`, `area`, `psiN`,
  `psiprime`) agree with IDL to <0.2%, and the final Chirikov parameter agrees to ~1% RMS (median 0.5%, max
  2.6%) across the RMP-relevant edge region.

### Other, independent tools

- **`m3dc1_units.py`** — self-contained (no `fpy`/`m3dc1`/`flare` dependency) SI ⟷ M3D-C1-normalized-unit
  converter for the `C1input` scalar namelist parameters (`eta_wall`, `eta_zone`, etc.) that have to be
  converted by hand — explicitly *not* for profile/data files, which M3D-C1 normalizes internally. All 17
  conversion factors are derived from `L0`, `B0`, `n0`, `ion_mass` (see the module docstring for the
  first-principles derivation and cross-check against `doc/units.tex`).
- **`plot_profiles.py`** — plots M3D-C1 equilibrium profiles (q, pressure, temperature, density, toroidal
  velocity, current density) directly from a `C1.h5` file via the `fpy`/`m3dc1` Python API
  (`flux_coordinates`, `eval_field`).
- **`plot_field_topdown.py`** — plots an M3D-C1 field on a constant-Z plane as a Cartesian top-down
  view of the tokamak (X = R cos φ, Y = R sin φ); thin wrapper around
  `m3dc1.get_field.get_field_vs_phi(..., cutz=Z_cut)` plus an optional first-wall crossing-radius
  overlay. **This file is a backup/dev copy of a routine meant to live in the fusion-io `m3dc1`
  package** (`~/software/fusion-io/build_shared/lib/m3dc1/`, same deploy pattern as
  `plot_profiles.py`) — if you change it here, the deployed copy must be updated too (and vice versa).
- **`plot_adapted_mesh.py`** — visualizes a PUMI-adapted M3D-C1 mesh straight from `apf::writeVtkFiles`
  VTK/XML output (`.pvtu` manifest + per-rank `.vtu`), hand-parsing the base64-encoded binary DataArrays —
  deliberately has *no* PUMI/VTK/meshio/pyvista dependency. Per-element face/zone coloring only works on
  VTK written by `m3dc1_mfmgen`, not on adapt-run output (see module docstring for why).
- **`plot_sizefieldParams.py`** — plots the normal/tangential mesh size-field profile
  `h(psi)` used by M3D-C1's `sizefieldParam` mesh adaptation model, from its 13-float parameter file.
- **`eval_footprint_area.py`** / **`plot_footprint.py`** / **`plot_manifold.py`** — read/plot divertor
  footprint and manifold data produced by the `maglib` code `fpgen` (plain-text column files: R, Z, phi,
  connection length, psi_min, toroidal turns). Vertical vs. horizontal divertor plate is auto-detected from
  whether column 0 (R) is constant across the file.
- **`notebooks/`** — exploratory Jupyter notebooks plus two supporting modules,
  `intersections.py` (curve-intersection helpers built on `shapely.geometry.LineString`) and `manifold.py`
  (a `Manifold` class wrapping R/Z/phi manifold data), imported from the notebooks rather than run as CLIs.

### `template/` — M3D-C1 scenario-tree generator (not a root CLI script)

Master copy of the kit that generates, validates, and (optionally) remotely runs the M3D-C1
RMP-response run-case trees (the `n<mode>/<fluid_model>/coil_sets_<kAt>kAt/<coil_set>_set_<index>/`
parameter scans, generated wherever `--output-dir` points — not necessarily in this repo). Contents:
`tools/generate_scenario.py` (stdlib-only CLI with `init` and `validate` subcommands),
`tools/remote_run.py` (stdlib-only CLI with `push`/`submit`/`status`/`pull` subcommands that drive a
generated tree's cases on an HPC cluster over SSH — push inputs, `sbatch` them, poll SLURM state, pull
solver output back — so the tree's canonical copy can live on a local analysis machine instead of the
cluster filesystem; tracks job IDs/link status in a generated `.remote_state.json`), `tools/templates/
*.template` (the per-case `C1input`, `run`, `make_links`, SLURM script bodies), `tools/reference_data/`
(fixed RMP coil geometry and per-coil-set current patterns), and `scenario.json` (the shot0009 config,
now including an optional `cluster` block consumed only by `remote_run.py`). `template/CLAUDE.md`
documents the generated tree's directory/file conventions and workflow — it is copied along with the
kit into each generated scenario tree, so keep it in sync with `generate_scenario.py`/`remote_run.py`
when editing any of them.

## Conventions across the codebase

- Every public function has a full NumPy-style docstring (Parameters/Returns) — match this style in new code.
- Physical quantities in array-returning functions are commented inline with shape/units/convention where
  not obvious from the docstring (following the user's general shape/units annotation convention).
- Plotting scripts share a common CLI shape: `figsize`, `dpi`, `cmap`/`colors`, and always end in
  `plt.show()`; several accept a `savefig` prefix and append a plot-type suffix (`_cl.png`, `_psi.png`, ...).
- No error-recovery abstractions — scripts raise `ValueError`/`FileNotFoundError` directly with a descriptive
  message and let argparse/the traceback surface it; don't add try/except layers beyond what a given script
  already does.
