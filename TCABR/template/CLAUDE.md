# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is **not a software codebase** — it is a parameter-scan directory tree of run cases for
[M3D-C1](https://m3dc1.pppl.gov/), an extended-MHD plasma simulation code, configured to compute the
**linear plasma response to RMP (Resonant Magnetic Perturbation) coils** on the TCABR tokamak
(equilibrium `shot0009`). There is no source code, build system, or test suite here — only input decks,
helper shell scripts, and (for some cases) run output.

Each leaf directory is one independent M3D-C1 case: a self-contained set of input files plus scripts to
link in shared equilibrium/mesh data, execute the solver, and clean up outputs.

## Directory naming convention

```
n{1,3,6,9}/{single_fluid,two_fluid}/coil_sets_1kAt/{CPL,CPM,CPU,IL,IM,IU}_set_000/
```

- **`n1` / `n3` / `n6` / `n9`** — toroidal mode number of the RMP response (`ntor` in `C1input`). Each
  top-level folder is a full copy of the scan for that mode number.
- **`single_fluid` / `two_fluid`** — physics model. The `C1input` files are identical between the two
  except for `ipres` (electron pressure equation: 0 vs 1) and `db_fac` (ion skin depth / two-fluid term:
  0.0 vs 1.0).
- **`coil_sets_1kAt`** — all cases apply 1000 A-turns to a pair of RMP coils.
- **`{CPL,CPM,CPU,IL,IM,IU}_set_000`** — which RMP coil pair is energized (12 coils total, defined once
  in `rmp_coil.dat`, identical across all six sets). Only `rmp_current.dat` differs between sets — it's a
  12-line (real, imag) current list that is all-zero except a `+1 / -1` pair marking the driven coils for
  that set. `CP` = coil on the high-field side of the tokamak, `I` = low-field side; `L`/`M`/`U` =
  Lower/Middle/Upper poloidal position.
  The `_000` suffix is the `set_index` field in `scenario.json` — room for additional
  current-amplitude/phase variants (`_001`, ...) not present yet.

## Per-case files

- **`C1input`** — M3D-C1 Fortran namelist (`&inputnl`). Controls I/O, numerics, physics model, geometry,
  equilibrium reconstruction (reads `geqdsk`), boundary/zone resistivities, mesh filenames, and RMP
  (`irmp = 1`). This is a linear (`linear = 1`), toroidal (`itor = 1`), equilibrium-subtracted
  (`eqsubtract = 1`) run reading a pre-adapted unstructured mesh.
- **`rmp_coil.dat`** — (R, Z, dR, dZ) geometry of the 12 RMP coils. Same in every case.
- **`rmp_current.dat`** — per-coil (real, imag) drive current for *this* case; distinguishes the six coil
  sets.
- **`make_links`** — creates symlinks in the case directory to the shared equilibrium/mesh inputs under
  `/scratch/ntm/jose.fernandes/shot0009/adapt/adapt_1-3/`: adapted mesh parts (`ts0-adaptedN.smb`), mesh
  model (`tcabr-5-region.txt`), equilibrium coil/current (`coil.dat`, `current.dat`), `geqdsk`, and
  kinetic profiles (`profile_ne`, `profile_omega`, `profile_te`). **`single_fluid` cases have not had
  `make_links` run yet** (no symlinks present); most `two_fluid` cases have. Run `./make_links` before
  attempting to execute a case that's missing them.
- **`clear_links`** — removes the symlinks created by `make_links`.
- **`rmall`** — removes M3D-C1 output artifacts (`C1ke`, `normcurv`, `*out`, `profiles*`, `*.h5`) to reset
  a case before rerunning.
- **`run`** — direct execution: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 mpirun -np 12 <path-to-m3dc1_2d_complex> 2>&1 | tee run.log`.
- **`batch_script.response`** — SLURM batch script (`sequana_cpu` partition, 48 tasks) for cluster
  submission via `sbatch batch_script.response`; runs `m3dc1_2d_complex_2025` with MUMPS as the linear
  solver and logs to `C1stdout`.
- **`run.log`** (present only where a case has already been executed, e.g. some `n6`/`n9` `two_fluid`
  cases) — solver stdout, useful for checking convergence/timing of a completed run.

## Common workflow for a single case

```bash
cd n<mode>/<fluid_model>/coil_sets_1kAt/<COILSET>_set_000
./make_links      # link in shared mesh/equilibrium/profile data (skip if already linked)
./run             # or: sbatch batch_script.response   (on the cluster)
./rmall           # clean generated output before a rerun
./clear_links     # remove the shared-data symlinks when done
```

Each of the `4 (n) × 2 (fluid) × 6 (coil set) = 48` cases is still launched individually — there's no
aggregate script to submit the whole scan at once, only to generate/validate the case directories
(see below).

## `tools/` — scenario generator and validator

This tree (and any future scenario tree for a different shot) is produced and checked by
`tools/generate_scenario.py`, a stdlib-only Python CLI:

```bash
# (Re)generate a scenario from its config. Existing case directories are left alone unless --force
# is given, in which case only the 8 known input files are re-rendered — run.log/*.h5/symlinks and
# any other solver output are never touched.
python3 tools/generate_scenario.py init --config scenario.json --output-dir <path> [--force]

# Check a scenario tree for consistency: unique SLURM job names, rmp_coil.dat/rmp_current.dat
# matching the reference geometry/current patterns, C1input's ntor/ipres/db_fac matching their
# directory, no unexpected drift between cases, no broken symlinks, no missing files.
python3 tools/generate_scenario.py validate --path <path>
```

- **`scenario.json`** (repo root) — the config that produced this tree: shot number,
  `shared_data_dir` (the shared mesh/equilibrium source for `make_links`), which `n_values` /
  `fluid_models` / `current_levels_kAt` / `coil_sets` to generate, the M3D-C1 executable
  path/version, and the local (`run`) and SLURM (`batch_script.response`) execution settings.
  Axis lists are optional in a hand-written config — omitted ones default to the full historical
  matrix (n=1,3,6,9 × both fluid models × 1 kAt × all 6 coil sets).
- **`tools/templates/*.template`** — the `C1input`, `batch_script.response`, `run`, and
  `make_links` bodies with the scenario/case-specific values as `{placeholder}` fields.
  `clear_links` isn't a separate template — it's derived at generation time from the same
  `ln -s ... .` lines in `make_links.template`, so the two can't drift apart.
- **`tools/reference_data/`** — the RMP coil geometry (`rmp_coil.dat`) and the six per-coil-set
  current patterns (`rmp_current/{CPL,CPM,CPU,IL,IM,IU}.dat`). These are fixed properties of the
  physical RMP coil array, not scenario config — every scenario, regardless of shot number, uses
  the same bundled files.
- SLURM job names are generated as `n<mode>_<1f|2f>_<coilset>` (e.g. `n1_2f_CPL`) so concurrent
  jobs are distinguishable in `squeue`.

To start a new scenario (e.g. a different shot number), write a new `scenario.json` (at minimum
`shot`, `shared_data_dir`, `executable`, `run`, `slurm`) and run `init` with `--output-dir` pointed
at wherever that scenario should live — there's no fixed path convention baked into the tool.
