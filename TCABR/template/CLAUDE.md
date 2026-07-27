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

Each case can still be launched individually this way. For driving the whole `4 (n) × 2 (fluid) ×
6 (coil set) = 48`-case scan at once from a local analysis machine over SSH, see `tools/remote_run.py`
below — it wraps exactly this `make_links` → `sbatch batch_script.response` → (wait) → collect-output
cycle, batched across cases.

## `tools/` — scenario generator, validator, and remote execution

This tree (and any future scenario tree for a different shot) is produced and checked by
`tools/generate_scenario.py`, a stdlib-only Python CLI:

```bash
# (Re)generate a scenario from its config. --config defaults to ./scenario.json, --output-dir to
# the current directory, so `init` with no flags works when run from a tree's root. Existing case
# directories are left alone unless --force is given, in which case only the 8 known input files
# are re-rendered — run.log/*.h5/symlinks and any other solver output are never touched.
python3 tools/generate_scenario.py init [--config scenario.json] [--output-dir <path>] [--force]

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

### `tools/remote_run.py` — driving the cluster over SSH from a local scenario tree

Lets the generated tree live on a local analysis machine as the source of truth, and pushes/submits/
monitors/pulls/cleans cases on the HPC cluster over SSH (key-based, non-interactive — no MFA/jump-host
handling built in) instead of the older manual workflow of submitting on the cluster and tarring
results back by hand. Five subcommands, each mirroring `validate`'s `--path` (default `.`) plus a
`--cases` selector (comma-separated `fnmatch` globs against case relpaths, e.g.
`n1/*/coil_sets_1kAt/CPL_set_000`; omitted = all 48 cases — batching is purely a connection-count
optimization for whatever subset is selected, not a requirement to operate on the whole tree at once;
a single case is just a batch of size one):

```bash
# Push the 8 known input files for the selected cases to the cluster, and run ./make_links
# remotely for any case not yet linked there (ln -s isn't idempotent, so this is tracked in
# .remote_state.json rather than re-run blindly). Never touches remote-side solver output.
python3 tools/remote_run.py push --path <path> [--cases <selector>] [--force-relink] [--dry-run]

# sbatch batch_script.response remotely for the selected cases, recording the returned SLURM
# job ID per case. Skips (reports, doesn't error) any case with a non-terminal job already
# tracked, unless --force.
python3 tools/remote_run.py submit --path <path> [--cases <selector>] [--force] [--dry-run]

# Query sacct/squeue for the tracked job IDs and print each case's SLURM state plus a summary
# count (completed/failed/in flight/not submitted).
python3 tools/remote_run.py status --path <path> [--cases <selector>]

# rsync back C1stdout/*.h5/C1ke/normcurv/*out/profiles* for cases whose last known state is
# COMPLETED (or any tracked case with --all), into the matching local case directory.
python3 tools/remote_run.py pull --path <path> [--cases <selector>] [--all] [--dry-run]

# Run ./rmall remotely (the same per-case cleanup script used locally) to remove solver output
# on the cluster, freeing scratch space. Default: only cases confirmed COMPLETED and already
# pulled locally (--force bypasses that, e.g. to clean a FAILED case with nothing worth keeping).
# Always refuses a case whose job is still non-terminal (RUNNING/PENDING/...), even with --force,
# since rmall would delete output files an active job is still writing. Inputs/symlinks are left
# in place, so a cleaned case is still ready to resubmit without re-push/re-link.
python3 tools/remote_run.py clean --path <path> [--cases <selector>] [--force] [--dry-run]
```

This requires a `"cluster"` block in `scenario.json`, in addition to the keys `generate_scenario.py`
itself needs (which ignores `cluster` entirely):

```json
"cluster": {
  "host": "<ssh-alias-or-user@hostname, key-based/non-interactive>",
  "remote_base_dir": "/scratch/.../scenario_runs",
  "user": "<remote-username, used for readable status output only>"
}
```

State — SLURM job ID, link status, last-known job state per case — is cached in
`<scenario-root>/.remote_state.json` (one file for the whole tree). It's a local cache, not source
data: safe to inspect, not meant to be hand-edited, and deleting it while jobs are in flight loses the
case→job-ID mapping (`status`/`pull` will no longer know what to check or fetch), though the jobs
themselves keep running on the cluster unaffected. It also records which `cluster.host`/
`remote_base_dir` it was populated against, and refuses to reuse cached job IDs if `scenario.json`'s
`cluster` block later points somewhere else.

`push`, `submit`, and `clean` batch every selected case into a single ssh round trip (rather than one
connection per case) since ssh handshake overhead is non-negligible across 48 cases; `pull` is the one
exception, running one rsync per eligible case, since rsync's glob-matching `--include`/`--exclude`
filters can't be cheaply scoped to an arbitrary subset of case directories in one call.
