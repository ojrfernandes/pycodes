#!/home/jfernandes/.venv/bin/python
"""Backfill the 3 Chirikov-support keys (area_res, qprime_res, psiprime) into
flare_surfmn.py .npz files that predate the Chirikov extension, without
rerunning the expensive Fourier-transform computation (db_matrix/db_res are
untouched and reused as-is).

qprime_res is rebuilt purely from the already-saved (psiN_values, q_vals)
merged grid (same forward CubicSpline used in flare_surfmn.py's
fluxsurf_params()) -- no FLARE reload needed for that one. area_res and
psiprime were never saved by the old flare_surfmn.py, so they do need a brief
FLARE reload (fluxsurf2d_parameters at the already-known psiN_res points, and
equi2d.poloidal_flux) -- run in a subprocess per file, mirroring
flare_phase_map.py's per-task isolation (the FLARE Fortran/f2py state isn't
safely reusable across repeated model.load/model.free cycles in one process).
"""
import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REQUIRED_KEYS = ("area_res", "qprime_res", "psiprime")


def backfill_chirikov_keys(npz_path: str, flare_model_dir: str) -> None:
    """
    Add area_res/qprime_res/psiprime to an existing flare_surfmn.py .npz file in place.

    Parameters
    ----------
    npz_path : str
        Path to the existing flare_surfmn.py .npz output.
    flare_model_dir : str
        Path to the FLARE model directory (.bfield/.boundary) that produced it.

    Returns
    -------
    None
        Overwrites npz_path with the original keys plus the 3 new ones.
    """
    import numpy as np
    from scipy.interpolate import CubicSpline
    from flare import model
    from flare.analysis import equi2d, equi2d_rzarray, fluxsurf2d_parameters

    with np.load(npz_path) as f:
        data = dict(f.items())

    if all(k in data for k in REQUIRED_KEYS):
        print(f"{npz_path}: already has Chirikov keys, skipping.")
        return

    psiN_values, q_vals, psiN_res = data["psiN_values"], data["q_vals"], data["psiN_res"]
    idx_forward = np.argsort(psiN_values)
    qprime_spline = CubicSpline(psiN_values[idx_forward], q_vals[idx_forward]).derivative()
    qprime_res = qprime_spline(psiN_res)

    model.load(flare_model_dir)
    try:
        R_vals, Z_vals = equi2d_rzarray(psiN_res, 0)
        area_res = np.array([fluxsurf2d_parameters((R, Z))[2] for R, Z in zip(R_vals, Z_vals)])
        psiprime = equi2d.poloidal_flux
    finally:
        model.free()

    data["area_res"] = area_res
    data["qprime_res"] = qprime_res
    data["psiprime"] = psiprime
    np.savez(npz_path, **data)
    print(f"Backfilled {npz_path}")


def _backfill_one(args) -> str:
    npz_path, flare_model_dir = args
    cmd = [
        sys.executable, "-c",
        f"from flare_surfmn_backfill import backfill_chirikov_keys; "
        f"backfill_chirikov_keys(r'{npz_path}', r'{flare_model_dir}')"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return f"FAILED {npz_path}: {proc.stderr[-500:]}"
    return f"OK {npz_path}"


def find_backfill_tasks(npz_root: str, database_root: str) -> list:
    """
    Walk npz_root for dephase_*.npz files and pair each with its corresponding
    FLARE model directory under database_root (same relative path, with the
    .npz stem as an extra subdirectory level -- the .bfield/.boundary live in
    a subdirectory named after the dephase stem, while the .npz sits flat in
    map_*/).

    Parameters
    ----------
    npz_root : str
        Root to search for dephase_*.npz files (e.g. ~/m3dc1_data/shot0009/n3).
    database_root : str
        Root of the corresponding FLARE model directories
        (e.g. ~/DATABASE/flare/TCABR/shot0009/n3).

    Returns
    -------
    list of tuple
        (npz_path, flare_model_dir) pairs.
    """
    npz_root = Path(npz_root)
    database_root = Path(database_root)
    tasks = []
    for npz_path in sorted(npz_root.glob("**/dephase_*.npz")):
        rel = npz_path.relative_to(npz_root)
        model_dir = database_root / rel.parent / npz_path.stem
        if not model_dir.exists():
            print(f"Warning: no model directory found for {npz_path} (expected {model_dir}), skipping.")
            continue
        tasks.append((str(npz_path), str(model_dir)))
    return tasks


def flare_surfmn_backfill(npz_root: str, database_root: str, nprocs: int = 4) -> None:
    """
    Backfill Chirikov-support keys into every dephase_*.npz found under npz_root.

    Parameters
    ----------
    npz_root : str
        Root to search for dephase_*.npz files.
    database_root : str
        Root of the corresponding FLARE model directories.
    nprocs : int
        Parallel worker processes. Default is 4 (each backfill is a few
        seconds, not minutes, so this doesn't need as large a budget as
        flare_phase_map_queue.py).

    Returns
    -------
    None
    """
    tasks = find_backfill_tasks(npz_root, database_root)
    print(f"Found {len(tasks)} .npz files to backfill under {npz_root}.")

    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        futures = [executor.submit(_backfill_one, t) for t in tasks]
        for f in as_completed(futures):
            print(f.result())

    print("All backfill tasks completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Chirikov-support keys into existing flare_surfmn.py .npz files.")
    parser.add_argument("npz_root", type=str, help="Root directory to search for dephase_*.npz files.")
    parser.add_argument("database_root", type=str, help="Root of the corresponding FLARE model directories.")
    parser.add_argument("--nprocs", type=int, default=4, help="Parallel worker processes. Default is 4.")
    args = parser.parse_args()

    flare_surfmn_backfill(args.npz_root, args.database_root, args.nprocs)
