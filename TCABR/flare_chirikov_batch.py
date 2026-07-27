#!/home/jfernandes/.venv/bin/python
"""Batch-run flare_chirikov.py over every flare_surfmn.py .npz output found under
one or more phase-map roots (e.g. ~/m3dc1_data/shot0009/{n3,n6,n9}), producing a
sibling <stem>_chirikov.npz next to each surfmn .npz -- the same naming convention
as running flare_chirikov.py by hand (dephase_CPL_060_CPU_060.npz ->
dephase_CPL_060_CPU_060_chirikov.npz, same directory).

Unlike flare_phase_map.py/flare_surfmn_backfill.py, flare_chirikov.py touches no
FLARE/Fortran state -- it's pure np.load/numpy/np.savez -- so tasks run directly
in worker processes, no per-task subprocess isolation needed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from flare_chirikov import flare_chirikov


def find_chirikov_tasks(roots: list) -> list:
    """
    Walk each root for surfmn dephase_*.npz files and pair each with its
    <stem>_chirikov.npz output path, skipping files that are themselves
    already a Chirikov output and pairs whose output already exists.

    Parameters
    ----------
    roots : list of str
        Root directories to search recursively for dephase_*.npz surfmn files.

    Returns
    -------
    list of tuple
        (surfmn_npz_path, output_path) pairs still needing computation.
    """
    tasks = []
    for root in roots:
        for npz_path in sorted(Path(root).glob("**/dephase_*.npz")):
            if npz_path.stem.endswith("_chirikov"):
                continue
            out_path = npz_path.with_name(npz_path.stem + "_chirikov.npz")
            if out_path.exists():
                continue
            tasks.append((str(npz_path), str(out_path)))
    return tasks


def _run_one(args) -> str:
    surfmn_npz, filename = args
    try:
        flare_chirikov(surfmn_npz, filename)
        return f"OK {filename}"
    except Exception as e:
        return f"FAILED {surfmn_npz}: {e}"


def flare_chirikov_batch(roots: list, nprocs: int = 8) -> None:
    """
    Run flare_chirikov.py over every not-yet-processed surfmn .npz found under roots.

    Parameters
    ----------
    roots : list of str
        Root directories to search recursively for dephase_*.npz surfmn files.
    nprocs : int
        Parallel worker processes. Default is 8 (each task is cheap -- no FLARE
        reload, just small-array numpy -- so this can run with more workers than
        flare_surfmn_backfill.py's default).

    Returns
    -------
    None
        Writes a <stem>_chirikov.npz next to each processed surfmn .npz.
    """
    tasks = find_chirikov_tasks(roots)
    print(f"Found {len(tasks)} surfmn .npz file(s) needing Chirikov output.")
    if not tasks:
        return

    n_failed = 0
    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        futures = [executor.submit(_run_one, t) for t in tasks]
        for f in as_completed(futures):
            result = f.result()
            if result.startswith("FAILED"):
                n_failed += 1
                print(result)

    print(f"All Chirikov batch tasks completed. {len(tasks) - n_failed}/{len(tasks)} succeeded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-run flare_chirikov.py over all surfmn .npz files under one or more roots."
    )
    parser.add_argument("roots", type=str, nargs="+",
                         help="Root directories to search for dephase_*.npz surfmn files.")
    parser.add_argument("--nprocs", type=int, default=8, help="Parallel worker processes. Default is 8.")
    args = parser.parse_args()

    flare_chirikov_batch(args.roots, args.nprocs)
