#!/home/jfernandes/.venv/bin/python
import os
import glob
import argparse
import sys
import subprocess
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed

from phase_grid import phase_grid_size


def _build_tasks(model_path: str, save_to_path: str, n_tor: int, m_max: int, d_phase: int, n_pol: int,
                  force: bool) -> list:
    """
    Existing-output guard, save_to_path/logs setup, and (phase_L, phase_U) task-tuple
    construction for one (model_path, save_to_path) job -- shared by flare_phase_map()
    and flare_phase_map_queue.py, which combines tasks from several such jobs into one
    worker pool.

    Returns
    -------
    list of tuple
        (model_path, save_to_path, n_tor, m_max, phase_L, phase_U, n_pol) per grid point.
    """
    n_elements = phase_grid_size(n_tor, d_phase)

    existing = glob.glob(os.path.join(save_to_path, "dephase_*.npz"))
    if existing and not force:
        raise FileExistsError(
            f"{len(existing)} dephase_*.npz file(s) already exist in '{save_to_path}' "
            "from a previous run. Use force=True (or --force on the CLI) to overwrite them, "
            "or choose a different save_to_path."
        )
    elif existing and force:
        print(f"Warning: overwriting {len(existing)} existing .npz file(s) in '{save_to_path}'.")

    if not os.path.exists(save_to_path):
        os.makedirs(save_to_path)
    if not os.path.exists(os.path.join(save_to_path, "logs")):
        os.makedirs(os.path.join(save_to_path, "logs"))

    return [
        (model_path, save_to_path, n_tor, m_max, i * d_phase, j * d_phase, n_pol)
        for i in range(n_elements)
        for j in range(n_elements)
    ]


def flare_phase_map(model_path: str, save_to_path: str, n_tor: int, m_max: int, d_phase: int=10, nprocs: int=1,
                     n_pol: int=400, force: bool=False) -> None:
    """
    Run flare_surfmn over a grid of phase_L and phase_U values to generate a phase map.

    Parameters
    ----------
    model_path : str
        Path to the directory containing flare model files.
    save_to_path : str
        Directory to save the output .npz files.
    n_tor : int
        Toroidal mode number.
    m_max : int
        Maximum mapped poloidal mode number.
    d_phase : int
        Phase step in degrees. Default is 10.
    nprocs : int
        Number of parallel processes. Default is 1.
    n_pol : int
        Poloidal Fourier transform resolution passed through to flare_surfmn,
        decoupled from m_max. Default is 400 (see flare_surfmn.py).
    force : bool
        If False (default) and save_to_path already contains dephase_*.npz files
        from a previous run, abort before launching any task. If True, proceed
        and overwrite them.

    Returns
    -------
    None
        Saves surfmn data in .npz files in the specified directory.
    """

    tasks = _build_tasks(model_path, save_to_path, n_tor, m_max, d_phase, n_pol, force)

    print(f"\nLaunching {len(tasks)} evaluations using {nprocs} processes...")

    try:
        with ProcessPoolExecutor(max_workers=nprocs) as executor:
            futures = [executor.submit(_process_phase_pair, t) for t in tasks]
            for f in as_completed(futures):
                try:
                    f.result()  # trigger any raised exceptions
                except Exception as e:
                    print(f" \nTask failed: {e}")

        print("All tasks completed.")
    except KeyboardInterrupt:
        print("Process interrupted by user. Terminating...")
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        raise



def _process_phase_pair(args):
    """
    Worker function for one (phase_L, phase_U) pair.
    
    """
    model_path, save_to_path, n_tor, m_max, phase_L, phase_U, n_pol = args

    file_I = os.path.join(model_path, f'dephase_IL_{int(phase_L):03d}_IU_{int(phase_U):03d}')
    file_CP = os.path.join(model_path, f'dephase_CPL_{int(phase_L):03d}_CPU_{int(phase_U):03d}')  

    if os.path.exists(file_I):
        flare_model = file_I
        filename = os.path.join(save_to_path, f'dephase_IL_{int(phase_L):03d}_IU_{int(phase_U):03d}.npz')
    elif os.path.exists(file_CP):
        flare_model = file_CP
        filename = os.path.join(save_to_path, f'dephase_CPL_{int(phase_L):03d}_CPU_{int(phase_U):03d}.npz')
    else:
        print(f"No valid file found for phases {phase_L}, {phase_U}")
        return

    print(f"\n-> Evaluating model {flare_model}")

    log_file = os.path.join(save_to_path, f"logs/log_{int(phase_L):03d}_{int(phase_U):03d}.txt")

    # Run flare_surfmn in a subprocess so Fortran prints log into a file
    cmd = [
        sys.executable, "-c",
        f"from flare_surfmn import flare_surfmn; "
        f"flare_surfmn(r'{flare_model}', {n_tor}, {m_max}, r'{filename}', n_pol={n_pol})"
    ]

    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, preexec_fn=os.setsid)
        try:
            proc.wait()
        except KeyboardInterrupt:
            # Kill the whole process group if interrupted
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            raise

    return f"Finished {os.path.basename(flare_model)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate phase map using flare_surfmn.")
    parser.add_argument("model_path", type=str, help="Path to the directory containing flare model files.")
    parser.add_argument("save_to_path", type=str, help="Directory to save the output .npz files.")
    parser.add_argument("n_tor", type=int, help="Toroidal mode number.")
    parser.add_argument("m_max", type=int, help="Maximum poloidal mode number.")
    parser.add_argument("--d_phase", type=int, default=10, help="Phase step in degrees. Default is 10.")
    parser.add_argument("--nprocs", type=int, default=1, help="Number of parallel processes. Default is 1.")
    parser.add_argument("--n_pol", type=int, default=400,
                         help="Poloidal Fourier transform resolution passed to flare_surfmn (default: 400).")
    parser.add_argument("--force", action="store_true",
                         help="Overwrite existing .npz output files in save_to_path.")

    args = parser.parse_args()

    flare_phase_map(
        args.model_path,
        args.save_to_path,
        args.n_tor,
        args.m_max,
        args.d_phase,
        args.nprocs,
        args.n_pol,
        args.force
    )