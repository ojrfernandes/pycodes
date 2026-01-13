#!/home/jfernandes/.venv/bin/python
import os
import argparse
import sys
import subprocess
import signal  
from concurrent.futures import ProcessPoolExecutor, as_completed


def flare_phase_map(model_path, save_to_path, ntor, m_max, d_phase=10, nprocs=1):
    """
    Run flare_surfmn over a grid of phase_L and phase_U values to generate a phase map.

    Parameters
    ----------
    model_path : str
        Path to the directory containing flare model files.
    save_to_path : str
        Directory to save the output .npz files.
    ntor : int
        Toroidal mode number.
    m_max : int
        Maximum mapped poloidal mode number.
    d_phase : int
        Phase step in degrees. Default is 10.
    nprocs : int
        Number of parallel processes. Default is 1.

    Returns
    -------
    None
        Saves surfmn data in .npz files in the specified directory.
    """

    n_elements = int((360 / ntor / d_phase) + 1)

    if not os.path.exists(save_to_path):
        os.makedirs(save_to_path)
    if not os.path.exists(os.path.join(save_to_path, "logs")):
        os.makedirs(os.path.join(save_to_path, "logs"))

    # Prepare task list
    tasks = [
        (model_path, save_to_path, ntor, m_max, i * d_phase, j * d_phase)
        for i in range(n_elements)
        for j in range(n_elements)
    ]

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
    model_path, save_to_path, ntor, m_max, phase_L, phase_U = args

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
        f"flare_surfmn(r'{flare_model}', {ntor}, {m_max}, r'{filename}')"
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
    parser.add_argument("ntor", type=int, help="Toroidal mode number.")
    parser.add_argument("m_max", type=int, help="Maximum poloidal mode number.")
    parser.add_argument("--d_phase", type=int, default=10, help="Phase step in degrees. Default is 10.")
    parser.add_argument("--nprocs", type=int, default=1, help="Number of parallel processes. Default is 1.")

    args = parser.parse_args()

    flare_phase_map(
        args.model_path,
        args.save_to_path,
        args.ntor,
        args.m_max,
        args.d_phase,
        args.nprocs
    )