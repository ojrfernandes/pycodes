#!/home/jfernandes/.venv/bin/python
import argparse
import os
import signal
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from phase_grid import phase_grid_size

_DEFAULT_FPGEN_BIN = os.path.expanduser("~/software/maglib/build/bin/fpgen")
# fpgen's wall reader (maglit/collider.cpp:load_shape) reads "R Z" pairs as a
# flat whitespace token stream with no header/comment handling -- a leading
# point-count line (as in machines_geo/input_geo/tcabr_first_wall.txt, used
# by FLARE) would misalign every point. plot_geo's copy has no such header.
_DEFAULT_FIRST_WALL = "/home/jfernandes/machines_geo/plot_geo/tcabr_first_wall.txt"


def _build_fpgen_input(source_L: str, source_M: str, source_U: str, timeslice: int,
                        phase_L: int, phase_U: int, amplitudes: list, phase_signal: list,
                        output_path: str, first_wall: str, manifold: int,
                        grid_R1: float, grid_Z1: float, grid_R2: float, grid_Z2: float,
                        nRZ: int, nPhi: int, n_tor: int, num_threads: int, max_turns: int,
                        h_init: float, h_min: float, h_max: float) -> str:
    """
    Build the fpgen INI-style input-file text for one (phase_L, phase_U) grid
    point, mirroring the section layout of a hand-built fpgen input file (see
    map_CPL_CPU/input_fpgen/*.txt) -- generated directly in code rather than
    via find/replace over a template file, the same approach flare_model_gen.py
    uses for .bfield/.boundary.
    """
    phase_0 = phase_signal[0] * phase_L
    phase_2 = phase_signal[1] * phase_U
    return (
        "################################ FPGEN #################################\n"
        "#\n"
        "#=============== I/O PARAMETERS\n"
        "#\n"
        f"        first_wall_path = {first_wall}\n"
        f"        output_path     = {output_path}\n"
        "#\n"
        "[M3DC1 SOURCE]\n"
        "#\n"
        "       nsources    = 3\n"
        "#\n"
        f"       source_0    = {source_L}\n"
        f"       timeslice_0 = {timeslice}\n"
        f"       phase_0     = {phase_0}\n"
        f"       amplitude_0 = {amplitudes[0]}\n"
        "#\n"
        f"       source_1    = {source_M}\n"
        f"       timeslice_1 = {timeslice}\n"
        "       phase_1     = 0.0\n"
        f"       amplitude_1 = {amplitudes[1]}\n"
        "#\n"
        f"       source_2    = {source_U}\n"
        f"       timeslice_2 = {timeslice}\n"
        f"       phase_2     = {phase_2}\n"
        f"       amplitude_2 = {amplitudes[2]}\n"
        "#\n"
        "#=============== MAPPING PARAMETERS\n"
        "#\n"
        f"        manifold = {manifold}\n"
        f"        grid_R1  = {grid_R1}\n"
        f"        grid_Z1  = {grid_Z1}\n"
        f"        grid_R2  = {grid_R2}\n"
        f"        grid_Z2  = {grid_Z2}\n"
        f"        nRZ      = {nRZ}\n"
        f"        nPhi     = {nPhi}\n"
        f"        ntor     = {n_tor}\n"
        "#\n"
        "#=============== INTEGRATOR PARAMETERS\n"
        "#\n"
        f"        num_threads = {num_threads}\n"
        f"        max_turns   = {max_turns}\n"
        f"        h_init = {h_init}\n"
        f"        h_min  = {h_min}\n"
        f"        h_max  = {h_max}\n"
        "#\n"
        "########################################################################\n"
    )


def _build_tasks(coils: str, save_to_path: str, n_tor: int, sets: str, timeslice: int, d_phase: int,
                  grid_R1: float, grid_Z1: float, grid_R2: float, grid_Z2: float, nRZ: int, nPhi: int,
                  manifold: int, amplitudes: list, num_threads: int, max_turns: int,
                  h_init: float, h_min: float, h_max: float, first_wall: str, fpgen_bin: str,
                  phase_signal: list, force: bool) -> list:
    """
    (phase_L, phase_U) task-tuple construction for one (sets, save_to_path) job.
    Each task tuple is fully self-contained (includes every fixed parameter the
    worker needs, not just the per-point ones) so it can be pickled and sent to
    a separate ProcessPoolExecutor worker, matching flare_phase_map.py's pattern.

    Idempotent: grid points whose output .dat already exists are skipped unless
    force=True, rather than an all-or-nothing FileExistsError -- fpgen runs are
    individually expensive, so a rerun of the same call just fills in what's
    still missing.
    """
    if coils not in ('I', 'CP'):
        raise ValueError("Invalid coils type. Options are 'I' or 'CP'.")
    prefix_L, prefix_M, prefix_U = ('IL', 'IM', 'IU') if coils == 'I' else ('CPL', 'CPM', 'CPU')

    if not sets.endswith('/'):
        sets += '/'
    if not save_to_path.endswith('/'):
        save_to_path += '/'

    os.makedirs(save_to_path, exist_ok=True)
    os.makedirs(os.path.join(save_to_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(save_to_path, "input_fpgen"), exist_ok=True)

    n_elements = phase_grid_size(n_tor, d_phase)

    tasks = []
    for i in range(n_elements):
        phase_L = i * d_phase
        for j in range(n_elements):
            phase_U = j * d_phase
            stem = f'dephase_{prefix_L}_{phase_L:03d}_{prefix_U}_{phase_U:03d}'
            out_path = os.path.join(save_to_path, f'{stem}_ftpt.dat')
            if os.path.exists(out_path) and not force:
                continue

            tasks.append((
                out_path,
                os.path.join(save_to_path, "input_fpgen", f'input_fp_{stem}.txt'),
                os.path.join(save_to_path, "logs", f'log_ftpt_{phase_L:03d}_{phase_U:03d}.txt'),
                f'{sets}{prefix_L}_set_000/C1.h5',
                f'{sets}{prefix_M}_set_000/C1.h5',
                f'{sets}{prefix_U}_set_000/C1.h5',
                timeslice, phase_L, phase_U, phase_signal, amplitudes, manifold,
                grid_R1, grid_Z1, grid_R2, grid_Z2, nRZ, nPhi, n_tor,
                num_threads, max_turns, h_init, h_min, h_max, first_wall, fpgen_bin,
            ))
    return tasks


def fpgen_phase_map(coils: str, save_to_path: str, n_tor: int, sets: str, timeslice: int,
                     grid_R1: float, grid_Z1: float, grid_R2: float, grid_Z2: float,
                     nRZ: int, nPhi: int, d_phase: int = 10, nprocs: int = 1,
                     manifold: int = 1, amplitudes: list = [1.0, 1.0, 1.0],
                     num_threads: int = 1, max_turns: int = 100,
                     h_init: float = 1e-4, h_min: float = 1e-6, h_max: float = 1e-2,
                     first_wall: str | None = None, fpgen_bin: str | None = None,
                     phase_signal: list = [-1, 1], force: bool = False) -> None:
    """
    Run fpgen over a grid of phase_L and phase_U values to generate footprint
    phase-map data, analogous to flare_phase_map.py but driving maglib's fpgen
    binary instead of flare_surfmn.py. One M3D-C1 run per coil (L/M/U) is
    reused across the whole grid, with the phase shift encoded directly in the
    generated fpgen input file's phase_i field (matching flare_model_gen.py's
    flare_phase=True strategy) rather than requiring a separate M3D-C1 run per
    phase.

    Parameters
    ----------
    coils : str
        Type of coils to use. Options are 'I' or 'CP'.
    save_to_path : str
        Directory to save the generated fpgen input files (input_fpgen/),
        per-point logs (logs/), and output footprint .dat files.
    n_tor : int
        Toroidal mode number.
    sets : str
        Path to the directory containing the M/L/U M3D-C1 coil-set C1.h5
        files (e.g. coil_sets_1kAt/), same convention as flare_model_gen.py.
        For a vacuum_field case (which has no coil_sets of its own), point
        this at the corresponding single_fluid/two_fluid coil_sets_1kAt
        directory and pass timeslice=0.
    timeslice : int
        M3D-C1 timeslice for every source: -1=equilibrium, 0=vacuum, 1=full
        response.
    grid_R1, grid_Z1, grid_R2, grid_Z2 : float
        (R, Z) endpoints (m) of the target-plate line segment to map.
    nRZ : int
        Number of grid points along the target surface.
    nPhi : int
        Number of toroidal starting angles sampled over [0, 2*pi/n_tor).
    d_phase : int
        Phase step in degrees. Default is 10.
    nprocs : int
        Number of parallel fpgen processes. Default is 1 (fpgen itself is
        already OpenMP-parallel via num_threads, so keep this at 1 unless
        num_threads is reduced accordingly to avoid oversubscription).
    manifold : int
        0 = unstable footprint, 1 = stable footprint. Default is 1.
    amplitudes : list of float
        Amplitudes for L, M, U sources respectively. Default is [60.0, 60.0, 60.0].
    num_threads : int
        OpenMP threads per fpgen run. Default is 13.
    max_turns : int
        Maximum toroidal turns per field line. Default is 100.
    h_init, h_min, h_max : float
        Adaptive step-size bounds for the field-line integrator.
    first_wall : str or None
        Path to the machine first-wall boundary file fpgen expects (no
        leading point-count header -- see module docstring note above). If
        None, defaults to machines_geo/plot_geo/tcabr_first_wall.txt.
    fpgen_bin : str or None
        Path to the fpgen binary. If None, defaults to
        ~/software/maglib/build/bin/fpgen.
    phase_signal : list of int
        Phase signal for L and U sets respectively. Default is [-1, 1].
    force : bool
        If True, recompute grid points whose output .dat already exists.
        Default is False (skip them).

    Returns
    -------
    None
        Writes fpgen input files, logs, and footprint .dat files under
        save_to_path.
    """
    if first_wall is None:
        first_wall = _DEFAULT_FIRST_WALL
    if fpgen_bin is None:
        fpgen_bin = _DEFAULT_FPGEN_BIN

    tasks = _build_tasks(
        coils, save_to_path, n_tor, sets, timeslice, d_phase,
        grid_R1, grid_Z1, grid_R2, grid_Z2, nRZ, nPhi,
        manifold, amplitudes, num_threads, max_turns, h_init, h_min, h_max,
        first_wall, fpgen_bin, phase_signal, force,
    )

    print(f"\n{len(tasks)} footprint grid point(s) to compute using {nprocs} process(es)...")
    if not tasks:
        print("Nothing to do (all outputs already exist; use force=True to recompute).")
        return

    try:
        with ProcessPoolExecutor(max_workers=nprocs) as executor:
            futures = [executor.submit(_run_one_footprint, t) for t in tasks]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"\nTask failed: {e}")

        print("All tasks completed.")
    except KeyboardInterrupt:
        print("Process interrupted by user. Terminating...")
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        raise


def _run_one_footprint(args) -> str:
    """
    Worker function for one (phase_L, phase_U) pair: write the fpgen input
    file, then run fpgen as a subprocess with stdout/stderr redirected to a
    log file, mirroring flare_phase_map.py's _process_phase_pair.
    """
    (out_path, input_path, log_path, source_L, source_M, source_U,
     timeslice, phase_L, phase_U, phase_signal, amplitudes, manifold,
     grid_R1, grid_Z1, grid_R2, grid_Z2, nRZ, nPhi, n_tor,
     num_threads, max_turns, h_init, h_min, h_max, first_wall, fpgen_bin) = args

    input_text = _build_fpgen_input(
        source_L, source_M, source_U, timeslice, phase_L, phase_U, amplitudes, phase_signal,
        out_path, first_wall, manifold, grid_R1, grid_Z1, grid_R2, grid_Z2, nRZ, nPhi, n_tor,
        num_threads, max_turns, h_init, h_min, h_max,
    )
    with open(input_path, "w") as f:
        f.write(input_text)

    print(f"\n-> Running fpgen for phases {phase_L}, {phase_U}")

    with open(log_path, "w") as f:
        proc = subprocess.Popen([fpgen_bin, input_path], stdout=f, stderr=f, preexec_fn=os.setsid)
        try:
            proc.wait()
        except KeyboardInterrupt:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            raise

    return f"Finished phases {phase_L}, {phase_U}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a footprint phase map by driving fpgen over a phase grid.")
    parser.add_argument("coils", type=str, help="Type of coils to use. Options are 'I' or 'CP'.")
    parser.add_argument("save_to_path", type=str, help="Directory to save fpgen inputs, logs, and footprint .dat files.")
    parser.add_argument("n_tor", type=int, help="Toroidal mode number.")
    parser.add_argument("sets", type=str, help="Path to the directory containing M/L/U M3D-C1 coil-set C1.h5 files.")
    parser.add_argument("timeslice", type=int, help="M3D-C1 timeslice: -1=equilibrium, 0=vacuum, 1=full response.")
    parser.add_argument("grid_R1", type=float, help="R coordinate of first endpoint of target surface (m).")
    parser.add_argument("grid_Z1", type=float, help="Z coordinate of first endpoint of target surface (m).")
    parser.add_argument("grid_R2", type=float, help="R coordinate of second endpoint of target surface (m).")
    parser.add_argument("grid_Z2", type=float, help="Z coordinate of second endpoint of target surface (m).")
    parser.add_argument("nRZ", type=int, help="Number of grid points along the target surface.")
    parser.add_argument("nPhi", type=int, help="Number of toroidal planes sampled.")
    parser.add_argument("--d_phase", type=int, default=10, help="Phase step in degrees. Default is 10.")
    parser.add_argument("--nprocs", type=int, default=1, help="Number of parallel fpgen processes. Default is 1.")
    parser.add_argument("--manifold", type=int, default=1, choices=[0, 1],
                         help="0 = unstable footprint, 1 = stable footprint. Default is 1.")
    parser.add_argument("--amplitudes", type=float, nargs=3, default=[60.0, 60.0, 60.0],
                         help="Amplitudes for L, M, U sources respectively. Default is [60.0, 60.0, 60.0].")
    parser.add_argument("--num_threads", type=int, default=13, help="OpenMP threads per fpgen run. Default is 13.")
    parser.add_argument("--max_turns", type=int, default=100, help="Maximum toroidal turns per field line. Default is 100.")
    parser.add_argument("--h_init", type=float, default=1e-4, help="Initial step-size for field-line integration.")
    parser.add_argument("--h_min", type=float, default=1e-6, help="Minimum step-size for field-line integration.")
    parser.add_argument("--h_max", type=float, default=1e-2, help="Maximum step-size for field-line integration.")
    parser.add_argument("--first_wall", type=str, default=None,
                         help="Path to the first-wall file fpgen expects. Default is machines_geo/plot_geo/tcabr_first_wall.txt.")
    parser.add_argument("--fpgen_bin", type=str, default=None,
                         help="Path to the fpgen binary. Default is ~/software/maglib/build/bin/fpgen.")
    parser.add_argument("--phase_signal", type=int, nargs=2, default=[-1, 1],
                         help="Phase signal for L and U sets respectively. Default is [-1, 1].")
    parser.add_argument("--force", action="store_true", help="Recompute grid points whose output .dat already exists.")

    args = parser.parse_args()

    fpgen_phase_map(
        args.coils,
        args.save_to_path,
        args.n_tor,
        args.sets,
        args.timeslice,
        args.grid_R1,
        args.grid_Z1,
        args.grid_R2,
        args.grid_Z2,
        args.nRZ,
        args.nPhi,
        d_phase=args.d_phase,
        nprocs=args.nprocs,
        manifold=args.manifold,
        amplitudes=args.amplitudes,
        num_threads=args.num_threads,
        max_turns=args.max_turns,
        h_init=args.h_init,
        h_min=args.h_min,
        h_max=args.h_max,
        first_wall=args.first_wall,
        fpgen_bin=args.fpgen_bin,
        phase_signal=args.phase_signal,
        force=args.force,
    )
