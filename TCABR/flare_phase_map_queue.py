#!/home/jfernandes/.venv/bin/python
"""Run several flare_phase_map.py jobs (e.g. different fluid models/coil families,
or different n_tor) sharing ONE global worker-process budget, instead of each job
spawning its own independent nprocs-sized pool -- running N such jobs each with
--nprocs P concurrently would use up to N*P processes at once, not P.

All (phase_L, phase_U) tasks from every job are combined into a single task list
and submitted to one ProcessPoolExecutor, so at most `nprocs` FLARE subprocesses
run at any time across the whole queue, regardless of how the tasks are split
across jobs. This replaces manually supervising flare_phase_map.py invocations
one at a time to keep total concurrency bounded.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

from flare_phase_map import _build_tasks, _process_phase_pair


def flare_phase_map_queue(jobs: list, nprocs: int) -> None:
    """
    Parameters
    ----------
    jobs : list of dict
        Each dict has keys: model_path, save_to_path, n_tor, m_max, and
        optionally d_phase (default 10), n_pol (default 400), force (default False)
        -- same meaning as flare_phase_map()'s arguments.
    nprocs : int
        Total worker processes shared across every job's tasks combined.

    Returns
    -------
    None
        Saves surfmn data in .npz files in each job's save_to_path, as flare_phase_map() would.
    """
    all_tasks = []
    for job in jobs:
        tasks = _build_tasks(
            job["model_path"], job["save_to_path"], job["n_tor"], job["m_max"],
            job.get("d_phase", 10), job.get("n_pol", 400), job.get("force", False),
        )
        print(f"Queued {len(tasks)} tasks from {job['save_to_path']}")
        all_tasks.extend(tasks)

    print(f"\nLaunching {len(all_tasks)} total evaluations across {len(jobs)} job(s) "
          f"using a shared pool of {nprocs} processes...")

    try:
        with ProcessPoolExecutor(max_workers=nprocs) as executor:
            futures = [executor.submit(_process_phase_pair, t) for t in all_tasks]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple flare_phase_map jobs sharing one worker-process budget.")
    parser.add_argument("jobs_json", type=str,
                         help="Path to a JSON file listing jobs: "
                              '[{"model_path": ..., "save_to_path": ..., "n_tor": ..., "m_max": ..., '
                              '"d_phase": 10, "n_pol": 400, "force": false}, ...]')
    parser.add_argument("--nprocs", type=int, default=13,
                         help="Total worker processes shared across all jobs. Default is 13.")
    args = parser.parse_args()

    with open(args.jobs_json) as f:
        jobs = json.load(f)

    flare_phase_map_queue(jobs, args.nprocs)
