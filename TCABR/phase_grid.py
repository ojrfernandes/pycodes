#!/home/jfernandes/.venv/bin/python
import numpy as np


def phase_grid_size(n_tor: int, d_phase: int) -> int:
    """
    Number of phase steps (inclusive of both endpoints) covering one
    elementary period 360/n_tor degrees, in increments of d_phase.

    Warns if d_phase does not evenly divide 360/n_tor: the sweep then falls
    short of a full period, and any full-360-degree tiling built from it
    (e.g. plot_phase_map.py's fullspace option) will show a phase
    discontinuity at the seam.
    """
    period = 360.0 / n_tor
    n_steps = period / d_phase
    if not np.isclose(n_steps, round(n_steps)):
        print(f"Warning: d_phase={d_phase} does not evenly divide 360/n_tor={period:.3f} deg "
              f"(360/n_tor/d_phase = {n_steps:.4f}). The phase sweep will fall short of a full "
              "period, and full-360-degree tiling will show a discontinuity at the seam.")
    return int(n_steps) + 1
