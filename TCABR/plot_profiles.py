#!/home/jfernandes/.venv/bin/python
import argparse
from collections.abc import Callable
from dataclasses import dataclass

import fpy
import numpy as np
import matplotlib.pyplot as plt
from m3dc1.flux_average import flux_average

# Plotting palette
COLORS = {
    1: (237/255, 32/255, 36/255),      # Red
    2: (164/255, 194/255, 219/255),    # Light blue
    3: (57/255, 83/255, 164/255),      # Dark Blue
    4: (50/255, 180/255, 80/255),      # Light Green
    5: (0, 0, 0),                      # Black
    6: (0, 110/255, 0),                # Green
    7: (1.0, 0.0, 1.0),                # Magenta
    8: (0.0, 1.0, 1.0),                # Cyan
    9: (0.0, 0.0, 1.0),                # Normal blue
    10: (1.0, 0.0, 0.0),               # Normal red
    12: (255/255, 102/255, 154/255),   # Lilac
}


@dataclass
class EquilibriumProfiles:
    """Flux-surface-averaged M3D-C1 equilibrium profiles vs. normalized poloidal flux."""
    psi_norm: np.ndarray  # shape=(n,), dimensionless, normalized poloidal flux
    q: np.ndarray         # shape=(n,), dimensionless, safety factor
    F: np.ndarray         # shape=(n,), units=T*m, F = R*Bphi (poloidal current function)
    shear: np.ndarray     # shape=(n,), dimensionless, magnetic shear s = 2*V*(dq/dV)/q
    pe: np.ndarray        # shape=(n,), units=Pa, electron pressure
    pi: np.ndarray        # shape=(n,), units=Pa, ion pressure
    p: np.ndarray         # shape=(n,), units=Pa, total pressure
    te: np.ndarray        # shape=(n,), units=eV, electron temperature
    ti: np.ndarray        # shape=(n,), units=eV, ion temperature
    ne: np.ndarray        # shape=(n,), units=m^-3, electron density
    ni: np.ndarray        # shape=(n,), units=m^-3, ion density
    v_phi: np.ndarray     # shape=(n,), units=m/s, toroidal fluid velocity
    j_phi: np.ndarray     # shape=(n,), units=A/m^2, toroidal current density
    j_pol: np.ndarray     # shape=(n,), units=A/m^2, |poloidal-plane current density| = sqrt(j_R^2+j_Z^2)
    kappa: np.ndarray     # shape=(n,), dimensionless, elongation b/a of each flux surface
    delta: np.ndarray     # shape=(n,), dimensionless, mean triangularity (upper+lower)/2 of each flux surface


def _flux_surface_shape(R: np.ndarray, Z: np.ndarray, R_axis: float) -> tuple[float, float, float, float]:
    """
    Elongation/triangularity of a single closed flux-surface contour, ported from IDL
    get_shape.pro (left/right crossings at the magnetic-axis height via linear interpolation,
    plus top/bottom extrema).

    Parameters
    ----------
    R, Z : np.ndarray
        shape=(m,), units=m, poloidal-angle-ordered points on one flux surface (closed contour,
        not explicitly repeating the first point).
    R_axis : float
        units=m, major-radius coordinate of the magnetic axis (the height Z=Z_axis is not used
        here since get_shape.pro finds left/right crossings at the *axis* Z via its own axis
        estimate; this port instead uses the surface's own vertical center, see below).

    Returns
    -------
    kappa, delta_upper, delta_lower, a : float
        Elongation, upper/lower triangularity, and minor radius of this flux surface.
    """
    top_idx = np.argmax(Z)
    bottom_idx = np.argmin(Z)
    top, top_x = Z[top_idx], R[top_idx]
    bottom, bottom_x = Z[bottom_idx], R[bottom_idx]
    z0 = 0.5 * (top + bottom)

    n = R.size
    left = right = None
    for i in range(n):
        x0, y0 = R[i], Z[i]
        x1, y1 = R[(i + 1) % n], Z[(i + 1) % n]
        dy0, dy1 = z0 - y0, z0 - y1
        if dy0 * dy1 < 0.0:
            xx = (x0 * dy1 - x1 * dy0) / (dy1 - dy0)
            if (R_axis - x0) < 0:
                right = xx
            else:
                left = xx

    a = 0.5 * (right - left)
    r0 = 0.5 * (right + left)
    b = 0.5 * (top - bottom)
    kappa = b / a
    delta_upper = (r0 - top_x) / a
    delta_lower = (r0 - bottom_x) / a
    return kappa, delta_upper, delta_lower, a


def _flux_surface_shape_profile(fc) -> tuple[np.ndarray, np.ndarray]:
    """
    Elongation kappa(psi_N) and mean triangularity delta(psi_N) for every flux surface traced
    by flux_coordinates(), applying the single-surface geometry of _flux_surface_shape() to
    each column of fc.rpath/fc.zpath (shape=(m_poloidal, n_radial), units=m).
    """
    n = fc.rpath.shape[1]
    kappa = np.zeros(n)
    delta = np.zeros(n)
    R_axis = fc.rma
    for j in range(n):
        k, du, dl, _a = _flux_surface_shape(fc.rpath[:, j], fc.zpath[:, j], R_axis)
        kappa[j] = k
        delta[j] = 0.5 * (du + dl)
    return kappa, delta


def _load_profile_data(filename: str = "C1.h5", time: int = 1, fcoords: str = 'pest', points: int = 121) -> EquilibriumProfiles:
    """
    Load and flux-surface-average M3D-C1 equilibrium profiles from a C1.h5 file.

    A single fpy.sim_data object is used for both flux-coordinate geometry and every field
    evaluation, so time=-1 (equilibrium) or time=N (a specific slice) is honored consistently
    everywhere -- unlike sampling a fixed poloidal ray, flux_average() performs a proper
    Jacobian-weighted average over each flux surface, matching IDL's flux_average.pro.

    Parameters
    ----------
    filename : str
        Path to the M3D-C1 HDF5 file. Default is "C1.h5".
    time : int
        Time slice to load (-1 for the equilibrium slice). Default is 1.
    fcoords : str
        Flux coordinate system to use. Default is 'pest'.
    points : int
        Number of flux surfaces (radial resolution). Default is 121.

    Returns
    -------
    EquilibriumProfiles
    """
    sim = fpy.sim_data(filename, time=time)
    kw = dict(sim=sim, filename=filename, fcoords=fcoords, points=points, units='mks')

    psin, q = flux_average('q', coord='scalar', **kw)
    _, F = flux_average('f', coord='scalar', **kw)
    _, shear = flux_average('shear', coord='scalar', **kw)
    _, pe = flux_average('pe', coord='scalar', **kw)
    _, pi = flux_average('pi', coord='scalar', **kw)
    _, p = flux_average('p', coord='scalar', **kw)
    _, te = flux_average('te', coord='scalar', **kw)
    _, ti = flux_average('ti', coord='scalar', **kw)
    _, ne = flux_average('ne', coord='scalar', **kw)
    _, ni = flux_average('ni', coord='scalar', **kw)
    _, v_phi = flux_average('v', coord='phi', **kw)
    _, j_phi = flux_average('j', coord='phi', **kw)
    _, j_R = flux_average('j', coord='R', **kw)
    _, j_Z = flux_average('j', coord='Z', **kw)
    kappa, delta = _flux_surface_shape_profile(sim.fc)

    return EquilibriumProfiles(
        psi_norm=psin, q=q, F=F, shear=shear, pe=pe, pi=pi, p=p, te=te, ti=ti, ne=ne, ni=ni,
        v_phi=v_phi, j_phi=j_phi, j_pol=np.sqrt(j_R**2 + j_Z**2), kappa=kappa, delta=delta,
    )


def _standalone(ax: plt.Axes | None) -> tuple[plt.Axes, bool]:
    """Create a figure/axes if none was given; return (ax, owns_figure)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.3, 5))
        return ax, True
    return ax, False


def plot_q(data: EquilibriumProfiles, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the safety factor q vs. normalized poloidal flux."""
    ax, owns = _standalone(ax)
    ax.plot(data.psi_norm, np.abs(data.q), color=COLORS[3], label='q')
    ax.set_xlim(data.psi_norm[0], data.psi_norm[-1])
    ax.set_ylim(0, np.abs(data.q[-1]))
    ax.set_xlabel('Normalized poloidal flux')
    ax.set_ylabel('q')
    ax.legend()
    ax.grid()
    if owns:
        plt.show()
    return ax


def plot_poloidal_current_function(data: EquilibriumProfiles, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the poloidal current function F = R*Bphi vs. normalized poloidal flux."""
    ax, owns = _standalone(ax)
    ax.plot(data.psi_norm, data.F, color=COLORS[3], label='F')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Normalized poloidal flux')
    ax.set_ylabel('F = R B$_{\\,\\phi}$ ( T$\\,\\cdot\\,$m )')
    ax.legend()
    ax.grid()
    if owns:
        plt.show()
    return ax


def plot_shear(data: EquilibriumProfiles, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the magnetic shear s = 2*V*(dq/dV)/q vs. normalized poloidal flux."""
    ax, owns = _standalone(ax)
    ax.plot(data.psi_norm, data.shear, color=COLORS[3], label='s')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Normalized poloidal flux')
    ax.set_ylabel('Magnetic shear')
    ax.legend()
    ax.grid()
    if owns:
        plt.show()
    return ax


def plot_pressure(data: EquilibriumProfiles, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot electron, ion, and total pressure profiles vs. normalized poloidal flux."""
    ax, owns = _standalone(ax)
    ax.plot(data.psi_norm, data.pe * 1e-3, label='P$_{\\,\\mathrm{e}}$', color=COLORS[1])
    ax.plot(data.psi_norm, data.pi * 1e-3, label='P$_{\\,\\mathrm{i}}$', color=COLORS[3])
    ax.plot(data.psi_norm, data.p * 1e-3, label='P', color=COLORS[5])
    ax.set_xlim(0, 1)
    ax.set_ylim(0,)
    ax.set_xlabel('Normalized poloidal flux')
    ax.set_ylabel('Pressure ( kPa )')
    ax.legend()
    ax.grid()
    if owns:
        plt.show()
    return ax


def plot_temperature(data: EquilibriumProfiles, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot electron and ion temperature profiles vs. normalized poloidal flux."""
    ax, owns = _standalone(ax)
    ax.plot(data.psi_norm, data.te, label='T$_{\\,\\mathrm{e}}$', color=COLORS[1])
    ax.plot(data.psi_norm, data.ti, label='T$_{\\,\\mathrm{i}}$', color=COLORS[3])
    ax.set_xlim(0, 1)
    ax.set_ylim(0,)
    ax.set_xlabel('Normalized poloidal flux')
    ax.set_ylabel('Temperature ( eV )')
    ax.legend()
    ax.grid()
    if owns:
        plt.show()
    return ax


def plot_density(data: EquilibriumProfiles, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot electron and ion density profiles vs. normalized poloidal flux."""
    ax, owns = _standalone(ax)
    ax.plot(data.psi_norm, data.ne, label='n$_{\\,\\mathrm{e}}$', color=COLORS[1])
    ax.plot(data.psi_norm, data.ni, label='n$_{\\,\\mathrm{i}}$', color=COLORS[3])
    ax.set_xlim(0, 1)
    ax.set_ylim(0,)
    ax.set_xlabel('Normalized poloidal flux')
    ax.set_ylabel('Density ( m$^{-3}$ )')
    ax.legend()
    ax.grid()
    if owns:
        plt.show()
    return ax


def plot_velocity(data: EquilibriumProfiles, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the toroidal fluid velocity profile vs. normalized poloidal flux."""
    ax, owns = _standalone(ax)
    ax.plot(data.psi_norm, data.v_phi / 1e3, label='v$_{\\,\\mathrm{\\phi}}$', color=COLORS[3])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Normalized poloidal flux')
    ax.set_ylabel('Velocity ( km/s )')
    ax.legend()
    ax.grid()
    if owns:
        plt.show()
    return ax


def plot_current(data: EquilibriumProfiles, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the toroidal and poloidal-plane current density profiles vs. normalized poloidal flux."""
    ax, owns = _standalone(ax)
    ax.plot(data.psi_norm, data.j_phi / 1e3, label='j$_{\\,\\mathrm{\\phi}}$', color=COLORS[1])
    ax.plot(data.psi_norm, -data.j_pol / 1e3, label='j$_{\\,\\mathrm{R+Z}}$', color=COLORS[3])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Normalized poloidal flux')
    ax.set_ylabel('Current density ( kA$\\,/\\,$m$^2$ )')
    ax.legend()
    ax.grid()
    if owns:
        plt.show()
    return ax


def plot_shape(data: EquilibriumProfiles, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot elongation kappa and mean triangularity delta profiles vs. normalized poloidal flux."""
    ax, owns = _standalone(ax)
    ax.plot(data.psi_norm, data.kappa, label='$\\kappa$', color=COLORS[1])
    ax.plot(data.psi_norm, data.delta, label='$\\delta$', color=COLORS[3])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Normalized poloidal flux')
    ax.set_ylabel('Shape parameter')
    ax.legend()
    ax.grid()
    if owns:
        plt.show()
    return ax


_PANELS: dict[str, Callable[[EquilibriumProfiles, plt.Axes | None], plt.Axes]] = {
    'q': plot_q,
    'F': plot_poloidal_current_function,
    'shear': plot_shear,
    'pressure': plot_pressure,
    'temperature': plot_temperature,
    'density': plot_density,
    'velocity': plot_velocity,
    'current': plot_current,
    'shape': plot_shape,
}


def plot_profiles(filename: str = "C1.h5", time: int = 1, fcoords: str = 'pest', points: int = 121,
                   quantities: list[str] | None = None) -> None:
    """
    Plot M3D-C1 equilibrium profiles: q, F=R*Bphi, magnetic shear, pressure, temperature,
    density, toroidal velocity, current density, and elongation/triangularity.

    Parameters
    ----------
    filename : str
        Path to the M3D-C1 HDF5 file. Default is "C1.h5".
    time : int
        Time slice to plot (-1 for the equilibrium slice). Default is 1.
    fcoords : str
        Flux coordinate system to use. Default is 'pest'.
    points : int
        Number of flux surfaces (radial resolution). Default is 121.
    quantities : list[str] | None
        Subset of panel names (see _PANELS keys) to plot. Default (None) plots all of them.

    Returns
    -------
    None
        Displays the equilibrium profiles plot.
    """
    data = _load_profile_data(filename=filename, time=time, fcoords=fcoords, points=points)
    keys = quantities or list(_PANELS)

    ncols = 3
    nrows = -(-len(keys) // ncols)  # ceil division
    fig, axs = plt.subplots(nrows, ncols, figsize=(7.3 * ncols, 5 * nrows))
    axs = np.atleast_1d(axs).ravel()

    for ax, key in zip(axs, keys):
        _PANELS[key](data, ax=ax)
    for ax in axs[len(keys):]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot M3D-C1 equilibrium profiles.')
    parser.add_argument('--filename', type=str, default='C1.h5', help='Path to the M3D-C1 HDF5 file.')
    parser.add_argument('--time', type=int, default=1, help='Time slice to plot.')
    parser.add_argument('--fcoords', type=str, default='pest', help='Flux coordinate system to use.')
    parser.add_argument('--points', type=int, default=121, help='Number of points along the flux surface.')
    parser.add_argument('--quantities', nargs='+', choices=list(_PANELS), default=None,
                         help='Subset of panels to plot (default: all).')
    args = parser.parse_args()

    plot_profiles(filename=args.filename,
                  time=args.time,
                  fcoords=args.fcoords,
                  points=args.points,
                  quantities=args.quantities)
