#!/home/jfernandes/.venv/bin/python
"""
Port of IDL's plot_omega.pro: flux-surface-averaged plasma rotation-frequency profiles
from M3D-C1 equilibrium/time-slice HDF5 output.

Physics: the reduced-MHD velocity ansatz V = R^2*grad(U) x grad(phi) + Omega*R^2*grad(phi)
+ grad(chi) (U='phi' field, Omega='V'/'omega' field, chi='chi' field, all raw M3D-C1
potentials, distinct from the reconstructed physical fields eval_field() normally exposes)
implies the raw toroidal angular velocity Omega is Doppler-shifted within a flux surface by
the poloidal (U-, chi-driven) flow. v_omega below removes that shift, giving the
flux-surface-consistent rotation frequency IDL's plot_omega.pro plots as "omega".

omega_*i/omega_*e (ion/electron diamagnetic frequencies) require the raw M3D-C1-normalized
`db` (ion skin depth, doc/inputs.tex) parameter, only meaningful in code-normalized units. p,
pe, psi, I, den come from eval_field()'s typedict route and are true SI values (confirmed
empirically: fetching the same HDF5 dataset via the typedict route vs. the raw fio
FIO_SCALAR_FIELD route gives a ratio exactly equal to the field's SI normalization constant,
e.g. p0 for pressure), so these are round-tripped SI -> M3D-C1-normalized -> SI around the
db-arithmetic step. phi ('phi'=U), chi, and omega ('V') are opened via that same raw
FIO_SCALAR_FIELD route (scale factor exactly 1. in m3dc1_field.cpp) and so come back already
code-normalized -- their VALUES are used as-is, and their GRADIENTS (still differentiated
w.r.t. real SI (R,Z), since the mesh/coordinate convention is uniform across every field) are
rescaled by L0 [m] to match IDL's own d/d(R/L0) gradient convention.

Validated against a genuine two-fluid, rotating TCABR run
(~/m3dc1_data/shot0009/n1/two_fluid/coil_sets_1kAt/IM_set_000/C1.h5, db=9.108e-3, cross-checked
against C1stdout's "Physical value of db"): omega_star_i/omega_star_e come out nonzero with
opposite sign (ion vs. electron diamagnetic frequency, as expected for a peaked pressure
profile), and the internal identities omega_ExB = omega_i - omega_star_i and
omega_e = omega_ExB + omega_star_e hold exactly.
"""
import argparse
from dataclasses import dataclass

import fpy
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from m3dc1.eval_field import eval_field, eval_m3dc1_field_deriv
from m3dc1.flux_average import flux_average, flux_average_field
from m3dc1.read_h5 import readParameter
from m3dc1.unit_conv import unit_conv

# Plotting palette (matches plot_profiles.py)
COLORS = {
    1: (237/255, 32/255, 36/255),      # Red
    3: (57/255, 83/255, 164/255),      # Dark Blue
    4: (50/255, 180/255, 80/255),      # Light Green
    5: (0, 0, 0),                      # Black
    6: (0, 110/255, 0),                # Green
}


@dataclass
class OmegaProfiles:
    """Flux-surface-averaged rotation-frequency profiles vs. normalized poloidal flux."""
    psi_norm: np.ndarray       # shape=(n,), dimensionless, normalized poloidal flux
    omega_ExB: np.ndarray      # shape=(n,), units=krad/s, E x B rotation frequency
    omega_i: np.ndarray        # shape=(n,), units=krad/s, ion fluid (Doppler-corrected) rotation frequency
    omega_e: np.ndarray        # shape=(n,), units=krad/s, electron fluid rotation frequency
    omega_star_i: np.ndarray   # shape=(n,), units=krad/s, ion diamagnetic frequency
    omega_star_e: np.ndarray   # shape=(n,), units=krad/s, electron diamagnetic frequency


def _psi_value_and_grad(R: np.ndarray, phi: np.ndarray, Z: np.ndarray, sim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Poloidal flux psi = R*A_phi and its R,Z-derivatives via the product rule, matching
    m3dc1/flux_coordinates.py's field_at_point('psi0'/'psi0_r'/'psi0_z').
    """
    Aphi = eval_field('A', R, phi, Z, coord='phi', sim=sim, quiet=True)
    dA = eval_m3dc1_field_deriv('A', R, phi, Z, sim=sim)  # shape=(9,...): RR,phiR,ZR,Rphi,phiphi,Zphi,RZ,phiZ,ZZ
    dAphi_dR, dAphi_dZ = dA[1], dA[7]
    psi = R * Aphi
    psi_r = Aphi + R * dAphi_dR
    psi_z = R * dAphi_dZ
    return psi, psi_r, psi_z


def _compute_omega_profiles(filename: str = "C1.h5", time: int = 1, fcoords: str = 'pest', points: int = 200) -> tuple[OmegaProfiles, object]:
    """
    Load and flux-surface-average M3D-C1 rotation-frequency profiles from a C1.h5 file.

    Geometry (flux coordinates), psi, I=R*Bphi, and den=ni are taken from the equilibrium
    slice (time=-1); the dynamic potentials (phi=U, chi, V=omega, p, pe) are taken from the
    requested time slice -- matching IDL plot_omega.pro's /equilibrium vs. time-slice split.

    Parameters
    ----------
    filename : str
        Path to the M3D-C1 HDF5 file. Default is "C1.h5".
    time : int
        Time slice for the dynamic (rotation/pressure) fields. Default is 1.
    fcoords : str
        Flux coordinate system to use. Default is 'pest'.
    points : int
        Number of flux surfaces (radial resolution). Default is 200.

    Returns
    -------
    OmegaProfiles, fc
        Profiles and the underlying flux-coordinate object (for the resonant-surface overlay).
    """
    sim_eq = fpy.sim_data(filename, time=-1)
    sim_t = fpy.sim_data(filename, time=time)

    # Establish flux-coordinate geometry from the equilibrium slice.
    flux_average('q', coord='scalar', sim=sim_eq, filename=filename, fcoords=fcoords, points=points, units='mks')
    fc = sim_eq.fc

    torphi = np.zeros_like(fc.rpath)
    r = fc.rpath if fc.itor == 1 else np.ones_like(fc.rpath)

    _, psi_r, psi_z = _psi_value_and_grad(fc.rpath, torphi, fc.zpath, sim_eq)
    I_field = eval_field('I', fc.rpath, torphi, fc.zpath, coord='scalar', sim=sim_eq, quiet=True)
    den = eval_field('ni', fc.rpath, torphi, fc.zpath, coord='scalar', sim=sim_eq, quiet=True)

    dp = eval_m3dc1_field_deriv('p', fc.rpath, torphi, fc.zpath, sim=sim_t)
    p_r, p_z = dp[0], dp[2]
    dpe = eval_m3dc1_field_deriv('pe', fc.rpath, torphi, fc.zpath, sim=sim_t)
    pe_r, pe_z = dpe[0], dpe[2]

    # 'phi' (U), 'chi', 'V' (raw omega) are opened via fio's generic FIO_SCALAR_FIELD path
    # (m3dc1_field.cpp: m3dc1_scalar_field(this, field_name, 1.) -- scale factor exactly 1),
    # unlike typedict fields (p, pe, ni, A, B, ...) which fio scales to SI internally. So
    # these three come back as raw M3D-C1-normalized VALUES already (confirmed empirically:
    # fetching the same HDF5 dataset via the typedict route vs. this raw route gives a ratio
    # exactly equal to the field's SI normalization constant, e.g. p0 for pressure). Their
    # GRADIENTS, however, are still differentiated w.r.t. real SI (R,Z) (the mesh/coordinate
    # convention is uniform for every field), so d(raw code value)/d(R_SI) must be rescaled by
    # L0 [m] to match IDL's own gradient convention (d/d(R/L0)) before combining with the
    # code-unit psi_r/psi_z/I below.
    L0_m = unit_conv(np.array(1.0), arr_dim='m3dc1', sim=sim_eq, length=1)
    du = eval_m3dc1_field_deriv('phi', fc.rpath, torphi, fc.zpath, sim=sim_t)
    u_r_n, u_z_n = du[0] * L0_m, du[2] * L0_m
    dchi = eval_m3dc1_field_deriv('chi', fc.rpath, torphi, fc.zpath, sim=sim_t)
    chi_r_n, chi_z_n = dchi[0] * L0_m, dchi[2] * L0_m
    omega_raw_n = eval_field('V', fc.rpath, torphi, fc.zpath, coord='scalar', sim=sim_t, quiet=True)

    # SI -> M3D-C1-normalized code units for the properly SI-scaled ingredients (db is only
    # meaningful in code-normalized units).
    conv = lambda val, **dims: unit_conv(val, arr_dim='mks', sim=sim_eq, **dims)
    psi_r_n = conv(psi_r, magnetic_field=1, length=1)
    psi_z_n = conv(psi_z, magnetic_field=1, length=1)
    I_n = conv(I_field, magnetic_field=1, length=1)
    den_n = conv(den, particles=1)
    p_r_n = conv(p_r, pressure=1, length=-1)
    p_z_n = conv(p_z, pressure=1, length=-1)
    pe_r_n = conv(pe_r, pressure=1, length=-1)
    pe_z_n = conv(pe_z, pressure=1, length=-1)

    db = readParameter('db', sim=sim_eq)

    psipsi = psi_r_n**2 + psi_z_n**2
    pprime = (p_r_n * psi_r_n + p_z_n * psi_z_n) / psipsi
    peprime = (pe_r_n * psi_r_n + pe_z_n * psi_z_n) / psipsi
    piprime = pprime - peprime

    v_omega_n = omega_raw_n - I_n / (r**2 * psipsi) * (
        r**2 * (u_r_n * psi_r_n + u_z_n * psi_z_n) + (chi_z_n * psi_r_n - chi_r_n * psi_z_n) / r
    )
    w_star_i_n = db * piprime / den_n
    w_star_e_n = -db * peprime / den_n

    # M3D-C1-normalized -> SI (1/time dimension).
    v_omega = unit_conv(v_omega_n, arr_dim='m3dc1', sim=sim_eq, time=-1)
    w_star_i = unit_conv(w_star_i_n, arr_dim='m3dc1', sim=sim_eq, time=-1)
    w_star_e = unit_conv(w_star_e_n, arr_dim='m3dc1', sim=sim_eq, time=-1)

    omega_ExB = v_omega - w_star_i
    ve_omega = omega_ExB + w_star_e

    def fa(field_2d, name):
        return flux_average_field(field_2d, fc.j, fc.n, 'mks', name, sim_eq) / 1e3  # rad/s -> krad/s

    profiles = OmegaProfiles(
        psi_norm=fc.psi_norm,
        omega_ExB=fa(omega_ExB, 'omega_ExB'),
        omega_i=fa(v_omega, 'omega_i'),
        omega_e=fa(ve_omega, 'omega_e'),
        omega_star_i=fa(w_star_i, 'omega_star_i'),
        omega_star_e=fa(w_star_e, 'omega_star_e'),
    )
    return profiles, fc


def _resonant_surfaces(fc, q_val: list[float], sim_eq, mtop: float = 0.05, mslope: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Invert q(psi_N) at the requested q values; return (psin_res, m) mode labels for n=ntor."""
    q_interp = interp1d(np.abs(fc.q), fc.psi_norm, bounds_error=False)
    psin_res = q_interp(np.abs(np.asarray(q_val)))
    ntor = readParameter('ntor', sim=sim_eq)
    m = np.round(np.asarray(q_val) * ntor).astype(int)
    return psin_res, m


def plot_omega(filename: str = "C1.h5", time: int = 1, fcoords: str = 'pest', points: int = 200,
               q_val: list[float] | None = None, mtop: float = 0.05, mslope: float = 0.0,
               ax: plt.Axes | None = None) -> plt.Axes:
    """
    Plot flux-surface-averaged plasma rotation-frequency profiles: E x B, ion/electron fluid
    rotation, and ion/electron diamagnetic frequency, vs. normalized poloidal flux.

    Parameters
    ----------
    filename : str
        Path to the M3D-C1 HDF5 file. Default is "C1.h5".
    time : int
        Time slice for the dynamic (rotation/pressure) fields. Default is 1.
    fcoords : str
        Flux coordinate system to use. Default is 'pest'.
    points : int
        Number of flux surfaces (radial resolution). Default is 200.
    q_val : list[float] | None
        Safety-factor values at which to overlay resonant (q=m/n) surfaces. Default is None.
    mtop, mslope : float
        Vertical positioning of the resonant-surface mode-number labels (fraction of the
        y-axis range from the top, plus a per-surface slope), matching IDL's keywords.
    ax : plt.Axes | None
        Axes to draw on; if None, a new figure is created and shown.

    Returns
    -------
    plt.Axes
    """
    profiles, fc = _compute_omega_profiles(filename=filename, time=time, fcoords=fcoords, points=points)

    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.plot(profiles.psi_norm, profiles.omega_ExB, color=COLORS[5], label=r'$\omega_{\,\mathrm{E \times B}}$')
    ax.plot(profiles.psi_norm, profiles.omega_i, color=COLORS[1], label=r'$\omega$')
    ax.plot(profiles.psi_norm, profiles.omega_e, color=COLORS[3], label=r'$\omega_{\,\mathrm{e}}$')
    ax.plot(profiles.psi_norm, profiles.omega_star_i, color=COLORS[4], linestyle='--', label=r'$\omega_{\,\mathrm{*i}}$')
    ax.plot(profiles.psi_norm, profiles.omega_star_e, color=COLORS[6], linestyle='--', label=r'$\omega_{\,\mathrm{*e}}$')
    ax.axhline(0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('Normalized poloidal flux')
    ax.set_ylabel(r'$\Omega$ ( krad/s )')
    ax.legend()
    ax.grid()

    if q_val:
        sim_eq = fpy.sim_data(filename, time=-1)
        psin_res, m = _resonant_surfaces(fc, q_val, sim_eq, mtop=mtop, mslope=mslope)
        ylo, yhi = ax.get_ylim()
        top = (yhi - ylo) * mslope * np.arange(len(m)) + (yhi - ylo) * (1.0 - mtop) + ylo
        for i, (psin_i, m_i) in enumerate(zip(psin_res, m)):
            if np.isnan(psin_i):
                continue
            ax.axvline(psin_i, color='k', linestyle=':', linewidth=1)
            label = f'm = {m_i}' if i == 0 else str(m_i)
            ax.text(psin_i, top[i], label, fontsize=8, ha='center')

    if owns_fig:
        plt.show()
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot M3D-C1 rotation-frequency profiles (E x B, fluid, diamagnetic).')
    parser.add_argument('--filename', type=str, default='C1.h5', help='Path to the M3D-C1 HDF5 file.')
    parser.add_argument('--time', type=int, default=1, help='Time slice for the dynamic fields.')
    parser.add_argument('--fcoords', type=str, default='pest', help='Flux coordinate system to use.')
    parser.add_argument('--points', type=int, default=200, help='Number of flux surfaces (radial resolution).')
    parser.add_argument('--q_val', type=float, nargs='+', default=None, help='Safety-factor values to overlay as resonant surfaces.')
    parser.add_argument('--mtop', type=float, default=0.05, help='Resonant-surface label vertical offset from the top.')
    parser.add_argument('--mslope', type=float, default=0.0, help='Resonant-surface label vertical slope across surfaces.')
    args = parser.parse_args()

    plot_omega(filename=args.filename,
               time=args.time,
               fcoords=args.fcoords,
               points=args.points,
               q_val=args.q_val,
               mtop=args.mtop,
               mslope=args.mslope)
