#!/home/jfernandes/.venv/bin/python
"""Validate flare_surfmn.py against IDL's plot_br.pro/schaffer_plot.pro reference
NetCDF output (bmn_vac.nc / bmn_res.nc), for a single response_probeg-style case.

Corrected normalization derived+validated against IDL (see project notes):

    Bmn [G] = 2 * (2*pi)^2 * |Phi_mn(psi_norm)| / area(psi_norm) * 1e4

replacing flare_surfmn.py's original (buggy):

    db_matrix = |Phi_mn| * area * 1e4

Root cause: IDL's PEST jacobian is jac_IDL = -Delta_psi * R^2 * q / F
(flux_coordinates.pro:335), i.e. FLARE's own jacobian J = q*R^2/F
(fourier_transform.f90) times -Delta_psi. schaffer_plot.pro's final
`d/fc.area/fc.dpsi_dchi` step divides by that same Delta_psi again, so it
cancels -- meaning FLARE's raw Phi_mn should NOT be divided by Delta_psi at
all, only by area, times (2*pi)^2. The extra factor of 2 comes from IDL's
`bx`,`bz` being read via /complex (i.e. already the single-harmonic complex
amplitude F_c in the convention physical_field=Re[F_c*exp(i*n*phi)]); a
Fourier projection of Re[F_c*exp(inphi)] against exp(inphi) over a full
toroidal turn picks up a factor of 1/2 relative to using F_c directly, so
comparing against IDL's convention requires multiplying by 2.

IMPORTANT: also note IDL's `m` array in the .nc file is stored in
*descending* order (e.g. 199..-200), while FLARE's fourier_transform returns
Phimn ordered *ascending* (-mmax/2+1..mmax/2, off-by-one relative to IDL's
range too, e.g. -199..200 vs IDL's -200..199) -- any direct array-index
comparison between the two must first align by m *value*, not position.
"""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent))
from flare import model
from flare.analysis import fourier_transform, fluxsurf2d_parameters, equi2d_rzarray
from flare_surfmn import fluxsurf_params

TWO_PI_SQ_TIMES_2 = 2.0 * (2.0 * np.pi) ** 2


def load_nc(path: str) -> dict:
    d = netcdf_file(path, "r", mmap=False)
    return {k: np.array(v[:]) for k, v in d.variables.items()}


def corrected_bmn(psiN: float, n_tor: int, mmax: int, area: float) -> np.ndarray:
    """|Bmn| [Gauss] at fixed psiN, for m = -mmax/2+1 .. mmax/2 (ascending)."""
    Phimn = fourier_transform(psiN, n_tor, mmax)
    return TWO_PI_SQ_TIMES_2 * np.abs(Phimn) / area * 1e4


def compute_case(flare_model_dir: str, nc_path: str, n_tor: int, m_max_lowres: int) -> dict:
    """Recompute the surfmn spectrum with the corrected formula, at both the
    tool's original (low) poloidal resolution and at IDL's own (high)
    resolution/grid, and load the IDL reference for comparison."""
    idl = load_nc(nc_path)

    model.load(flare_model_dir)
    try:
        t0 = time.time()
        psiN_lr, q_lr, area_lr, psiN_res, q_res = fluxsurf_params(n_tor, m_max_lowres)
        m_lr = np.arange(-(m_max_lowres + 1) + 1, (m_max_lowres + 1) + 1)
        db_lr = np.zeros((len(psiN_lr), len(m_lr)))
        for i, psiN in enumerate(psiN_lr):
            db_lr[i, :] = corrected_bmn(psiN, n_tor, 2 * (m_max_lowres + 1), area_lr[i])
        print(f"  low-res ({len(psiN_lr)} pts, mmax={len(m_lr)}): {time.time()-t0:.1f} s")

        t0 = time.time()
        psiN_idl = idl["psi_norm"]
        m_idl = idl["m"]
        mmax_hr = len(m_idl)
        m_hr = np.arange(-mmax_hr / 2 + 1, mmax_hr / 2 + 1)
        db_hr = np.zeros((len(psiN_idl), mmax_hr))
        area_hr = np.zeros(len(psiN_idl))
        for i, psiN in enumerate(psiN_idl):
            R, Z = equi2d_rzarray(np.array([psiN]), 0.0)
            _, _, area_pt, _, _ = fluxsurf2d_parameters((R[0], Z[0]))
            area_hr[i] = area_pt
            db_hr[i, :] = corrected_bmn(psiN, n_tor, mmax_hr, area_pt)
        print(f"  high-res ({len(psiN_idl)} pts, mmax={mmax_hr}): {time.time()-t0:.1f} s")
    finally:
        model.free()

    idl_amp = np.sqrt(idl["bmn_real"] ** 2 + idl["bmn_imag"] ** 2)
    return dict(
        psiN_lr=psiN_lr, m_lr=m_lr, db_lr=db_lr, q_lr=q_lr, area_lr=area_lr,
        psiN_res=psiN_res, q_res=q_res,
        psiN_hr=psiN_idl, m_hr=m_hr, db_hr=db_hr, area_hr=area_hr,
        idl_psiN=psiN_idl, idl_m=m_idl, idl_amp=idl_amp,
        idl_area=idl["area"], idl_q=idl["q"], idl_psi=idl["psi"],
    )


def relative_error(a_vals: np.ndarray, a_m: np.ndarray, b_vals: np.ndarray, b_m: np.ndarray,
                    a_psiN: np.ndarray | None = None, b_psiN: np.ndarray | None = None,
                    amp_floor: float = 0.02) -> dict:
    """Relative error between two |Bmn|(m,psiN) arrays, aligning by m *value*
    (not index -- IDL's m array is descending and off-by-one from FLARE's),
    and interpolating over psiN if the grids differ."""
    common_m = np.intersect1d(a_m, b_m)
    a_pos = {v: i for i, v in enumerate(a_m)}
    b_pos = {v: i for i, v in enumerate(b_m)}
    ia = np.array([a_pos[v] for v in common_m])
    ib = np.array([b_pos[v] for v in common_m])

    a_sel = a_vals[:, ia]
    b_sel = b_vals[:, ib]

    if a_psiN is not None and b_psiN is not None and not np.array_equal(a_psiN, b_psiN):
        b_interp = np.zeros((len(a_psiN), len(common_m)))
        for j in range(len(common_m)):
            f = interp1d(b_psiN, b_sel[:, j], bounds_error=False, fill_value=np.nan)
            b_interp[:, j] = f(a_psiN)
        b_sel = b_interp

    mask = b_sel > amp_floor
    rel_err = np.abs(a_sel - b_sel) / b_sel
    return dict(
        common_m=common_m, rel_err=rel_err, mask=mask,
        rms=float(np.sqrt(np.nanmean(rel_err[mask] ** 2))),
        median=float(np.nanmedian(rel_err[mask])),
        p95=float(np.nanpercentile(rel_err[mask], 95)),
        max=float(np.nanmax(rel_err[mask])),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("flare_model", help="FLARE model directory (.bfield/.boundary)")
    ap.add_argument("nc_file", help="IDL schaffer_plot.pro reference NetCDF (bmn_*.nc)")
    ap.add_argument("n_tor", type=int)
    ap.add_argument("m_max_lowres", type=int, help="matches the m_max used to generate the .npz being validated")
    ap.add_argument("--out", default="validation_results.npz")
    args = ap.parse_args()

    result = compute_case(args.flare_model, args.nc_file, args.n_tor, args.m_max_lowres)

    hr = relative_error(result["db_hr"], result["m_hr"], result["idl_amp"], result["idl_m"])
    print(f"\nHigh-res (matched grid) vs IDL: RMS={hr['rms']:.4f} median={hr['median']:.4f} "
          f"p95={hr['p95']:.4f} max={hr['max']:.4f}")

    lr = relative_error(result["db_lr"], result["m_lr"], result["idl_amp"], result["idl_m"],
                         a_psiN=result["psiN_lr"], b_psiN=result["idl_psiN"])
    print(f"Low-res (tool resolution) vs IDL: RMS={lr['rms']:.4f} median={lr['median']:.4f} "
          f"p95={lr['p95']:.4f} max={lr['max']:.4f}")

    np.savez(args.out, **result)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
