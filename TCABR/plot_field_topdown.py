#!/home/jfernandes/.venv/bin/python
"""
Backup/dev copy of a routine meant to live in the fusion-io m3dc1 package
(~/software/fusion-io/build_shared/lib/m3dc1/). Deploy by copying this file
verbatim into that directory and adding
    from m3dc1.plot_field_topdown import plot_field_topdown
to that package's __init__.py (same pattern already used for plot_profiles.py).
"""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from distutils.version import StrictVersion

import m3dc1.fpylib as fpyl
from m3dc1.get_field import get_field_vs_phi


def plot_field_topdown(field: str, coord: str = 'scalar', row: int = 1, sim=None, filename: str = 'C1.h5',
                        time=None, Z_cut: float = 0.0, linear: bool = False, diff: bool = False,
                        phi_res: int = 200, points: int = 250, units: str = 'mks', cmap: str = 'viridis',
                        cmap_midpt: float = None, clvl: int = 100, prange=None, wall: str = None,
                        quiet: bool = False, save: bool = False, savedir: str = None, pub: bool = False,
                        titlestr: str = None, showtitle: bool = True, shortlbl: bool = False, ntor: int = None,
                        export: bool = False, txtname: str = None) -> None:
    """
    Plot an M3D-C1 field over a constant-Z plane, rendered as a Cartesian top-down view of the
    tokamak (X = R*cos(phi), Y = R*sin(phi)), i.e. the analogue of plot_field.py's poloidal
    (R,Z at fixed phi) cross-section but for a fixed Z sweeping the full toroidal angle instead.

    Thin wrapper around m3dc1.get_field.get_field_vs_phi(..., cutz=Z_cut), which already builds
    the (R, phi, Z=Z_cut) grid and evaluates the field there via eval_field/fpy field.evaluate()
    -- see that function for the grid/evaluation mechanics. This routine only adds the
    polar-to-Cartesian projection, the top-down rendering, and an optional first-wall overlay.

    Parameters
    ----------
    field : str
        Field to evaluate, e.g. 'ne', 'psi', 'B', 'j' (see sim.available_fields).
    coord : str
        Field component: 'scalar', 'R', 'phi', 'Z', 'poloidal', 'radial', 'vector', 'tensor'.
    row : int
        For tensor fields, which row (1, 2, or 3) to plot.
    sim : fpy.sim_data, list of fpy.sim_data, or None
        Existing sim_data object(s); if None, one is built from filename/time.
    filename : str
        Path to the M3D-C1 HDF5 file.
    time : int, str, or None
        Time slice to evaluate ('last', or an int; see fpy.sim_data / get_field_vs_phi for the
        diff/linear two-time-slice conventions).
    Z_cut : float
        The constant-Z plane to plot, in meters (e.g. 0.0 for the midplane).
    linear : bool
        Subtract the equilibrium (time=-1) from the field. Default False.
    diff : bool
        Plot the difference between two times/files. Default False.
    phi_res : int
        Number of points in the toroidal direction, spanning the full 0 to 2*pi. Default 200.
    points : int
        Number of points in the R direction. Default 250.
    units : str
        Units for the field ('mks' or 'm3dc1'). Default 'mks'.
    cmap : str
        Colormap. Default 'viridis'.
    cmap_midpt : float or None
        If given, center the colormap (TwoSlopeNorm) at this value.
    clvl : int
        Number of contour levels. Default 100.
    prange : tuple of float or None
        (vmin, vmax) to clip the colormap range.
    wall : str or None
        Path to an R,Z first-wall/limiter contour file (tcabr_first_wall.txt convention: one
        leading point-count header line, then whitespace-separated 'R Z' rows). If given, the
        radii where this contour crosses Z=Z_cut are drawn as reference circles.
    quiet : bool
        Suppress progress printing. Default False.
    save : bool
        Save the figure as a PNG. Default False.
    savedir : str or None
        Directory prefix for the saved figure.
    pub : bool
        Use larger fonts/linewidths for publication figures.
    titlestr : str or None
        Plot title; auto-generated if None.
    showtitle : bool
        Whether to display the title. Default True.
    shortlbl : bool
        Use shortened colorbar labels. Default False.
    ntor : int or None
        Toroidal mode number used in the saved filename; defaults to sim[0].ntor.
    export : bool
        Save the X, Y, and field arrays as text files. Default False.
    txtname : str or None
        Filename prefix used when export=True.

    Returns
    -------
    None
        Displays (and optionally saves/exports) the top-down field plot.
    """

    sim, time, mesh_ob, R, phi_list, Z, R_mesh, Z_mesh, R_ave, Z_ave, phi_ave, field1_ave = get_field_vs_phi(
        field=field, coord=coord, row=row, sim=sim, filename=filename, time=time, cutz=Z_cut, cutr=None,
        linear=linear, diff=diff, phi_res=phi_res, res=points, units=units, quiet=quiet)

    # phi_ave is already in radians (get_field_vs_phi sweeps np.linspace(0, 2*pi, phi_res))
    X = R_ave * np.cos(phi_ave)  # shape=(phi_res,points), units=m
    Y = R_ave * np.sin(phi_ave)  # shape=(phi_res,points), units=m

    fieldlabel, unitlabel = fpyl.get_fieldlabel(units, field, shortlbl=shortlbl)
    cbarlbl = fieldlabel + ' (' + unitlabel + ')'

    if pub:
        axlblfs, titlefs, cbarlblfs, cbarticklblfs, ticklblfs = 20, 18, 14, 14, 18
    else:
        axlblfs = titlefs = cbarlblfs = cbarticklblfs = ticklblfs = None

    if coord not in ['vector', 'tensor']:
        fig, axs = plt.subplots(1, 1)
        fig.set_figheight(7)
        axs = np.asarray([axs])
    else:
        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(14, 7))
        comp = ['R', r'\phi', 'Z']

    if titlestr is None:
        titlestr = field + ' at time=' + str(sim[0].timeslice)
        if linear:
            titlestr = titlestr + ' linear'
        elif diff:
            titlestr = titlestr + ' - time=' + str(sim[1].timeslice)
        try:
            species = sim[0].available_fields[field][2]
        except Exception:
            species = None
        if species is not None:
            titlestr = titlestr + ' - Species: ' + str(species)
        titlestr = titlestr + f', top view (Z={Z_cut:.3g} m)'
    if showtitle:
        plt.title(titlestr, fontsize=titlefs)

    wall_radii = _wall_crossing_radii(wall, Z_cut) if wall is not None else np.array([])
    if wall is not None and wall_radii.size == 0:
        print(f"Warning: Z_cut={Z_cut} does not cross the wall contour in {wall}; skipping wall overlay.")
    theta = np.linspace(0, 2 * np.pi, 200)

    for i, ax in enumerate(axs):
        if cmap_midpt is not None:
            field1_ave_clean = field1_ave[i][np.logical_not(np.isnan(field1_ave[i]))]
            vmin, vmax = np.amin(field1_ave_clean), np.amax(field1_ave_clean)
            vmid = cmap_midpt
            if not np.all(np.diff([vmin, vmid, vmax]) >= 0):
                vmid = vmin + (vmax - vmin) / 2
                fpyl.printwarn('WARNING: cmap_midpt is either too low or too high! '
                                'Resetting cmap_midpt to vmin + (vmax-vmin)/2.')
            if StrictVersion(sys.modules[plt.__package__].__version__) < StrictVersion('3.2'):
                norm = colors.DivergingNorm(vmin=vmin, vcenter=vmid, vmax=vmax)
            else:
                norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)
            cont = ax.contourf(X, Y, field1_ave[i], clvl, cmap=cmap, norm=norm)
        elif isinstance(prange, (tuple, list)):
            cont = ax.contourf(X, Y, field1_ave[i], clvl, cmap=cmap, vmin=prange[0], vmax=prange[1])
        else:
            cont = ax.contourf(X, Y, field1_ave[i], clvl, cmap=cmap)

        for R_cross in wall_radii:
            ax.plot(R_cross * np.cos(theta), R_cross * np.sin(theta), color='k', linewidth=1, zorder=5)

        ax.set_xlabel(r'$X/m$', fontsize=axlblfs)
        ax.set_ylabel(r'$Y/m$', fontsize=axlblfs)
        ax.tick_params(axis='both', which='major', labelsize=ticklblfs)
        ax.set_aspect('equal')
        if coord in ['vector', 'tensor']:
            compstr = r'$' + field + '_{' + comp[i] + '}$'
            ax.set_title(compstr, fontsize=titlefs)

        sfmt = ticker.ScalarFormatter()
        sfmt.set_powerlimits((-3, 4))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(cont, format=sfmt, ax=ax, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=cbarticklblfs)
        cbar.ax.yaxis.offsetText.set(size=cbarticklblfs)
        cbar.set_label(cbarlbl, fontsize=cbarlblfs)
        cont.set_edgecolor("face")

    plt.rcParams["axes.axisbelow"] = False
    plt.tight_layout()
    plt.show()

    if save:
        timestr = 't' + (str(time[0]) + '-' + str(time[1]) if diff else str(time[0]))
        fieldstr = ('delta_' + field) if linear else field
        if savedir is not None:
            fieldstr = savedir + fieldstr
        nout = sim[0].ntor if ntor is None else ntor
        plt.savefig(fieldstr + '_' + timestr + '_n' + "{:d}".format(nout) + '_topdown.png',
                    format='png', dpi=900, bbox_inches='tight')

    if export:
        np.savetxt(txtname + '_X', np.vstack(X), delimiter='   ')
        np.savetxt(txtname + '_Y', np.vstack(Y), delimiter='   ')
        np.savetxt(txtname + '_field', np.vstack(field1_ave), delimiter='   ')


def _wall_crossing_radii(wall_path: str, Z_cut: float) -> np.ndarray:
    """
    R values (m) where a closed R,Z wall/limiter contour crosses the horizontal plane Z=Z_cut.

    File convention (tcabr_first_wall.txt): one leading point-count header line, then
    whitespace-separated 'R Z' rows. Crossings are found by linear interpolation along each
    contour segment, including the closing segment from the last point back to the first.

    Parameters
    ----------
    wall_path : str
        Path to the wall/limiter contour file.
    Z_cut : float
        Height (m) of the constant-Z cutting plane.

    Returns
    -------
    np.ndarray
        1D array of crossing radii (m), one per contour segment that straddles Z_cut. Empty if
        Z_cut lies outside the contour's Z range.
    """
    wall_rz = np.loadtxt(wall_path, skiprows=1, ndmin=2)  # shape=(npts,2), columns=(R,Z), units=m
    R1, Z1 = wall_rz[:, 0], wall_rz[:, 1]
    R2, Z2 = np.roll(R1, -1), np.roll(Z1, -1)

    d1 = Z1 - Z_cut
    d2 = Z2 - Z_cut
    straddles = (d1 == 0) | (d1 * d2 < 0)

    crossings = []
    for i in np.where(straddles)[0]:
        if d1[i] == 0:
            crossings.append(R1[i])
        else:
            t = d1[i] / (Z1[i] - Z2[i])
            crossings.append(R1[i] + t * (R2[i] - R1[i]))
    return np.array(crossings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot an M3D-C1 field over a constant-Z plane "
                                                   "(top-down Cartesian view of the tokamak).")
    parser.add_argument("field", type=str, help="Field to evaluate, e.g. 'ne', 'psi', 'B', 'j'.")
    parser.add_argument("--filename", type=str, default="C1.h5", help="Path to the M3D-C1 HDF5 file.")
    parser.add_argument("--coord", type=str, default="scalar",
                         help="Field component: scalar, R, phi, Z, poloidal, radial, vector, tensor.")
    parser.add_argument("--row", type=int, default=1, help="For tensor fields, row (1, 2, or 3) to plot.")
    parser.add_argument("--time", type=int, default=None, help="Time slice to evaluate.")
    parser.add_argument("--Z", type=float, default=0.0, dest="Z_cut", help="Constant-Z cutting plane, in meters.")
    parser.add_argument("--phi_res", type=int, default=200, help="Number of points in the toroidal direction.")
    parser.add_argument("--points", type=int, default=250, help="Number of points in the R direction.")
    parser.add_argument("--units", type=str, default="mks", help="Units for the field ('mks' or 'm3dc1').")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap.")
    parser.add_argument("--cmap_midpt", type=float, default=None, help="Center the colormap at this value.")
    parser.add_argument("--clvl", type=int, default=100, help="Number of contour levels.")
    parser.add_argument("--prange", type=float, nargs=2, default=None, help="(vmin, vmax) colormap range.")
    parser.add_argument("--wall", type=str, default=None,
                         help="Path to an R,Z first-wall contour file, drawn as crossing-radius circles.")
    parser.add_argument("--linear", action="store_true", help="Subtract the equilibrium from the field.")
    parser.add_argument("--diff", action="store_true", help="Plot the difference between two times/files.")
    parser.add_argument("--save", action="store_true", help="Save the figure as a PNG.")
    parser.add_argument("--savedir", type=str, default=None, help="Directory prefix for the saved figure.")
    parser.add_argument("--pub", action="store_true", help="Use larger fonts/linewidths for publication figures.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress printing.")
    parser.add_argument("--titlestr", type=str, default=None, help="Plot title; auto-generated if not given.")
    parser.add_argument("--no-title", action="store_false", dest="showtitle", help="Do not display the title.")
    parser.add_argument("--shortlbl", action="store_true", help="Use shortened colorbar labels.")
    parser.add_argument("--ntor", type=int, default=None, help="Toroidal mode number for the saved filename.")
    parser.add_argument("--export", action="store_true", help="Save the X, Y, and field arrays as text files.")
    parser.add_argument("--txtname", type=str, default=None, help="Filename prefix used when --export is set.")

    args = parser.parse_args()

    plot_field_topdown(
        field=args.field,
        coord=args.coord,
        row=args.row,
        filename=args.filename,
        time=args.time,
        Z_cut=args.Z_cut,
        linear=args.linear,
        diff=args.diff,
        phi_res=args.phi_res,
        points=args.points,
        units=args.units,
        cmap=args.cmap,
        cmap_midpt=args.cmap_midpt,
        clvl=args.clvl,
        prange=args.prange,
        wall=args.wall,
        quiet=args.quiet,
        save=args.save,
        savedir=args.savedir,
        pub=args.pub,
        titlestr=args.titlestr,
        showtitle=args.showtitle,
        shortlbl=args.shortlbl,
        ntor=args.ntor,
        export=args.export,
        txtname=args.txtname,
    )
