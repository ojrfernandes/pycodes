#!/home/jfernandes/.venv/bin/python
import argparse
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from phase_grid import phase_grid_size

# quantity -> fn(data, threshold) -> percentage of field lines satisfying the
# quantity's threshold comparison. fpgen output columns (see
# eval_footprint_area.py/plot_footprint.py): 0=R, 1=Z, 2=phi,
# 3=connection length, 4=psi_min, 5=turns.
_FTPT_QUANTITY = {
    'turns_frac': lambda data, thr: np.mean(data[:, 5] >= thr) * 100.0,
    'psin_frac':  lambda data, thr: np.mean(data[:, 4] <= thr) * 100.0,
    'cl_fac':     lambda data, thr: np.mean(data[:, 3] >= thr) * 100.0,
    # H = 1 / (connection_length * psiN_min)
    'H_frac':      lambda data, thr: np.mean( (1.0 / (data[:, 3] * data[:, 4]) - float(np.nanmin(1.0 / (data[:, 3] * data[:, 4])))) / (float(np.nanmax(1.0 / (data[:, 3] * data[:, 4]))) - float(np.nanmin(1.0 / (data[:, 3] * data[:, 4])))) <= thr) * 100.0,
}


def plot_footprint_phase_map(directory: str, n_tor: int, d_phase: int, quantity: str = 'turns_frac',
                              threshold: float = 10.0, figsize: tuple = (7, 5), dpi: int = 100,
                              levels: int = 100, cmap: str = 'jet', fullspace: bool = False,
                              vmin: float | None = None, vmax: float | None = None, cmap_key: int | None = None,
                              phase_signal: list = [-1, 1], ax: plt.Axes | None = None, title: str | None = None,
                              xlabel: str | None = None, ylabel: str | None = None, cbar_label: str | None = None,
                              tick_step: int | None = None, savefig: str | None = None, show: bool = True) -> tuple:
    """
    Map and plot a two dimensional phase map of a footprint-derived quantity
    from fpgen_phase_map.py's dephase_*_ftpt.dat output files in a specified
    directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing fpgen_phase_map.py output _ftpt.dat files.
    n_tor : int
        Toroidal mode number.
    d_phase : int
        Phase difference increment in degrees.
    quantity : str
        Which quantity to map: 'turns_frac' (default, percentage of field
        lines with toroidal turn count >= threshold), 'psin_frac'
        (percentage of field lines with minimum psiN <= threshold),
        'cl_fac' (percentage of field lines with connection length >=
        threshold), or 'H_frac' (percentage of field lines with
        H = 1/(connection_length * psiN_min) >= threshold).
    threshold : float
        Threshold value used by `quantity`. Default is 10.0.
    figsize : tuple
        Size of the figure (width, height). Default is (7, 5). Ignored if ax is given.
    dpi : int
        Dots per inch for the figure. Default is 100. Ignored if ax is given.
    levels : int
        Number of contour levels for the plot. Default is 100.
    cmap : str
        Colormap to use for the plot. Default is 'jet'.
    fullspace : bool
        If True, replicate the phase map to cover the full 360 degrees. Default is False.
    vmin : float or None
        Minimum value for the color scale. If None, determined from data.
    vmax : float or None
        Maximum value for the color scale. If None, determined from data.
    cmap_key : int or None
        If given, the colormap is resampled to this number of colors.
    phase_signal : list of int
        Phase signal for IL and IU (or CPL/CPU) sets respectively. Default is [-1, 1].
    ax : matplotlib.axes.Axes or None
        Axes to plot onto. If None, a new figure and axes are created.
    title : str or None
        Plot title. If None, no title is set.
    xlabel : str or None
        X axis label. If None, defaults to the coil-dependent phase label.
    ylabel : str or None
        Y axis label. If None, defaults to the coil-dependent phase label.
    cbar_label : str or None
        Colorbar label. If None, defaults to a quantity/threshold-dependent label.
    tick_step : int or None
        Spacing between displayed ticks when fullspace=True. If None, defaults to
        the current heuristic (n_elements // 9 + 1).
    savefig : str or None
        If given, path to save the figure to.
    show : bool
        Whether to call plt.show(). Default is True.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The figure and axes containing the footprint phase map plot.
    """
    if quantity not in _FTPT_QUANTITY:
        raise ValueError(f"quantity must be one of {list(_FTPT_QUANTITY)}, got {quantity!r}")
    compare = _FTPT_QUANTITY[quantity]

    n_elements = phase_grid_size(n_tor, d_phase)
    db_map = np.zeros((n_elements, n_elements))
    coil = None

    for i in range(n_elements):
        phase_L = int(i * d_phase)
        for j in range(n_elements):
            phase_U = int(j * d_phase)
            file_IL = os.path.join(directory, f'dephase_IL_{phase_L:03d}_IU_{phase_U:03d}_ftpt.dat')
            file_CP = os.path.join(directory, f'dephase_CPL_{phase_L:03d}_CPU_{phase_U:03d}_ftpt.dat')

            if os.path.exists(file_IL):
                datafile = file_IL
                coil = 0
            elif os.path.exists(file_CP):
                datafile = file_CP
                coil = 1
            else:
                raise FileNotFoundError(
                    f"No valid _ftpt.dat file found for phases {phase_L}, {phase_U} in '{directory}'. "
                    "Run fpgen_phase_map.py on this directory first."
                )

            data = np.loadtxt(datafile)
            db_map[i, j] = compare(data, threshold)

    if fullspace:
        db_map = np.tile(db_map[:-1, :-1], (n_tor, n_tor))

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if cbar_label is None:
        if quantity == 'turns_frac':
            cbar_label = f'% field lines with turns $\\geq$ {threshold:g}'
        elif quantity == 'psin_frac':
            cbar_label = f'% field lines with $\\psi_{{N,min}} \\leq$ {threshold:g}'
        elif quantity == 'cl_fac':
            cbar_label = f'% field lines with $L_c \\geq$ {threshold:g}'
        elif quantity == 'H_frac':
            cbar_label = f'% field lines with $H \\geq$ {threshold:g}'

    if cmap_key is not None:
        db_map = np.clip(db_map, vmin, vmax)
        cmap = mpl.colormaps.get_cmap(cmap).resampled(cmap_key)
        bounds = np.linspace(vmin, vmax, cmap_key + 1)
        contour = ax.contourf(db_map, levels=bounds, cmap=cmap)
        cbar = fig.colorbar(contour, ax=ax, boundaries=bounds, ticks=bounds, label=cbar_label)
    else:
        contour = ax.contourf(db_map, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(contour, ax=ax, label=cbar_label)

    if xlabel is None:
        if coil == 0:
            xlabel = r'$\Delta\Phi_{IU}$ ( deg )'
        elif coil == 1:
            xlabel = r'$\Delta\Phi_{CPU}$ ( deg )'
    if ylabel is None:
        if coil == 0:
            ylabel = r'$\Delta\Phi_{IL}$ ( deg )'
        elif coil == 1:
            ylabel = r'$\Delta\Phi_{CPL}$ ( deg )'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    n_elements = db_map.shape[0]
    if not fullspace:
        ticks = np.arange(n_elements)
    else:
        if tick_step is None:
            tick_step = n_elements // 9 + 1
        ticks = np.arange(0, n_elements, tick_step)

    tick_labels = ticks * d_phase

    ax.set_xticks(ticks, (tick_labels * phase_signal[1]).astype(int))
    ax.set_yticks(ticks, (tick_labels * phase_signal[0]).astype(int))

    if savefig is not None:
        fig.savefig(savefig, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and plot a footprint phase map from fpgen_phase_map.py data files.")
    parser.add_argument("directory", type=str, help="Path to the directory containing fpgen_phase_map.py output _ftpt.dat files")
    parser.add_argument("n_tor", type=int, help="Toroidal mode number")
    parser.add_argument("d_phase", type=int, help="Phase difference increment in degrees")
    parser.add_argument("--quantity", type=str, default='turns_frac', choices=list(_FTPT_QUANTITY),
                         help="Quantity to map: 'turns_frac' (default, %% field lines with turns >= threshold), "
                              "'psin_frac' (%% field lines with psiN_min <= threshold), "
                              "'cl_fac' (%% field lines with connection length >= threshold), or "
                              "'H_frac' (%% field lines with H = 1/(connection_length * psiN_min) >= threshold).")
    parser.add_argument("--threshold", type=float, default=10.0, help="Threshold value used by --quantity. Default is 10.0.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(7, 5), help="Figure size (width, height)")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for the figure")
    parser.add_argument("--levels", type=int, default=100, help="Number of contour levels for the plot")
    parser.add_argument("--cmap", type=str, default='jet', help="Colormap to use in the plot")
    parser.add_argument("--fullspace", action='store_true', help="If set, replicate the phase map to cover full 360 degrees")
    parser.add_argument("--phase_signal", type=int, nargs=2, default=[-1, 1], help="Phase signal for IL/CPL and IU/CPU sets respectively. Default is [-1, 1]")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--xlabel", type=str, default=None, help="X axis label (default: coil-dependent phase label)")
    parser.add_argument("--ylabel", type=str, default=None, help="Y axis label (default: coil-dependent phase label)")
    parser.add_argument("--cbar-label", type=str, default=None, help="Colorbar label")
    parser.add_argument("--tick-step", type=int, default=None, help="Spacing between displayed ticks when --fullspace is set")
    parser.add_argument("--savefig", type=str, default=None, help="Path to save the figure. If None, figure is not saved.")
    parser.add_argument("--no-show", action="store_false", dest="show", help="Do not display the figure interactively")
    args = parser.parse_args()

    plot_footprint_phase_map(
        directory=args.directory,
        n_tor=args.n_tor,
        d_phase=args.d_phase,
        quantity=args.quantity,
        threshold=args.threshold,
        figsize=args.figsize,
        dpi=args.dpi,
        levels=args.levels,
        cmap=args.cmap,
        fullspace=args.fullspace,
        phase_signal=args.phase_signal,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        cbar_label=args.cbar_label,
        tick_step=args.tick_step,
        savefig=args.savefig,
        show=args.show,
    )
