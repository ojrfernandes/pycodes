#!/home/jfernandes/.venv/bin/python
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

from phase_grid import phase_grid_size

# quantity -> (filename suffix, mode-index key, value key, current-scaling exponent)
# read from the .npz. 'bmn' reads flare_surfmn.py's dephase_*.npz directly and
# scales linearly with coil current (exponent 1); 'width'/'chirikov' read
# flare_chirikov.py's dephase_*_chirikov.npz (see flare_chirikov_batch.py to
# generate these in bulk over a whole phase-map directory) and scale as
# sqrt(current) (exponent 0.5) -- see SCALING_RULES.txt.
_QUANTITY = {
    'bmn':      (None,        'm_res',   'db_res',    1.0),
    'width':    ('_chirikov', 'm_res',   'width_res', 0.5),
    'chirikov': ('_chirikov', 'm_pairs', 'chirikov',  0.5),
}


def plot_phase_map(directory: str, n_tor: int, m_pol: int, d_phase: int, quantity: str='bmn',
                   scale: float=1.0, figsize: tuple=(7,5),
                   dpi: int=100, levels: int=100, cmap: str | mpl.colors.Colormap='jet', fullspace: bool=False,
                   vmin:float|None=None, vmax:float|None=None, cmap_key:int|None=None,
                   phase_signal: list=[-1,1], ax: plt.Axes|None=None, title: str|None=None,
                   xlabel: str|None=None, ylabel: str|None=None, cbar_label: str|None=None,
                   tick_step: int|None=None, savefig: str|None=None, show: bool=True) -> tuple:
    """
    Map and plot a two dimensional map of a resonant-surface quantity for a
    specific m/n mode from "flare_surfmn.py" (or "flare_chirikov.py") .npz
    data files in a specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing flare_surfmn/flare_chirikov output .npz files.
    n_tor : int
        Toroidal mode number.
    m_pol : int
        Poloidal mode number of the resonant surface to map (q = m_pol/n_tor).
        Must satisfy |m_pol| <= m_max used when the .npz files were
        generated, and q = m_pol/n_tor must lie within the equilibrium's
        q(psi) range. For quantity='bmn'/'width' it must appear in each
        file's 'm_res' array; for quantity='chirikov' it selects the gap
        between resonant surface m_pol and the next one out (matched
        against the lower element of each file's 'm_pairs' array, i.e.
        m_pairs[:, 0] == m_pol).
    d_phase : int
        Phase difference increment in degrees.
    quantity : str
        Which quantity to map: 'bmn' (default, |Bmn| at the resonant surface,
        from flare_surfmn.py's dephase_*.npz), 'width' (island half-width at
        the resonant surface, from flare_chirikov.py's dephase_*_chirikov.npz),
        or 'chirikov' (Chirikov overlap parameter between adjacent resonant
        surfaces, also from dephase_*_chirikov.npz). 'width'/'chirikov'
        require running flare_chirikov.py (or flare_chirikov_batch.py) on the
        directory's surfmn .npz files first.
    scale : float
        Multiplicative factor on the driving coil current, equivalent to
        plotting the result of a linear simulation driven at `scale` times
        the current used to generate the .npz files (see SCALING_RULES.txt).
        Applied as `scale` for quantity='bmn' and `sqrt(scale)` for
        quantity='width'/'chirikov'. Must be > 0. Default is 1.0 (no scaling).
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
        Phase signal for IL and IU sets respectively. Default is [-1, 1].
    ax : matplotlib.axes.Axes or None
        Axes to plot onto. If None, a new figure and axes are created.
    title : str or None
        Plot title. If None, no title is set.
    xlabel : str or None
        X axis label. If None, defaults to the coil-dependent phase label.
    ylabel : str or None
        Y axis label. If None, defaults to the coil-dependent phase label.
    cbar_label : str or None
        Colorbar label. If None, defaults to a quantity-dependent label:
        '$\\delta B_{m/n}$ ( G / kA )' for 'bmn', '$w_{m/n}$ ( $\\psi_N$ )' for
        'width', or '$\\sigma_{m/n}$' for 'chirikov'.
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
        The figure and axes containing the phase map plot.

    """

    if quantity not in _QUANTITY:
        raise ValueError(f"quantity must be one of {list(_QUANTITY)}, got {quantity!r}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    suffix, mode_key, value_key, exponent = _QUANTITY[quantity]
    suffix = suffix or ''

    # Determine number of elements in the phase map
    n_elements = phase_grid_size(n_tor, d_phase)
    db_map = np.zeros((n_elements, n_elements))
    coil = None

    # Loop over all combinations of phase differences
    for i in range(n_elements):
        phase_L = int(i * d_phase)
        for j in range(n_elements):
            phase_U = int(j * d_phase)
            file_IL = os.path.join(directory, f'dephase_IL_{phase_L:03d}_IU_{phase_U:03d}{suffix}.npz')
            file_CP = os.path.join(directory, f'dephase_CPL_{phase_L:03d}_CPU_{phase_U:03d}{suffix}.npz')

            if os.path.exists(file_IL):
                datafile = file_IL
                coil = 0
            elif os.path.exists(file_CP):
                datafile = file_CP
                coil = 1
            elif quantity != 'bmn':
                raise FileNotFoundError(
                    f"No valid {suffix}.npz file found for phases {phase_L}, {phase_U} in '{directory}'. "
                    f"quantity={quantity!r} requires flare_chirikov.py output -- run flare_chirikov.py "
                    "(or flare_chirikov_batch.py for a whole directory) on the flare_surfmn.py .npz files "
                    "there first."
                )
            else:
                raise FileNotFoundError(f"No valid .npz file found for phases {phase_L}, {phase_U}")

            # Read the value directly at the resonant surface / surface pair,
            # as precomputed by flare_surfmn.py/flare_chirikov.py
            with np.load(datafile) as f:
                modes = f[mode_key]
                values = f[value_key]

            if quantity == 'chirikov':
                match = np.where(modes[:, 0] == m_pol)[0]
            else:
                match = np.where(modes == m_pol)[0]

            if len(match) > 0:
                db_map[i, j] = values[match[0]]
            else:
                raise ValueError(
                    f"m_pol={m_pol} not found in {mode_key} for {datafile} "
                    "(check |m_pol| <= m_max and that q = m_pol/n_tor lies within "
                    "the equilibrium's q(psi) range)."
                )

    db_map *= scale ** exponent

    if fullspace:
        db_map = np.tile(db_map[:-1, :-1], (n_tor, n_tor))

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if cbar_label is None:
        if quantity == 'bmn':
            cbar_label = '$\delta B_{\,' + str(np.abs(m_pol)) + ' / ' + str(n_tor) + '}$ ( G / kA )'
        elif quantity == 'width':
            cbar_label = '$w_{\,' + str(np.abs(m_pol)) + ' / ' + str(n_tor) + '}$ ( $\psi_N$ )'
        elif quantity == 'chirikov':
            cbar_label = '$\sigma_{\,\,' + str(np.abs(m_pol)) + ' / ' + str(n_tor) + '}$'

    if cmap_key is not None:
        if vmin is None:
            vmin = db_map.min()
        if vmax is None:
            vmax = db_map.max()
        db_map = np.clip(db_map, vmin, vmax)
        cmap = mpl.colormaps.get_cmap(cmap).resampled(cmap_key)
        bounds = np.linspace(vmin, vmax, cmap_key + 1)
        contour = ax.contourf(db_map, levels=bounds, cmap=cmap)
        cbar = fig.colorbar(contour, ax=ax, boundaries=bounds, ticks=bounds, label=cbar_label)
    else:
        contour = ax.contourf(db_map, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(contour, ax=ax, label=cbar_label)

    # if logscale:
    #     positive = db_map[db_map > 0]
    #     if vmin is None:
    #         vmin = positive.min()
    #     if vmax is None:
    #         vmax = positive.max()

    #     norm = colors.PowerNorm(2, vmin=vmin, vmax=vmax)

    #     if cmap_key is None:
    #         levels = np.logspace(np.log10(vmin),
    #                             np.log10(vmax),
    #                             levels)
    #     else:
    #         levels = np.logspace(np.log10(vmin),
    #                             np.log10(vmax),
    #                             cmap_key + 1)
    
    # else:
    #     norm = colors.Normalize(vmin=vmin, vmax=vmax)

    #     if cmap_key is not None:
    #         levels = np.linspace(vmin, vmax, cmap_key + 1)

    # contour = ax.contourf(db_map, levels=levels, cmap=cmap, norm=norm)
    # cbar = fig.colorbar(contour, ax=ax, label=cbar_label)

    if xlabel is None:
        if coil == 0:
            xlabel = r'$\Phi_{IU} - \Phi_{IM}$ ( deg )'
        elif coil == 1:
            xlabel = r'$\Phi_{CPU} - \Phi_{CPM}$ ( deg )'
    if ylabel is None:
        if coil == 0:
            ylabel = r'$\Phi_{IL} - \Phi_{IM}$ ( deg )'
        elif coil == 1:
            ylabel = r'$\Phi_{CPL} - \Phi_{CPM}$ ( deg )'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    n_elements = db_map.shape[0]
    if tick_step is None:
        tick_step = 1 if not fullspace else n_elements // 9 + 1

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
    parser = argparse.ArgumentParser(description="Generate and plot a phase map from flare_surfmn data files.")
    parser.add_argument("directory", type=str, help="Path to the directory containing flare_surfmn output .npz files")
    parser.add_argument("n_tor", type=int, help="Toroidal mode number")
    parser.add_argument("m_pol", type=int, help="Poloidal mode number")
    parser.add_argument("d_phase", type=int, help="Phase difference increment in degrees")
    parser.add_argument("--quantity", type=str, default='bmn', choices=list(_QUANTITY),
                         help="Quantity to map: 'bmn' (|Bmn|, default), 'width' (island half-width), "
                              "or 'chirikov' (Chirikov overlap parameter). 'width'/'chirikov' require "
                              "flare_chirikov.py output (dephase_*_chirikov.npz) in the directory.")
    parser.add_argument("--scale", type=float, default=1.0,
                         help="Multiplicative factor on the driving coil current, applied as scale for "
                              "'bmn' or sqrt(scale) for 'width'/'chirikov' (see SCALING_RULES.txt).")
    parser.add_argument("--figsize", type=float, nargs=2, default=(7, 5), help="Figure size (width, height)")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for the figure")
    parser.add_argument("--levels", type=int, default=100, help="Number of contour levels for the plot")
    parser.add_argument("--cmap", type=str, default='jet', help="Colormap to use in the plot")
    parser.add_argument("--fullspace", action='store_true', help="If set, replicate the phase map to cover full 360 degrees")
    parser.add_argument("--phase_signal", type=int, nargs=2, default=[-1,1], help="Phase signal for IL and IU sets respectively. Default is [-1, 1]")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--xlabel", type=str, default=None, help="X axis label (default: coil-dependent phase label)")
    parser.add_argument("--ylabel", type=str, default=None, help="Y axis label (default: coil-dependent phase label)")
    parser.add_argument("--cbar-label", type=str, default=None, help="Colorbar label")
    parser.add_argument("--tick-step", type=int, default=None, help="Spacing between displayed ticks when --fullspace is set")
    parser.add_argument("--savefig", type=str, default=None, help="Path to save the figure. If None, figure is not saved.")
    parser.add_argument("--no-show", action="store_false", dest="show", help="Do not display the figure interactively")
    args = parser.parse_args()

    plot_phase_map(
        directory=args.directory,
        n_tor=args.n_tor,
        m_pol=args.m_pol,
        d_phase=args.d_phase,
        quantity=args.quantity,
        scale=args.scale,
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
