#!/home/jfernandes/.venv/bin/python
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_flare_surfmn(data: str, res_line: bool=True, mark_res: bool=True,
                      mark_color: str='k', scale: float=1.0, figsize: tuple=(7,5), dpi:int =100,
                      levels:int =100, cmap:str ='jet', ax: plt.Axes|None=None,
                      vmin:float|None=None, vmax:float|None=None, cmap_key:int|None=None,
                      title: str|None=None, xlabel: str|None=None, ylabel: str|None=None,
                      cbar_label: str|None=None, xlim: tuple|None=None,
                      res_line_color: str='k', res_line_style: str='--',
                      mark_marker: str='x', mark_size: float=40, mark_linewidth: float=1.5,
                      legend: bool=False, cbar: bool=True,
                      savefig: str|None=None, show: bool=True) -> tuple:
    """
    Plot the surfmn spectra from the given .npz data file.

    Parameters
    ----------
    data : str
        Path to the .npz file containing surfmn data.
    scale : float
        Multiplicative factor applied to db_matrix, equivalent to plotting
        the result of a linear simulation driven at `scale` times the coil
        current used to generate `data` (see SCALING_RULES.txt). Must be > 0.
        Default is 1.0 (no scaling).
    res_line : bool
        Whether to plot the resonance line. Default is True.
    mark_res : bool
        Whether to mark the rational (resonant) surfaces with an 'x'.
        Default is True.
    mark_color : str
        Color of the 'x' markers on the rational surfaces. Default is 'k'.
    figsize : tuple
        Size of the figure (width, height). Default is (7, 5). Ignored if ax is given.
    dpi : int
        Dots per inch for the figure. Default is 100. Ignored if ax is given.
    levels : int
        Number of contour levels. Default is 100.
    cmap : str
        Colormap to use for the plot. Default is 'jet'.
    ax : matplotlib.axes.Axes or None
        Axes to plot onto. If None, a new figure and axes are created.
    vmin : float or None
        Minimum value for the color scale. If None, determined from data.
    vmax : float or None
        Maximum value for the color scale. If None, determined from data.
    cmap_key : int or None
        If given, the colormap is resampled to this number of colors.
    title : str or None
        Plot title. If None, no title is set.
    xlabel : str or None
        X axis label. If None, defaults to 'Poloidal Mode Number'.
    ylabel : str or None
        Y axis label. If None, defaults to 'Normalized Poloidal Flux'.
    cbar_label : str or None
        Colorbar label. If None, defaults to '$\\delta B_{m\\,n}$ ( G / kA )'.
    xlim : tuple or None
        (xmin, xmax) for the x axis. If None and res_line=True, defaults to
        (min(m_mesh), max(m_mesh)); otherwise left at the automatic limits.
    res_line_color : str
        Color of the resonance line. Default is 'k'.
    res_line_style : str
        Line style of the resonance line. Default is '--'.
    mark_marker : str
        Marker style for the rational-surface markers. Default is 'x'.
    mark_size : float
        Marker size for the rational-surface markers. Default is 40.
    mark_linewidth : float
        Marker linewidth for the rational-surface markers. Default is 1.5.
    legend : bool
        Whether to display the legend (rational surfaces marker). Default is False.
    cbar : bool
        Whether to draw a colorbar attached to `ax`. Set False when overlaying
        several calls on subplots that should share a single external colorbar
        (build it from the returned `contour` instead). Default is True.
    savefig : str or None
        If given, path to save the figure to.
    show : bool
        Whether to call plt.show(). Default is True.

    Returns
    -------
    fig, ax, contour : matplotlib Figure, Axes, and QuadContourSet
        The figure and axes containing the surfmn spectra plot, plus the
        contourf mappable (usable to build a shared colorbar when cbar=False).

    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    # Load data
    print(f"Loading surfmn data from {data}...")
    try:
        with np.load(data) as f:
            n_tor = int(f['n_tor'])
            psiN_values = f['psiN_values']
            m_mesh = f['m_mesh']
            psiN_mesh = f['psiN_mesh']
            db_matrix = f['db_matrix']
            q_vals = f['q_vals']
            psiN_res = f['psiN_res']
            m_res = f['m_res']
    except KeyError as e:
        raise ValueError(f"Missing expected data in the file: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load surfmn data: {e}")
    print("Data loaded successfully.")

    db_matrix = db_matrix * scale

    # Plotting
    print("Plotting...")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if cbar_label is None:
        cbar_label = '$\delta B_{m\,n}$ ( G / kA )'

    if cmap_key is not None:
        db_matrix = np.clip(db_matrix, vmin, vmax)
        cmap = mpl.colormaps.get_cmap(cmap).resampled(cmap_key)
        bounds = np.linspace(vmin, vmax, cmap_key + 1)
        contour = ax.contourf(m_mesh, psiN_mesh, db_matrix, levels=bounds, cmap=cmap)
        if cbar:
            fig.colorbar(contour, ax=ax, boundaries=bounds, ticks=bounds, label=cbar_label)
    else:
        contour = ax.contourf(m_mesh, psiN_mesh, db_matrix, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        if cbar:
            fig.colorbar(contour, ax=ax, label=cbar_label)

    if res_line:
        ax.plot(n_tor * q_vals, psiN_values, color=res_line_color, linestyle=res_line_style)
        if xlim is None:
            xlim = (np.min(m_mesh), np.max(m_mesh))
    if xlim is not None:
        ax.set_xlim(*xlim)

    if mark_res:
        ax.scatter(m_res, psiN_res, marker=mark_marker, color=mark_color,
                   s=mark_size, linewidths=mark_linewidth, zorder=5, label='Rational surfaces')

    ax.set_xlabel(xlabel if xlabel is not None else 'Poloidal mode number')
    ax.set_ylabel(ylabel if ylabel is not None else 'Normalized poloidal flux')

    if title is not None:
        ax.set_title(title)

    if legend and mark_res:
        ax.legend()

    fig.tight_layout()

    if savefig is not None:
        fig.savefig(savefig, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()

    return fig, ax, contour

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FLARE surfmn spectra.")
    parser.add_argument("data", type=str, help="Path to the surfmn data file")
    parser.add_argument("--no-res-line", action="store_false", dest="res_line",
                        help="Disable resonance line plotting")
    parser.add_argument("--no-mark-res", action="store_false", dest="mark_res",
                        help="Disable 'x' markers on rational (resonant) surfaces")
    parser.add_argument("--mark-color", type=str, default="k",
                        help="Color of the 'x' markers on rational surfaces")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Multiplicative factor on db_matrix, equivalent to scaling the "
                             "driving coil current by this factor (see SCALING_RULES.txt)")
    parser.add_argument("--figsize", type=float, nargs=2, default=(7, 5),
                        help="Figure size (width height)")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for the figure")
    parser.add_argument("--levels", type=int, default=100, help="Number of contour levels")
    parser.add_argument("--cmap", type=str, default="jet", help="Colormap to use")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--xlabel", type=str, default=None, help="X axis label")
    parser.add_argument("--ylabel", type=str, default=None, help="Y axis label")
    parser.add_argument("--cbar-label", type=str, default=None, help="Colorbar label")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, help="X axis limits (xmin xmax)")
    parser.add_argument("--res-line-color", type=str, default="k", help="Color of the resonance line")
    parser.add_argument("--res-line-style", type=str, default="--", help="Line style of the resonance line")
    parser.add_argument("--mark-marker", type=str, default="x", help="Marker style for rational-surface markers")
    parser.add_argument("--mark-size", type=float, default=40, help="Marker size for rational-surface markers")
    parser.add_argument("--mark-linewidth", type=float, default=1.5, help="Marker linewidth for rational-surface markers")
    parser.add_argument("--no-legend", action="store_false", dest="legend", help="Do not display the legend")
    parser.add_argument("--no-cbar", action="store_false", dest="cbar", help="Do not draw a colorbar")
    parser.add_argument("--savefig", type=str, default=None, help="Path to save the figure. If None, figure is not saved.")
    parser.add_argument("--no-show", action="store_false", dest="show", help="Do not display the figure interactively")

    args = parser.parse_args()
    plot_flare_surfmn(
        data=args.data,
        res_line=args.res_line,
        mark_res=args.mark_res,
        mark_color=args.mark_color,
        scale=args.scale,
        figsize=args.figsize,
        dpi=args.dpi,
        levels=args.levels,
        cmap=args.cmap,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        cbar_label=args.cbar_label,
        xlim=args.xlim,
        res_line_color=args.res_line_color,
        res_line_style=args.res_line_style,
        mark_marker=args.mark_marker,
        mark_size=args.mark_size,
        mark_linewidth=args.mark_linewidth,
        legend=args.legend,
        cbar=args.cbar,
        savefig=args.savefig,
        show=args.show,
    )
