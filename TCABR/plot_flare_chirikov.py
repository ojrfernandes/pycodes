#!/home/jfernandes/.venv/bin/python
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_flare_harmonic import DEFAULT_COLORS


def plot_flare_chirikov(data_files: list[str], labels: list[str] = None, quantity: str = 'chirikov',
                         scale: float | list[float] = 1.0,
                         figsize: tuple = (7, 5), dpi: int = 100, colors: list[str] = None,
                         ax: plt.Axes = None, marker: str = 'o', linestyle: str = '-', title: str = None,
                         xlabel: str = None, ylabel: str = None, legend: bool = True,
                         savefig: str = None, show: bool = True) -> tuple:
    """
    Plot the Chirikov overlap parameter (or, alternatively, the underlying
    island half-widths) from one or more flare_chirikov.py .npz data files.
    Linear x-y line/marker plot, matching IDL's plot_bmn.pro (/chi or
    /width modes) -- not a contour plot.

    Parameters
    ----------
    data_files : list of str
        Paths to the flare_chirikov.py .npz files to plot, in legend order.
    labels : list of str or None
        Legend label for each file. If None, each file's stem is used.
    quantity : str
        Either 'chirikov' (plots `chirikov` vs `psimid`, default) or
        'width' (plots `width_res` vs `psiN_res`).
    scale : float or list of float
        Multiplicative factor on the driving coil current, equivalent to
        plotting the result of a linear simulation driven at `scale` times
        the current used to generate that file (see SCALING_RULES.txt).
        Both `quantity='chirikov'` and `quantity='width'` scale as
        `sqrt(scale)` (island width ~ sqrt(Bmn), and chirikov is a mean of
        two widths over a fixed spacing). A single float is broadcast to
        all data_files; a list must match len(data_files). All values must
        be > 0. Default is 1.0 (no scaling).
    figsize : tuple
        Size of the figure (width, height). Default is (7, 5). Ignored if ax is given.
    dpi : int
        Dots per inch for the figure. Default is 100. Ignored if ax is given.
    colors : list of str or None
        Color for each dataset. If None, uses DEFAULT_COLORS (from
        plot_flare_harmonic.py), cycling with a warning if needed.
    ax : matplotlib.axes.Axes or None
        Axes to plot onto. If None, a new figure and axes are created.
    marker : str
        Marker style applied to every series. Default is 'o'.
    linestyle : str
        Line style applied to every series. Default is '-'.
    title : str or None
        Plot title. If None, no title is set.
    xlabel : str or None
        X axis label. If None, defaults to 'Normalized Poloidal Flux'.
    ylabel : str or None
        Y axis label. If None, defaults based on `quantity`.
    legend : bool
        Whether to display the legend. Default is True.
    savefig : str or None
        If given, path to save the figure to.
    show : bool
        Whether to call plt.show(). Default is True.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if quantity not in ('chirikov', 'width'):
        raise ValueError(f"quantity must be 'chirikov' or 'width', got {quantity!r}.")

    if not data_files:
        raise ValueError("At least one data file must be provided.")

    if labels is None:
        labels = [Path(f).stem for f in data_files]
    elif len(labels) != len(data_files):
        raise ValueError(f"Got {len(data_files)} data files but {len(labels)} labels; they must match.")

    if colors is None:
        colors = DEFAULT_COLORS
    if len(data_files) > len(colors):
        print(f"Warning: {len(data_files)} datasets requested but only {len(colors)} "
              "default colors available; colors will repeat.")

    if isinstance(scale, (int, float)):
        scale = [scale] * len(data_files)
    elif len(scale) == 1:
        scale = list(scale) * len(data_files)
    elif len(scale) != len(data_files):
        raise ValueError(f"Got {len(data_files)} data files but {len(scale)} scale values; they must match.")
    if any(s <= 0 for s in scale):
        raise ValueError(f"All scale values must be > 0, got {scale}")

    datasets = [_load_flare_chirikov(f, label, quantity) for f, label in zip(data_files, labels)]

    print("Plotting...")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    for i, (label, (x, y)) in enumerate(zip(labels, datasets)):
        ax.plot(x, y * np.sqrt(scale[i]), marker=marker, linestyle=linestyle, label=label,
                color=colors[i % len(colors)])

    ax.set_xlabel(xlabel if xlabel is not None else 'Normalized Poloidal Flux')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif quantity == 'chirikov':
        ax.set_ylabel('Chirikov Parameter')
    else:
        ax.set_ylabel('Island Width ($\\psi_N$)')

    if title is not None:
        ax.set_title(title)

    if legend:
        ax.legend()

    fig.tight_layout()

    if savefig is not None:
        fig.savefig(savefig, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()

    return fig, ax


def _load_flare_chirikov(path, label, quantity):
    """
    Helper: load a flare_chirikov.py .npz file and return (x, y) for the
    requested quantity: ('psimid', 'chirikov') or ('psiN_res', 'width_res').
    """
    print(f"Loading {label} data from {path}...")
    x_key, y_key = ('psimid', 'chirikov') if quantity == 'chirikov' else ('psiN_res', 'width_res')
    try:
        with np.load(path) as f:
            x = f[x_key]
            y = f[y_key]
        print(f"{label} data loaded successfully.")
        return x, y
    except KeyError as e:
        raise ValueError(f"Missing key {e} in {label} file.")
    except Exception as e:
        raise ValueError(f"Failed to load {label} data: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the FLARE Chirikov overlap parameter (or island widths).")
    parser.add_argument("data_files", type=str, nargs='+', help="Paths to the flare_chirikov.py .npz data files")
    parser.add_argument("--quantity", type=str, choices=['chirikov', 'width'], default='chirikov',
                         help="Plot the Chirikov parameter vs psimid, or the island widths vs psiN_res")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
                         help="Legend labels for each data file, in the same order as data_files")
    parser.add_argument("--colors", type=str, nargs='+', default=None,
                         help="Colors for each data file, in the same order as data_files")
    parser.add_argument("--scale", type=float, nargs='+', default=1.0,
                         help="Multiplicative factor(s) on the driving coil current; both quantities "
                              "scale as sqrt(scale) (see SCALING_RULES.txt). Single value broadcasts "
                              "to all data_files, or one value per file.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(7, 5), help="Figure size (width, height)")
    parser.add_argument("--dpi", type=int, default=100, help="Dots per inch for the figure")
    parser.add_argument("--marker", type=str, default="o", help="Marker style applied to every series")
    parser.add_argument("--linestyle", type=str, default="-", help="Line style applied to every series")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--xlabel", type=str, default=None, help="X axis label")
    parser.add_argument("--ylabel", type=str, default=None, help="Y axis label")
    parser.add_argument("--no-legend", action="store_false", dest="legend", help="Do not display the legend")
    parser.add_argument("--savefig", type=str, default=None, help="Path to save the figure. If None, figure is not saved.")
    parser.add_argument("--no-show", action="store_false", dest="show", help="Do not display the figure interactively")
    args = parser.parse_args()

    plot_flare_chirikov(
        data_files=args.data_files,
        labels=args.labels,
        quantity=args.quantity,
        scale=args.scale,
        figsize=args.figsize,
        dpi=args.dpi,
        colors=args.colors,
        marker=args.marker,
        linestyle=args.linestyle,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        legend=args.legend,
        savefig=args.savefig,
        show=args.show,
    )
