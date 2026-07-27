#!/home/jfernandes/.venv/bin/python
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Fixed categorical color order (colorblind-safe), cycled if more datasets
# than colors are given.
DEFAULT_COLORS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]


def plot_flare_harmonic(data_files: list[str], labels: list[str] = None, scale: float | list[float] = 1.0,
                         figsize: tuple = (7, 5),
                         dpi: int = 100, colors: list[str] = None, ax: plt.Axes = None,
                         marker: str = 'o', linestyle: str = '-', title: str = None,
                         xlabel: str = None, ylabel: str = None, legend: bool = True,
                         savefig: str = None, show: bool = True) -> tuple:
    """
    Plot the amplitude of the poloidal Fourier harmonics along the resonant condition
    from one or more FLARE surfmn .npz data files.

    Parameters
    ----------
    data_files : list of str
        Paths to the .npz files to plot, in the desired legend/plot order.
    labels : list of str or None
        Legend label for each file, in the same order as data_files. If None,
        each file's stem (filename without extension) is used as its label.
    scale : float or list of float
        Multiplicative factor applied to each dataset's db, equivalent to
        plotting the result of a linear simulation driven at `scale` times
        the coil current used to generate that file (see SCALING_RULES.txt).
        A single float is broadcast to all data_files; a list must match
        len(data_files) and is applied per-file. All values must be > 0.
        Default is 1.0 (no scaling).
    figsize : tuple
        Size of the figure (width, height). Default is (7, 5). Ignored if ax is given.
    dpi : int
        Dots per inch for the figure. Default is 100. Ignored if ax is given.
    colors : list of str or None
        Color for each dataset, in the same order as data_files. If None,
        uses DEFAULT_COLORS, cycling (with a warning) if there are more
        datasets than colors.
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
        Y axis label. If None, defaults to '$|\\delta B_{m/n}|$ ( G / kA )'.
    legend : bool
        Whether to display the legend. Default is True.
    savefig : str or None
        If given, path to save the figure to.
    show : bool
        Whether to call plt.show(). Default is True.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The figure and axes containing the harmonic amplitude plot.
    """

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

    # Load datasets
    datasets = [_load_flare_data(f, label) for f, label in zip(data_files, labels)]

    # Consistency check: all datasets should share the same resonant psiN grid
    psi_ref = datasets[0][0]
    for label, (psi, _) in zip(labels, datasets):
        if not np.allclose(psi, psi_ref, atol=1e-6):
            print(f"Warning: psiN_res grid mismatch detected in {label} data.")

    print("Plotting...")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    for i, (label, (psi, db)) in enumerate(zip(labels, datasets)):
        # np.abs() guards against small negative overshoot from the cubic
        # interpolation onto resonant surfaces in flare_surfmn.py; db is
        # already a magnitude.
        ax.plot(psi, np.abs(db) * scale[i], marker=marker, linestyle=linestyle, label=label,
                color=colors[i % len(colors)])

    ax.set_xlabel(xlabel if xlabel is not None else 'Normalized Poloidal Flux')
    ax.set_ylabel(ylabel if ylabel is not None else '$|\\delta B_{m/n}|$ ( G / kA )')

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


def _load_flare_data(path, label):
    """
    Helper: load a FLARE surfmn .npz file and return (psiN_res, db_res).
    """
    print(f"Loading {label} data from {path}...")
    try:
        with np.load(path) as f:
            psiN_res = f["psiN_res"]
            db_res = f["db_res"]
        print(f"{label} data loaded successfully.")
        return psiN_res, db_res
    except KeyError as e:
        raise ValueError(f"Missing key {e} in {label} file.")
    except Exception as e:
        raise ValueError(f"Failed to load {label} data: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FLARE harmonic amplitudes on resonant surfaces.")
    parser.add_argument("data_files", type=str, nargs='+', help="Paths to the surfmn .npz data files")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
                         help="Legend labels for each data file, in the same order as data_files")
    parser.add_argument("--colors", type=str, nargs='+', default=None,
                         help="Colors for each data file, in the same order as data_files")
    parser.add_argument("--scale", type=float, nargs='+', default=1.0,
                         help="Multiplicative factor(s) on db, equivalent to scaling the driving "
                              "coil current by this factor (see SCALING_RULES.txt). Single value "
                              "broadcasts to all data_files, or one value per file.")
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

    plot_flare_harmonic(
        data_files=args.data_files,
        labels=args.labels,
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
