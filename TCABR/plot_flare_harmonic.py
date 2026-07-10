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


def plot_flare_harmonic(data_files: list[str], labels: list[str] = None, figsize: tuple = (7, 5),
                         dpi: int = 100, colors: list[str] = None) -> None:
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
    figsize : tuple
        Size of the figure (width, height). Default is (7, 5).
    dpi : int
        Dots per inch for the figure. Default is 100.
    colors : list of str or None
        Color for each dataset, in the same order as data_files. If None,
        uses DEFAULT_COLORS, cycling (with a warning) if there are more
        datasets than colors.

    Returns
    -------
    None
        Displays the plot of the harmonic amplitudes.
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

    # Load datasets
    datasets = [_load_flare_data(f, label) for f, label in zip(data_files, labels)]

    # Consistency check: all datasets should share the same resonant psiN grid
    psi_ref = datasets[0][0]
    for label, (psi, _) in zip(labels, datasets):
        if not np.allclose(psi, psi_ref, atol=1e-6):
            print(f"Warning: psiN_res grid mismatch detected in {label} data.")

    print("Plotting...")
    plt.figure(figsize=figsize, dpi=dpi)
    for i, (label, (psi, db)) in enumerate(zip(labels, datasets)):
        # np.abs() guards against small negative overshoot from the cubic
        # interpolation onto resonant surfaces in flare_surfmn.py; db is
        # already a magnitude.
        plt.plot(psi, np.abs(db), 'o-', label=label, color=colors[i % len(colors)])
    plt.xlabel('Normalized Poloidal Flux')
    plt.ylabel('$|\\delta B_{m/n}|$ ( G / kA )')
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    parser.add_argument("--figsize", type=float, nargs=2, default=(7, 5), help="Figure size (width, height)")
    parser.add_argument("--dpi", type=int, default=100, help="Dots per inch for the figure")
    args = parser.parse_args()

    plot_flare_harmonic(
        data_files=args.data_files,
        labels=args.labels,
        figsize=args.figsize,
        dpi=args.dpi,
        colors=args.colors,
    )
