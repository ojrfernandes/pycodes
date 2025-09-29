#!/home/jfernandes/.venv/bin/python
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_flare_surfmn(data, res_line=True, figsize=(7,5), dpi=100, levels=100, cmap='jet'):
    """
    Plot the surfmn spectra from the given .npz data file.

    Parameters
    ----------
    data : str
        Path to the .npz file containing surfmn data.
    res_line : bool
        Whether to plot the resonance line. Default is True.
    figsize : tuple
        Size of the figure (width, height). Default is (7, 5).
    dpi : int
        Dots per inch for the figure. Default is 100.
    levels : int
        Number of contour levels. Default is 100.
    cmap : str
        Colormap to use for the plot. Default is 'jet'.

    Returns
    -------
    None
        Displays the plot of the surfmn spectra.
        
    """
    # Load data
    print("Loading surfmn data from {}...".format(data))
    try:
        with np.load(data) as f:
            n_tor = f['n_tor']
            psiN_values = f['psiN_values']
            m_mesh = f['m_mesh']
            psiN_mesh = f['psiN_mesh']
            db_matrix = f['db_matrix']
            q_vals = f['q_vals']
    except Exception as e:
        raise ValueError(f"Failed to load surfmn data: {e}")
    print("Data loaded successfully.")
    
    # Plotting
    print("Plotting...")
    plt.figure(figsize=figsize, dpi=dpi)
    plt.contourf(m_mesh, psiN_mesh, db_matrix, levels=levels, cmap=cmap)
    plt.colorbar(label='$\delta B_{m\,n}$ ( G / kA )')
    if res_line:
        plt.plot(n_tor * q_vals, psiN_values, 'k--')
        plt.xlim(np.min(m_mesh), np.max(m_mesh))
    plt.xlabel('Poloidal Mode Number')
    plt.ylabel('Normalized Poloidal Flux')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FLARE surfmn spectra.")
    parser.add_argument("data", type=str, help="Path to the surfmn data file")
    parser.add_argument("--res_line", type=bool, default=True, help="Enable resonance line")
    parser.add_argument("--figsize", type=float, nargs=2, default=(7, 5), help="Figure size (width, height)")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for the figure")
    parser.add_argument("--levels", type=int, default=100, help="Number of contour levels")
    parser.add_argument("--cmap", type=str, default='jet', help="Colormap to use")
    args = parser.parse_args()

    plot_flare_surfmn(
        data=args.data,
        res_line=args.res_line,
        figsize=args.figsize,
        dpi=args.dpi,
        levels=args.levels,
        cmap=args.cmap
    )
