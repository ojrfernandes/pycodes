#!/home/jfernandes/.venv/bin/python
import argparse
import numpy as np
import matplotlib.pyplot as plt

def phase_map(directory, n_tor, m_pol, d_phase, figsize=(7,5), dpi=100, levels=100, cmap='jet', fullspace=False):
    """
    Generate and plot a phase map from flare_surfmn data files in a specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing flare_surfmn output .npz files.
    n_tor : int
        Toroidal mode number.
    m_pol : int
        Poloidal mode number.
    d_phase : float
        Phase difference increment in degrees.
    figsize : tuple
        Size of the figure (width, height). Default is (7, 5).
    dpi : int
        Dots per inch for the figure. Default is 100.
    levels : int
        Number of contour levels for the plot. Default is 100.
    cmap : str
        Colormap to use for the plot. Default is 'jet'.

    Returns
    -------
    None
        Displays the phase map plot.

    """
    
    # Determine number of elements in the phase map
    n_elements = int((360 / n_tor / d_phase) + 1)
    db_map = np.zeros((n_elements, n_elements))

    # Loop over all combinations of phase differences
    for i in range(n_elements):
        phase_IL = i * d_phase
        for j in range(n_elements):
            phase_IU = j * d_phase
            filename = f'dephase_IL_{phase_IL:03d}_IU_{phase_IU:03d}.npz'
            data = directory + filename

            # Load data
            try:
                with np.load(data) as f:
                    db_matrix = f['db_matrix']
                    q_vals = f['q_vals']
                    m_values = f['m_values']
            except Exception as e:
                raise ValueError(f"Failed to load surfmn data: {e}")

            # Find index of m_pol in m_values
            if m_pol in m_values:
                idx_m = np.where(m_values == m_pol)[0][0]
            else:
                raise ValueError(f"m_pol {m_pol} not found in m_values array.")

            q_res = m_pol / n_tor
            # Find index of q_res in q_vals
            if q_res < np.min(q_vals) or q_res > np.max(q_vals):
                raise ValueError(f"q_res {q_res} is out of bounds of q_vals array.")
            else:
                idx_q = (np.abs(q_vals - q_res)).argmin()

            # Find db at m_pol
            db_mpol = db_matrix[idx_q, idx_m]

            # Store in map
            db_map[i, j] = db_mpol
            

    if fullspace:
        db_map = np.tile(db_map[:-1, :-1], (n_tor, n_tor))

    # Plotting
    plt.figure(figsize=figsize, dpi=dpi)
    plt.contourf(db_map, levels=levels, cmap=cmap)
    plt.colorbar(label='$\delta B_{' + str(m_pol) + ' / ' + str(n_tor) + '} ( G / kA )$')
    plt.xlabel(r'$\Delta\Phi_{IU}$ ( deg )')
    plt.ylabel(r'$\Delta\Phi_{IL}$ ( deg )')

    n_elements = db_map.shape[0]
    if not fullspace:
        ticks = np.arange(n_elements)
    else:
        ticks = np.arange(0, n_elements, n_elements // 9 + 1)

    tick_labels = ticks * d_phase

    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and plot a phase map from flare_surfmn data files.")
    parser.add_argument("directory", type=str, help="Path to the directory containing flare_surfmn output .npz files")
    parser.add_argument("n_tor", type=int, help="Toroidal mode number")
    parser.add_argument("m_pol", type=int, help="Poloidal mode number")
    parser.add_argument("d_phase", type=float, help="Phase difference increment in degrees")
    parser.add_argument("--figsize", type=float, nargs=2, default=(7, 5), help="Figure size (width, height)")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for the figure")
    parser.add_argument("--levels", type=int, default=100, help="Number of contour levels for the plot")
    parser.add_argument("--cmap", type=str, default='jet', help="Colormap to use in the plot")
    args = parser.parse_args()

    phase_map(
        directory=args.directory,
        n_tor=args.n_tor,
        m_pol=args.m_pol,
        d_phase=args.d_phase,
        figsize=args.figsize,
        dpi=args.dpi,
        levels=args.levels,
        cmap=args.cmap
    )