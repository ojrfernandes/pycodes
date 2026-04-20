#!/home/jfernandes/.venv/bin/python
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_phase_map(directory: str, n_tor: int, m_pol: int, d_phase: int, figsize: tuple=(7,5), dpi: int=100, levels: int=100, cmap: str='jet', fullspace: bool=False, phase_signal: list=[-1,1]) -> None:
    """
    Map and plot a two dimensional map of the magnetic perturbation amplitude for a specific m/n mode from "flare_surfmn.py" .npz data files in a specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing flare_surfmn output .npz files.
    n_tor : int
        Toroidal mode number.
    m_pol : int
        Poloidal mode number.
    d_phase : int
        Phase difference increment in degrees.
    figsize : tuple
        Size of the figure (width, height). Default is (7, 5).
    dpi : int
        Dots per inch for the figure. Default is 100.
    levels : int
        Number of contour levels for the plot. Default is 100.
    cmap : str
        Colormap to use for the plot. Default is 'jet'.
    fullspace : bool
        If True, replicate the phase map to cover the full 360 degrees. Default is False.
    phase_signal : list of int
        Phase signal for IL and IU sets respectively. Default is [-1, 1].   

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
        phase_L = int(i * d_phase)
        for j in range(n_elements):
            phase_U = int(j * d_phase)
            file_IL = os.path.join(directory, f'dephase_IL_{phase_L:03d}_IU_{phase_U:03d}.npz')
            file_CP = os.path.join(directory, f'dephase_CPL_{phase_L:03d}_CPU_{phase_U:03d}.npz')

            if os.path.exists(file_IL):
                datafile = file_IL
                coil = 0
            elif os.path.exists(file_CP):
                datafile = file_CP
                coil = 1
            else:
                raise FileNotFoundError(f"No valid .npz file found for phases {phase_L}, {phase_U}")

            # Load data
            if not os.path.exists(datafile):
                db_map[i, j] = np.nan
                print(f"Warning: file {datafile} does not exist.")
                continue
            try:
                with np.load(datafile) as f:
                    db_matrix = f["db_matrix"]
                    q_vals = f["q_vals"]
                    m_values = f["m_values"]
            except Exception as e:
                print(f"Warning: failed to load {datafile}: {e}")
                db_map[i, j] = np.nan
                continue

            # Find index of m_pol in m_values
            if m_pol in m_values:
                idx_m = np.where(m_values == m_pol)[0][0]
            else:
                raise ValueError(f"m_pol {m_pol} not found in m_values array.")

            # Find index of q_res in q_vals
            q_res = m_pol / n_tor
            if q_res <= np.min(q_vals) or q_res >= np.max(q_vals):
                db_mpol = np.nan
                print(f"q_res {q_res} = {m_pol} / {n_tor} is out of bounds of q_vals array.")
            else:
                idx_q = (np.abs(q_vals - q_res)).argmin()
                db_mpol = db_matrix[idx_q, idx_m]

            # Store in map
            db_map[i, j] = db_mpol
            

    if fullspace:
        db_map = np.tile(db_map[:-1, :-1], (n_tor, n_tor))

    # Plotting
    plt.figure(figsize=figsize, dpi=dpi)
    contour = plt.contourf(db_map, levels=levels, cmap=cmap)
    plt.colorbar(contour, label='$\delta B_{' + str(np.abs(m_pol)) + ' / ' + str(n_tor) + '}$ ( G / kA )')
    if coil == 0:
        plt.xlabel(r'$\Delta\Phi_{IU}$ ( deg )')
        plt.ylabel(r'$\Delta\Phi_{IL}$ ( deg )')
    if coil == 1:
        plt.xlabel(r'$\Delta\Phi_{CPU}$ ( deg )')
        plt.ylabel(r'$\Delta\Phi_{CPL}$ ( deg )')

    n_elements = db_map.shape[0]
    if not fullspace:
        ticks = np.arange(n_elements)
    else:
        ticks = np.arange(0, n_elements, n_elements // 9 + 1)

    tick_labels = ticks * d_phase

    plt.xticks(ticks, (tick_labels * phase_signal[1]).astype(int))
    plt.yticks(ticks, (tick_labels * phase_signal[0]).astype(int))

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
    parser.add_argument("--fullspace", action='store_true', help="If set, replicate the phase map to cover full 360 degrees")
    parser.add_argument("--phase_signal", type=int, nargs=2, default=[-1,1], help="Phase signal for IL and IU sets respectively. Default is [-1, 1]")
    args = parser.parse_args()

    plot_phase_map(
        directory=args.directory,
        n_tor=args.n_tor,
        m_pol=args.m_pol,
        d_phase=args.d_phase,
        figsize=args.figsize,
        dpi=args.dpi,
        levels=args.levels,
        cmap=args.cmap,
        fullspace=args.fullspace,
        phase_signal=args.phase_signal
    )