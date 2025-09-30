#!/home/jfernandes/.venv/bin/python
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_flare_harmonic(data_vacuum=None, data_single_fluid=None, data_two_fluid=None, figsize=(7,5), dpi=100):
    """
    Plot the amplitude of the poloidal Fourier harmonics along the resonant condition from the given .npz data file(s).
    Up to three datasets can be plotted for comparison: vacuum, single fluid, and two fluid.

    Parameters
    ----------
    data_vacuum : str or None
        Path to the .npz file containing vacuum data. Default is None.
    data_single_fluid : str or None
        Path to the .npz file containing single fluid data. Default is None.
    data_two_fluid : str or None
        Path to the .npz file containing two fluid data. Default is None.
    figsize : tuple
        Size of the figure (width, height). Default is (7, 5).
    dpi : int
        Dots per inch for the figure. Default is 100.

    Returns
    -------
    None
        Displays the plot of the harmonic amplitudes.
    """
    if data_vacuum is None and data_single_fluid is None and data_two_fluid is None:
        raise ValueError("At least one data file must be provided.")

    # Load data
    psiN_res = None
    if data_vacuum is not None:
        print("Loading vacuum data from {}...".format(data_vacuum))
        try:
            with np.load(data_vacuum) as f:
                if psiN_res is None:
                    psiN_res = f['psiN_res']
                db_res_vacuum = f['db_res']
        except Exception as e:
            raise ValueError(f"Failed to load vacuum data: {e}")
        print("Vacuum data loaded successfully.")

    if data_single_fluid is not None:
        print("Loading single fluid data from {}...".format(data_single_fluid))
        try:
            with np.load(data_single_fluid) as f:
                if psiN_res is None:
                    psiN_res = f['psiN_res']
                db_res_single_fluid = f['db_res']
        except Exception as e:
            raise ValueError(f"Failed to load single fluid data: {e}")
        print("Single fluid data loaded successfully.")

    if data_two_fluid is not None: 
        print("Loading two fluid data from {}...".format(data_two_fluid))
        try:
            with np.load(data_two_fluid) as f:
                if psiN_res is None:
                    psiN_res = f['psiN_res']
                db_res_two_fluid = f['db_res']
        except Exception as e:
            raise ValueError(f"Failed to load two fluid data: {e}")
        print("Two fluid data loaded successfully.")

    # Plotting
    colors = {
    1: (237/255, 32/255, 36/255),      # Red
    2: (164/255, 194/255, 219/255),    # Light blue
    3: (57/255, 83/255, 164/255),      # Dark Blue
    4: (50/255, 180/255, 80/255),      # Light Green
    5: (0, 0, 0),                      # Black
    6: (0, 110/255, 0),                # Green
    7: (1.0, 0.0, 1.0),                # Magenta
    8: (0.0, 1.0, 1.0),                # Cyan
    9: (0.0, 0.0, 1.0),                # Normal blue
    10: (1.0, 0.0, 0.0),               # Normal red
    12: (255/255, 102/255, 154/255)    # Lilac
    }
    
    print("Plotting...")
    plt.figure(figsize=figsize, dpi=dpi)
    if data_vacuum is not None:
        plt.plot(psiN_res, np.abs(db_res_vacuum), 'o-', label='Vacuum', color=colors[1])
    if data_single_fluid is not None:
        plt.plot(psiN_res, np.abs(db_res_single_fluid), 'o-', label='Single Fluid', color=colors[3])
    if data_two_fluid is not None:
        plt.plot(psiN_res, np.abs(db_res_two_fluid), 'o-', label='Two Fluid', color=colors[4])
    plt.xlabel('Normalized Poloidal Flux')
    plt.ylabel('$|\delta B_{m/n}|$ ( G / kA )')
    # plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FLARE harmonic amplitudes on resonant surfaces.")
    parser.add_argument("--data_vacuum", type=str, default=None, help="Path to the vacuum data file")
    parser.add_argument("--data_single_fluid", type=str, default=None, help="Path to the single fluid data file")
    parser.add_argument("--data_two_fluid", type=str, default=None, help="Path to the two fluid data file")
    parser.add_argument("--figsize", type=float, nargs=2, default=(7, 5), help="Figure size (width, height)")
    parser.add_argument("--dpi", type=int, default=100, help="Dots per inch for the figure")
    args = parser.parse_args()
    
    plot_flare_harmonic(data_vacuum=args.data_vacuum, data_single_fluid=args.data_single_fluid, data_two_fluid=args.data_two_fluid, figsize=args.figsize, dpi=args.dpi)