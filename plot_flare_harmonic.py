#!/home/jfernandes/.venv/bin/python
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_flare_harmonic(data_vacuum=None, data_single_fluid=None, data_two_fluid=None, figsize=(7,5), dpi=100, label_vac='Vacuum', label_single_fluid='Single Fluid', label_two_fluid='Two Fluid'):
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
    label_vac : str
        Label for the vacuum data in the legend. Default is 'Vacuum'.
    label_single_fluid : str
        Label for the single fluid data in the legend. Default is 'Single Fluid'.   
    label_two_fluid : str
        Label for the two fluid data in the legend. Default is 'Two Fluid'.

    Returns
    -------
    None
        Displays the plot of the harmonic amplitudes.
    """

    # Check that at least one data file is provided
    if all(d is None for d in [data_vacuum, data_single_fluid, data_two_fluid]):
        raise ValueError("At least one data file must be provided.")

    # Load datasets
    psi_vac, db_vac = _load_flare_data(data_vacuum, label_vac)
    psi_sf, db_sf = _load_flare_data(data_single_fluid, label_single_fluid)
    psi_tf, db_tf = _load_flare_data(data_two_fluid, label_two_fluid)

    # Reference psiN_res for consistency check
    psi_ref = next((psi for psi in [psi_vac, psi_sf, psi_tf] if psi is not None), None)
    if psi_ref is None:
        raise ValueError("No valid psiN_res found in any file.")

    # Consistency check
    for name, psi in [(label_vac, psi_vac), (label_single_fluid, psi_sf), (label_two_fluid, psi_tf)]:
        if psi is not None and not np.allclose(psi, psi_ref, atol=1e-6):
            print(f"Warning: psiN_res grid mismatch detected in {name} data.")

    # Plotting palette
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
        plt.plot(psi_vac, np.abs(db_vac), 'o-', label=label_vac, color=colors[5])
    if data_single_fluid is not None:
        plt.plot(psi_sf, np.abs(db_sf), 'o-', label=label_single_fluid, color=colors[3])
    if data_two_fluid is not None:
        plt.plot(psi_tf, np.abs(db_tf), 'o-', label=label_two_fluid, color=colors[1])
    plt.xlabel('Normalized Poloidal Flux')
    plt.ylabel('$|\delta B_{m/n}|$ ( G / kA )')
    # plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()



def _load_flare_data(path, label):
    """
    Helper: load a FLARE surfmn .npz file and return (psiN_res, db_res).
    
    """
    if path is None:
        return None, None

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
    parser.add_argument("--data_vacuum", type=str, default=None, help="Path to the vacuum data file")
    parser.add_argument("--data_single_fluid", type=str, default=None, help="Path to the single fluid data file")
    parser.add_argument("--data_two_fluid", type=str, default=None, help="Path to the two fluid data file")
    parser.add_argument("--figsize", type=float, nargs=2, default=(7, 5), help="Figure size (width, height)")
    parser.add_argument("--dpi", type=int, default=100, help="Dots per inch for the figure")
    parser.add_argument("--label_vac", type=str, default="Vacuum", help="Label for vacuum data in legend")
    parser.add_argument("--label_single_fluid", type=str, default="Single Fluid", help="Label for single fluid data in legend")
    parser.add_argument("--label_two_fluid", type=str, default="Two Fluid", help="Label for two fluid data in legend")
    args = parser.parse_args()
    
    plot_flare_harmonic(
        data_vacuum=args.data_vacuum,
        data_single_fluid=args.data_single_fluid,
        data_two_fluid=args.data_two_fluid,
        figsize=args.figsize,
        dpi=args.dpi,
        label_vac=args.label_vac,
        label_single_fluid=args.label_single_fluid,
        label_two_fluid=args.label_two_fluid
    )