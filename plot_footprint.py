#!/home/jfernandes/.venv/bin/python
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter


def plot_footprint(filename=None, plate=None, which_plot="all", xaxis="rad", cmap="jet", cmap_key=10, figsize=(10, 5), sizef=1, dpi=80, norm_f=6, v_min=None, v_max=None, turn_cap=None , savefig=None):
    """
    Function to plot footprints evaluated by the maglib code fpgen.
    
    Parameters
    ----------
    filename : str
        Path to the file containing the data. Default is None.
    plate : str
        Divertor plate: "h" for horizontal (floor) or "v" for vertical (wall). Default is None.
    which_plot : str
        Type of plot to generate: "cl" (connection length), "psi" (psi min), "turns" (number of toroidalturns) or "au" (arbitrary units), "all". Default is "all".
    xaxis : str
        Type of x-axis: "rad" for radians or "deg" for degrees. Default is "rad".
    cmap : str
        Colormap to use for the plots. Default is "jet".
    cmap_key : int
        Number of colors in the discretized version of the colormap. Default is 10.
    figsize : tuple
        Size of the figure in inches. Default is (10, 5).
    sizef : float
        Size factor for the figure (increase or decrease figure size without changig aspect ratio). Default is 1.
    dpi : int
        Dots per inch for the figure. Default is 80 (default is low for better performance).
    norm_f : float
        Normalization factor for the arbitrary units plot. Default is 6.
    v_min : float
        Minimum value for the normalized colormap in arbitrary units. Default is None.
    v_max : float
        Maximum value for the normalized colormap in arbitrary units. Default is None.
    turn_cap : tuple
        Interval to cap the color scale for the number of toroidal turns plot. Default is None.
    savefig : str
        Path to save the figure(s). If None, figure(s) will not be saved. Default is None.

    Returns
    -------
    None
        Displays the plots.

    """


    # Load data
    if filename is None:
        raise ValueError("Filename cannot be None. Please provide a valid file path.")
    data = np.loadtxt(filename)

    # Extract relevant columns
    x = data[:, 2] #phi
    if plate == "h":
        y = data[:, 0] #r
    elif plate == "v":
        y = data[:, 1] #z
    else:
        raise ValueError("Invalid plate type. Use 'h' for horizontal or 'v' for vertical.")

    # Sort data by phi then by y (R or Z)
    sort_idx = np.lexsort((y, x))
    data = data[sort_idx]

    # Re-extract after sorting
    x = data[:, 2]
    if plate == "h":
        y = data[:, 0]
    elif plate == "v":
        y = data[:, 1]

    z_cl = data[:, 3] #connection length
    z_psi = data[:, 4] #psin
    z_turns = data[:, 5] #number of turns

    # Determine the grid size
    num_rows = int(len(np.unique(x))) # Number of rows
    num_columns = int(len(data)/num_rows) # Number of columns
    z_cl = np.reshape(z_cl, (num_rows, num_columns)).transpose() # z column to matrix
    z_psi = np.reshape(z_psi, (num_rows, num_columns)).transpose() # z column to matrix
    z_turns = np.reshape(z_turns, (num_rows, num_columns)).transpose() # z column to matrix

    # Custom x-axis formatter: convert degrees to radians
    def degrees_to_radians(x, pos):
        radians = x * np.pi / 180  # Convert degrees to radians
        return f"{radians:.1f}" if radians > 0 else "0"

    # Custom colormap
    cmap_cap = mpl.cm.get_cmap(cmap).copy()
    cmap_f = mpl.cm.get_cmap(cmap,cmap_key).copy()

    # Modify colormap to set under values to white
    cmap_cap.set_under("white")
    cmap_f.set_under("white")

    if which_plot not in ["cl", "psi", "turns", "au", "all"]:
        raise ValueError("Invalid plot type. Use 'cl' for connection length, 'psi' for psi min, 'turns' for number of turns, 'au' for arbitrary units, or 'all'.")

    #### Connection Length plot ####
    if which_plot == "all" or which_plot == "cl":
        fsize=(figsize[0]*sizef, figsize[1]*sizef)
        plt.figure(figsize=fsize, dpi=dpi)
        plt.imshow(z_cl, cmap=cmap, extent=[0, 360, np.min(y), np.max(y)], origin='lower', aspect='auto', norm=LogNorm())
        plt.colorbar().set_label("connection length ( m )")
        if xaxis == "rad":
            plt.gca().xaxis.set_major_formatter(FuncFormatter(degrees_to_radians))
            plt.xlabel("$\phi$ ( rad )")
        elif xaxis == "deg":
            plt.xlabel("$\phi$ ( deg )")
            pass
        else:
            raise ValueError("Invalid x-axis type. Use 'rad' for radians or 'deg' for degrees.")
        plt.ylabel("$R$ ( m )")

        # Save figure if savefig is provided
        if savefig is not None:
            plt.savefig(f'{savefig}_cl.png', dpi=dpi, bbox_inches='tight')
            print(f"Figure saved as {savefig}_cl.png")

        # Show plot
        plt.show(block=False)


    #### Psi min plot ####
    if which_plot == "all" or which_plot == "psi":
        fsize=(figsize[0]*sizef, figsize[1]*sizef)
        plt.figure(figsize=fsize, dpi=dpi)
        plt.imshow(z_psi, cmap=cmap, extent=[0, 360, np.min(y), np.max(y)], origin='lower', aspect='auto')
        plt.colorbar().set_label(r'$\psi_{N\,\,\mathrm{min}}$')
        if xaxis == "rad":
            plt.gca().xaxis.set_major_formatter(FuncFormatter(degrees_to_radians))
            plt.xlabel("$\phi$ ( rad )")
        elif xaxis == "deg":
            plt.xlabel("$\phi$ ( deg )")
            pass
        else:
            raise ValueError("Invalid x-axis type. Use 'rad' for radians or 'deg' for degrees.")
        plt.ylabel("$R$ ( m )")

        # Save figure if savefig is provided
        if savefig is not None:
            plt.savefig(f'{savefig}_psi.png', dpi=dpi, bbox_inches='tight')
            print(f"Figure saved as {savefig}_psi.png")

        # Show plot
        plt.show(block=False)


    #### number of turns plot ####
    if which_plot == "all" or which_plot == "turns":
        fsize=(figsize[0]*sizef, figsize[1]*sizef)
        plt.figure(figsize=fsize, dpi=dpi)
        plt.imshow(z_turns, cmap=cmap_f, extent=[0, 360, np.min(y), np.max(y)], origin='lower', aspect='auto', vmin=turn_cap[0], vmax=turn_cap[1])
        plt.colorbar().set_label("toroidal turns")
        if xaxis == "rad":
            plt.gca().xaxis.set_major_formatter(FuncFormatter(degrees_to_radians))
            plt.xlabel("$\phi$ ( rad )")
        elif xaxis == "deg":
            plt.xlabel("$\phi$ ( deg )")
            pass
        else:
            raise ValueError("Invalid x-axis type. Use 'rad' for radians or 'deg' for degrees.")
        plt.ylabel("$R$ ( m )")

        # Save figure if savefig is provided
        if savefig is not None:
            plt.savefig(f'{savefig}_turns.png', dpi=dpi, bbox_inches='tight')
            print(f"Figure saved as {savefig}_turns.png")

        # Show plot
        plt.show(block=False)


    #### arbitrary units plot ####
    if which_plot == "all" or which_plot == "au":

        # Check if v_min and v_max are provided
        if v_min is not None and v_max is not None:
            if not isinstance(v_min, (int, float)) or not isinstance(v_max, (int, float)):
                raise ValueError("v_min and v_max must be numbers.")
            if v_min >= v_max:
                raise ValueError("v_min must be less than v_max.")
            # Normalize z_norm to the range [v_min, v_max]
            z_norm = ((1 - (1/(z_psi*z_cl))) - v_min) / (v_max - v_min)
        else:
            # default normalization
            z_norm = (1-(1/(z_psi*z_cl)-np.min(1/(z_psi*z_cl)))*norm_f/np.max(1/(z_psi*z_cl)))
        
        fsize=(figsize[0]*sizef, figsize[1]*sizef)
        plt.figure(figsize=fsize, dpi=dpi)
        plt.imshow(z_norm, cmap=cmap_f, extent=[0, 360, np.min(y), np.max(y)], origin='lower', aspect='auto', vmin=0, vmax=1)
        plt.colorbar().set_label('$( \psi_{N\,\,\mathrm{min}}$ . connection length ) $^{-1}_\mathrm{norm}$') 
        if xaxis == "rad":
            plt.gca().xaxis.set_major_formatter(FuncFormatter(degrees_to_radians))
            plt.xlabel("$\phi$ ( rad )")
        elif xaxis == "deg":
            plt.xlabel("$\phi$ ( deg )")
            pass
        else:
            raise ValueError("Invalid x-axis type. Use 'rad' for radians or 'deg' for degrees.")
        plt.ylabel("$R$ ( m )")

        # Save figure if savefig is provided
        if savefig is not None:
            plt.savefig(f'{savefig}_au.png', dpi=dpi, bbox_inches='tight')
            print(f"Figure saved as {savefig}_au.png")

        # Show plot
        plt.show(block=False)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot footprints from fpgen output file.")
    parser.add_argument("filename", type=str, help="Path to the fpgen output file.")
    parser.add_argument("plate", type=str, choices=["h", "v"], help="Divertor plate: 'h' for horizontal (floor) or 'v' for vertical (wall).")
    parser.add_argument("--which_plot", type=str, default="all", choices=["cl", "psi", "turns", "au", "all"], help="Type of plot to generate: 'cl', 'psi', 'turns', 'au', or 'all'. Default is 'all'.")
    parser.add_argument("--xaxis", type=str, default="rad", choices=["rad", "deg"], help="Type of x-axis: 'rad' for radians or 'deg' for degrees. Default is 'rad'.")
    parser.add_argument("--cmap", type=str, default="jet", help="Colormap to use for the plots. Default is 'jet'.")
    parser.add_argument("--cmap_key", type=int, default=10, help="Number of colors in the discretized version of the colormap. Default is 10.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(10, 5), help="Size of the figure in inches. Default is (10, 5).")
    parser.add_argument("--sizef", type=float, default=1, help="Size factor for the figure. Default is 1.")
    parser.add_argument("--dpi", type=int, default=80, help="Dots per inch for the figure. Default is 80.")
    parser.add_argument("--norm_f", type=float, default=6, help="Normalization factor for the arbitrary units plot. Default is 6.")
    parser.add_argument("--v_min", type=float, default=None, help="Minimum value for the normalized colormap in arbitrary units. Default is None.")
    parser.add_argument("--v_max", type=float, default=None, help="Maximum value for the normalized colormap in arbitrary units. Default is None.")
    parser.add_argument("--turn_cap", type=int, nargs=2, default=None, help="Interval to cap the color scale for the number of toroidal turns plot. Default is None.")
    parser.add_argument("--savefig", type=str, default=None, help="Path to save the figure(s). If None, figure(s) will not be saved. Default is None.")

    args = parser.parse_args()

    plot_footprint(
        filename=args.filename,
        plate=args.plate,
        which_plot=args.which_plot,
        xaxis=args.xaxis,
        cmap=args.cmap,
        cmap_key=args.cmap_key,
        figsize=args.figsize,
        sizef=args.sizef,
        dpi=args.dpi,
        norm_f=args.norm_f,
        v_min=args.v_min,
        v_max=args.v_max,
        turn_cap=args.turn_cap,
        savefig=args.savefig
    )