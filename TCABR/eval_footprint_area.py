#!/home/jfernandes/.venv/bin/python
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def eval_footprint_area(filename: str=None, turn_cap: tuple=(5, 10), cmap_key: int=5, cmap: str="jet", figsize: tuple=(10, 5), sizef: float=1, dpi: int=80, xaxis: str="rad") -> None:
    """
    Evaluate footprint area from footprint data file
    
    Parameters
    ----------
    filename : str
        Path to the footprint data file.
    turn_cap : tuple of int
        Tuple specifying the turn cap limits (min_turns, max_turns). Default is (5, 10).
    cmap_key : int
        Colormap key for resampling. Default is 5.
    cmap : str
        Colormap name. Default is "jet".
    figsize : tuple of float
        Figure size as (width, height). Default is (10, 5).
    sizef : float
        Size factor for figure scaling preserving aspect ratio. Default is 1.
    dpi : int
        DPI for the figure. Default is 80.
    xaxis : str
        X-axis type: 'rad' for radians or 'deg' for degrees. Default is "rad".

    Returns
    -------
    None
        Displays plots of the footprint and area calculations.
        
    """
    # Load data
    print(f"\nLoading data from {filename}")
    if filename is None:
        raise ValueError("Filename cannot be None. Please provide a valid file path.")
    data = np.loadtxt(filename)

    # Determine divertor plate
    if np.unique(data[:, 0]).size == 1:
        plate = "v"  # vertical plate
        print("Vertical plate detected based on (R,Z) values.\n")
        ylabel = "$Z$ ( m )"
        radius = np.unique(data[:, 0])[0]
    else:
        plate = "h"  # horizontal plate
        print("Horizontal plate detected based on (R,Z) values.\n")
        ylabel = "$R$ ( m )"

    # Custom x-axis formatter: convert degrees to radians
    def degrees_to_radians(x, pos):
        radians = x * np.pi / 180  # Convert degrees to radians
        return f"{radians:.1f}" if radians > 0 else "0"
    
    if plate == "h":
        print("Processing data for horizontal plate...\n")
        # Extract relevant columns
        x = data[:, 2] # phi
        y = data[:, 0] # R
        z_turns = data[:, 5] # number of turns

        # Determine the grid size
        num_rows = int(len(np.unique(x)))
        num_columns = int(len(data)/num_rows)

        # Compute pixel area for each point
        unique_R = np.unique(y)
        unique_R_spacing = np.diff(unique_R)
        if unique_R_spacing.std() > 1e-5:
            print("Warning: Non-uniform spacing detected in R values.")
        else:
            unique_R_spacing = unique_R_spacing[0]
            print(f"R_max: {np.max(y):.5f}; R_min: {np.min(y):.5f}; nR: {len(unique_R)}; R spacing: {unique_R_spacing:.5f}")

        unique_phi = np.unique(x)
        unique_phi_spacing = np.diff(unique_phi)
        if unique_phi_spacing.std() > 1e-5:
            print("Warning: Non-uniform spacing detected in phi values.")
        else:
            unique_phi_spacing = unique_phi_spacing[0]
            print(f"Phi_max: {np.max(x):.5f}; Phi_min: {np.min(x):.5f}; nPhi: {len(unique_phi)}; Phi spacing: {unique_phi_spacing:.5f}\n")

        pixel_area = 1/2 * (unique_R_spacing * unique_phi_spacing) * (2 * y + unique_R_spacing)

        # Count how many points are outside the turn cap limits
        if turn_cap is not None:
            below_cap = np.sum(z_turns <= turn_cap[0])
            above_cap = np.sum(z_turns >= turn_cap[1])
            print(f"Total number of points: {len(data)}")
            print(f"Number of points below or equal to {turn_cap[0]} turns: {below_cap}")
            print(f"Number of points above or equal to {turn_cap[1]} turns: {above_cap}\n")
            print(f"Total number of points in the footprint (turns >= {turn_cap[0]}): {len(data) - below_cap}\n")

            # Compute area composed of valid points
            valid_points = np.sum(z_turns >= turn_cap[0])
            total_area = np.sum(pixel_area)
            valid_area = np.sum(pixel_area[z_turns >= turn_cap[0]])
            print(f"Total area: {total_area:.5f} m²")
            print(f"Valid area (turns >= {turn_cap[0]}): {valid_area:.5f} m²\n")

        # Reshape z data into matrices
        z_turns = np.reshape(z_turns, (num_rows, num_columns)).transpose()

        # Custom colormap
        cmap_f = mpl.colormaps.get_cmap(cmap).resampled(cmap_key)
        cmap_f.set_under("white")

        #### Number of turns plot ####
        fsize=(figsize[0]*sizef, figsize[1]*sizef)
        plt.figure(figsize=fsize, dpi=dpi)
        plt.imshow(z_turns, cmap=cmap_f, extent=[0, 360, np.min(y), np.max(y)], origin='lower', aspect='auto', vmin=turn_cap[0] if turn_cap is not None else None, vmax=turn_cap[1] if turn_cap is not None else None)
        plt.colorbar().set_label("toroidal turns")
        if xaxis == "rad":
            plt.gca().xaxis.set_major_formatter(FuncFormatter(degrees_to_radians))
            plt.xlabel("$\phi$ ( rad )")
        elif xaxis == "deg":
            plt.xlabel("$\phi$ ( deg )")
            pass
        else:
            raise ValueError("Invalid x-axis type. Use 'rad' for radians or 'deg' for degrees.")
        plt.ylabel(ylabel)

        # Show plot
        plt.show(block=False)

    if plate == "v":
        print("Processing data for vertical plate...\n")
        # Extract relevant columns
        x = data[:, 2] # phi
        y = data[:, 1] # Z
        z_turns = data[:, 5] # number of turns

        # Determine the grid size
        num_rows = int(len(np.unique(x)))
        num_columns = int(len(data)/num_rows)

        # Count how many points are outside the turn cap limits
        if turn_cap is not None:
            below_cap = np.sum(z_turns <= turn_cap[0])
            above_cap = np.sum(z_turns >= turn_cap[1])
            print(f"Total number of points: {len(data)}")
            print(f"Number of points below or equal to {turn_cap[0]} turns: {below_cap}")
            print(f"Number of points above or equal to {turn_cap[1]} turns: {above_cap}\n")
            print(f"Total number of points in the footprint (turns >= {turn_cap[0]}): {len(data) - below_cap}\n")

            # Compute area composed of valid points
            valid_points = np.sum(z_turns >= turn_cap[0])
            total_area = (np.max(y) - np.min(y)) * (2 * np.pi * radius)
            valid_area = (valid_points / len(data)) * total_area
            print(f"Total area: {total_area:.5f} m²")
            print(f"Valid area (turns >= {turn_cap[0]}): {valid_area:.5f} m²\n")

        # Reshape z data into matrices
        z_turns = np.reshape(z_turns, (num_rows, num_columns)).transpose()

        # Custom colormap
        cmap_f = mpl.colormaps.get_cmap(cmap).resampled(cmap_key)
        cmap_f.set_under("white")

        #### Number of turns plot ####
        fsize=(figsize[0]*sizef, figsize[1]*sizef)
        plt.figure(figsize=fsize, dpi=dpi)
        plt.imshow(z_turns, cmap=cmap_f, extent=[0, 360, np.min(y), np.max(y)], origin='lower', aspect='auto', vmin=turn_cap[0] if turn_cap is not None else None, vmax=turn_cap[1] if turn_cap is not None else None)
        plt.colorbar().set_label("toroidal turns")
        if xaxis == "rad":
            plt.gca().xaxis.set_major_formatter(FuncFormatter(degrees_to_radians))
            plt.xlabel("$\phi$ ( rad )")
        elif xaxis == "deg":
            plt.xlabel("$\phi$ ( deg )")
            pass
        else:
            raise ValueError("Invalid x-axis type. Use 'rad' for radians or 'deg' for degrees.")
        plt.ylabel(ylabel)

        # Show plot
        plt.show(block=False)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate footprint area from data file.")
    parser.add_argument("filename", type=str, help="Path to the footprint data file.")
    parser.add_argument("--turn_cap", type=int, nargs=2, default=(5,10), help="Turn cap limits as two integers.")
    parser.add_argument("--cmap_key", type=int, default=5, help="Colormap key for resampling.")
    parser.add_argument("--cmap", type=str, default="jet", help="Colormap name.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(10,5), help="Figure size as width and height.")
    parser.add_argument("--sizef", type=float, default=1, help="Size factor for figure scaling preserving aspect ratio.")
    parser.add_argument("--dpi", type=int, default=80, help="DPI for the figure.")
    parser.add_argument("--xaxis", type=str, choices=["rad", "deg"], default="rad", help="X-axis type: 'rad' for radians or 'deg' for degrees.")

    args = parser.parse_args()
    eval_footprint_area(
        filename=args.filename,
        turn_cap=args.turn_cap,
        cmap_key=args.cmap_key,
        cmap=args.cmap,
        figsize=args.figsize,
        sizef=args.sizef,
        dpi=args.dpi,
        xaxis=args.xaxis
    )