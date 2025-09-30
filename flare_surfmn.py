#!/home/jfernandes/.venv/bin/python
import argparse
import numpy as np
from flare import model
from flare.analysis import equi2d_rzarray, fluxsurf2d_parameters
from flare.analysis import fourier_transform
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata

def fluxsurf_params(n_max, m_max):
    """
    Build the psiN (normalized poloidal flux), q (safety factor) and area (flux surface area) arrays including the resonant values.

    Parameters
    ----------
    n_max : int
        Maximum toroidal mode number.
    m_max : int
        Maximum poloidal mode number.   

    Returns
    -------
    psiN : np.ndarray
        Array of sorted psiN (normalized poloidal flux) values including the resonant values.
    q : np.ndarray
        Array of sorted q (safety factor) values including the resonant values.
    area : np.ndarray
        Array of sorted area (flux surface area) values including the resonant values.
    psiN_res : np.ndarray
        Array of resonant psiN values.
    q_res : np.ndarray
        Array of resonant q values.
    
    """


    psiN_val = np.linspace(0.1, 0.9, 81)
    psiN_ped = np.linspace(0.901, 1.0, 100) 
    psiN_values = np.concatenate((psiN_val, psiN_ped))

    theta = 0
    rz_array = equi2d_rzarray(psiN_values, theta) # map  psiN to R,Z
    R_vals, Z_vals = rz_array[0], rz_array[1]
    flux_params = [fluxsurf2d_parameters((R, Z)) for R, Z in zip(R_vals, Z_vals)]
    q_vals = np.array([params[0] for params in flux_params])  # extract q values
    area_vals = np.array([params[2] for params in flux_params])  # extract area values
    q_vals = q_vals[:-1] # remove last element
    psiN_values = psiN_values[:-1]
    area_vals = area_vals[:-1]

    # array of sorted unique rational q values
    q_res = np.array([])
    for n in range (1, n_max + 1):
        for m in range (-m_max * n, m_max * n + 1):
            if m != 0:
                q_res = np.append(q_res, m / n)

    q_res.sort()
    q_res = np.unique(q_res)

    q_res = q_res[(q_res >= np.min(q_vals)) & (q_res <= np.max(q_vals))]

    #if q_vals is negative, invert the order of q_vals, psiN_values and area_vals
    if q_vals[-1] < 0:
        q_vals = q_vals[::-1]
        psiN_values = psiN_values[::-1]
        area_vals = area_vals[::-1]

    # interpolate the psiN and area values at the rational q values
    cs_psi = CubicSpline(q_vals, psiN_values)
    psiN_res = cs_psi(q_res)
    cs_a = CubicSpline(q_vals, area_vals)
    area_res = cs_a(q_res)

    # combine and sort the original and resonant values
    psiN = np.concatenate((psiN_values, psiN_res))
    q = np.concatenate((q_vals, q_res))
    area = np.concatenate((area_vals, area_res))

    idx_psiN = np.argsort(psiN)
    psiN = psiN[idx_psiN]
    q = q[idx_psiN]
    area = area[idx_psiN]

    return psiN, q, area, psiN_res, q_res



def flare_surfmn(flare_model, n_tor, m_max, filename):
    """
    Generate and save the surfmn data for a given FLARE model.

    Parameters
    ----------
    flare_model : str
        Path to the FLARE model file.
    n_tor : int
        Toroidal mode number.
    m_max : int
        Maximum poloidal mode number.
    filename : str
        Output filename for the data.
    n_psi : int
        Number of psi points. Default is 100.
    psiN_min : float
        Minimum psiN value. Default is 0.1.
    psiN_max : float
        Maximum psiN value. Default is 1.0.

    Returns
    -------
    None
        Saves the surfmn data to a .npz file.

    """


    # Load the model
    print("Loading flare model...")
    try:
        model.load(flare_model)
    except Exception as e:
        raise ValueError(f"Failed to load flare model: {e}")

    # Build arrays
    psiN_values, q_values, area_values, psiN_res, q_res_values = fluxsurf_params(n_tor, m_max)
    m_values= np.arange(-(m_max + 1) + 1, (m_max + 1) + 1)
    db_matrix = np.zeros((len(psiN_values), len(m_values)))

    # Compute the Fourier coefficients
    print("Computing Fourier coefficients...")
    try:
        for i, psiN in enumerate(psiN_values):
            db_spectrum = fourier_transform(psiN, n_tor, 2 * (m_max + 1))
            db_matrix[i, :] = np.abs(db_spectrum) * area_values[i] * 1e4 # convert to G / kA
        m_mesh, psiN_mesh = np.meshgrid(m_values, psiN_values)
        m_mesh = m_mesh[:, :-1]
        psiN_mesh = psiN_mesh[:, :-1]
        db_matrix = db_matrix[:, :-1]
    except Exception as e:
        raise ValueError(f"Failed to compute Fourier coefficients: {e}")
    

    # Compute harmonic amplitude on resonant surfaces
    print("Computing harmonic amplitudes on resonant surfaces...")
    try:
        m_res = q_res_values * n_tor
        points = np.column_stack((m_mesh.flatten(), psiN_mesh.flatten()))
        values = db_matrix.flatten()
        interp_points = np.column_stack((m_res, psiN_res))
        db_res = griddata(points, values, interp_points, method='cubic', fill_value=0)
    except Exception as e:
        raise ValueError(f"Failed to compute harmonic amplitudes on resonant surfaces: {e}")


    # Save data
    print("Saving data...")
    try:
        np.savez(filename, n_tor=n_tor, psiN_values=psiN_values, m_values=m_values, m_mesh=m_mesh, psiN_mesh=psiN_mesh, db_matrix=db_matrix, q_vals=q_values, db_res=db_res, psiN_res=psiN_res, q_res=q_res_values)
        print(f"Data saved to {filename}.npz")
    except Exception as e:
            raise ValueError(f"Failed to save data: {e}")
    
    # Free memory
    print("Freeing memory...")
    model.free()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FLARE surfmn spectra.")
    parser.add_argument("flare_model", type=str, help="Path to the flare model file")
    parser.add_argument("n_tor", type=int, help="Toroidal mode number")
    parser.add_argument("m_max", type=int, help="Maximum poloidal mode number")
    parser.add_argument("filename", type=str, help="Output filename for the data")
    args = parser.parse_args()

    flare_surfmn(
        flare_model=args.flare_model,
        n_tor=args.n_tor,
        m_max=args.m_max,
        filename=args.filename
    )
