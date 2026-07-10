#!/home/jfernandes/.venv/bin/python
import argparse
import numpy as np
from flare import model
from scipy.interpolate import CubicSpline, griddata
from flare.analysis import equi2d_rzarray, fluxsurf2d_parameters, fourier_transform


def flare_surfmn(flare_model: str, n_tor: int, m_max: int, filename: str, n_pol: int = 400) -> None:
    """
    Generate and save the surfmn data for a given FLARE model.

    Parameters
    ----------
    flare_model : str
        Path to the FLARE model file.
    n_tor : int
        Toroidal mode number.
    m_max : int
        Maximum poloidal mode number to keep in the output.
    filename : str
        Output filename for the data.
    n_pol : int
        Number of poloidal points used for the Fourier transform, decoupled
        from m_max (analogous to IDL schaffer_plot's `bins` keyword, which
        always computes at a fixed high resolution and lets the m range
        shown be a separate, downstream choice). Must resolve at least
        m_max (n_pol >= 2*(m_max+1)); auto-increased with a warning if not.
        Default of 400 matches the resolution used to validate this tool
        against IDL's plot_br.pro/schaffer_plot.pro reference output.

    Returns
    -------
    None
        Saves the surfmn data to a .npz file.

    """

    min_n_pol = 2 * (m_max + 1)
    if n_pol < min_n_pol:
        print(f"Warning: n_pol={n_pol} cannot resolve m_max={m_max}; increasing n_pol to {min_n_pol}.")
        n_pol = min_n_pol

    
    # Load the model
    print("Loading flare model...")
    try:
        model.load(flare_model)
    except Exception as e:
        raise ValueError(f"Failed to load flare model: {e}")

    try:
        status = "building flux surface parameters"
        print("Building flux surface parameters...")
        psiN_values, q_values, area_values, psiN_res, q_res_values, m_res_values = fluxsurf_params(n_tor, m_max)

        # Full ascending mode array returned by fourier_transform at this
        # resolution, and the mask selecting the requested +/-m_max range
        # out of it (matching IDL: FFT resolution and displayed m range are
        # independent; only this crop depends on m_max).
        m_full = np.arange(-(n_pol // 2) + 1, n_pol // 2 + 1)
        keep = np.abs(m_full) <= m_max
        m_values = m_full[keep]
        db_matrix = np.zeros((len(psiN_values), len(m_values)))

        # Compute the Fourier coefficients
        status = "computing fourier coefficients"
        print("Computing Fourier coefficients...")
        for i, psiN in enumerate(psiN_values):
            db_spectrum = fourier_transform(psiN, n_tor, n_pol)
            db_full = 2 * (2 * np.pi) ** 2 * np.abs(db_spectrum) / area_values[i] * 1e4  # G
            db_matrix[i, :] = db_full[keep]

        m_mesh, psiN_mesh = np.meshgrid(m_values, psiN_values)


        # Compute harmonic amplitude on resonant surfaces
        status = "computing harmonic amplitudes on resonant surfaces"
        print("Computing harmonic amplitudes on resonant surfaces...")
        points = np.column_stack((m_mesh.flatten(), psiN_mesh.flatten()))
        values = db_matrix.flatten()
        interp_points = np.column_stack((m_res_values, psiN_res))
        db_res = griddata(points, values, interp_points, method='cubic', fill_value=0)

        # Save data
        status = "saving data"
        print("Saving data...")
        if not filename.endswith(".npz"):
                filename += ".npz"
        np.savez(
            filename,
            n_tor=n_tor,
            psiN_values=psiN_values,
            m_values=m_values,
            m_mesh=m_mesh,
            psiN_mesh=psiN_mesh,
            db_matrix=db_matrix,
            q_vals=q_values,
            db_res=db_res,
            psiN_res=psiN_res,
            q_res=q_res_values,
            m_res=m_res_values)
        
        print(f"Data saved to {filename}")
    
    except Exception as e:
        raise ValueError(f"Failed on {status}: {e}")
    
    finally:
        print("Freeing FLARE model from memory...")
        model.free()
    
    
    print("Done.")

    

def fluxsurf_params(n_tor: int, m_max: int):
    """
    Build the psiN (normalized poloidal flux), q (safety factor) and area (flux surface area) arrays including the resonant values.

    Parameters
    ----------
    n_tor : int
        Toroidal mode number.
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
        Array of resonant q values (= m_res / n_tor).
    m_res : np.ndarray
        Array of resonant poloidal mode numbers (integers, |m| <= m_max).

    """


    psiN_val = np.linspace(0.1, 0.9, 81)
    psiN_ped = np.linspace(0.901, 0.9999, 100) 
    psiN_values = np.concatenate((psiN_val, psiN_ped))

    theta = 0
    rz_array = equi2d_rzarray(psiN_values, theta) # map  psiN to R,Z
    R_vals, Z_vals = rz_array[0], rz_array[1]
    flux_params = [fluxsurf2d_parameters((R, Z)) for R, Z in zip(R_vals, Z_vals)]
    q_vals = np.array([params[0] for params in flux_params])  # extract q values
    area_vals = np.array([params[2] for params in flux_params])  # extract area values
    
    # remove last element
    q_vals = q_vals[:-1] 
    psiN_values = psiN_values[:-1]
    area_vals = area_vals[:-1]

    # Resonant surfaces for this field: q = m/n_tor for integer m in
    # [-m_max, m_max] (matching the displayed poloidal mode range), excluding
    # m=0. m_res is kept as its own integer array so downstream code never
    # has to recover it by dividing back out of q_res.
    m_res = np.arange(-m_max, m_max + 1)
    m_res = m_res[m_res != 0]
    q_res = m_res / n_tor

    mask = (q_res >= np.min(q_vals)) & (q_res <= np.max(q_vals))
    q_res = q_res[mask]
    m_res = m_res[mask]

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

    return psiN, q, area, psiN_res, q_res, m_res



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FLARE surfmn spectra.")
    parser.add_argument("flare_model", type=str, help="Path to the flare model file")
    parser.add_argument("n_tor", type=int, help="Toroidal mode number")
    parser.add_argument("m_max", type=int, help="Maximum poloidal mode number")
    parser.add_argument("filename", type=str, help="Output filename for the data")
    parser.add_argument("--n_pol", type=int, default=400,
                         help="Poloidal Fourier transform resolution, decoupled from m_max (default: 400)")
    args = parser.parse_args()

    flare_surfmn(
        flare_model=args.flare_model,
        n_tor=args.n_tor,
        m_max=args.m_max,
        filename=args.filename,
        n_pol=args.n_pol
    )
