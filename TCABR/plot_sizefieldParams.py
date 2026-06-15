#!/home/jfernandes/.venv/bin/python
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_sizefieldParams(filename: str | None = None,
                        a1   = 1.0,
                        a2   = 2.6,
                        a3   = 2.05,   
                        a4p  = 0.035, 
                        a4v  = 0.05,   
                        a5p  = 0.035,  
                        a5v  = 0.05,   
                        a6   = 0.007,  
                        a7   = 0.007,  
                        lc1  = 100.,
                        lc2  = 100.,
                        Wc   = 0.07,
                        psic = 0.16) -> None:
        # a2 Exponent, internal (psi < a1)
        # a3 Exponent, external (psi > a1)
        # a4p Normal amplitude, internal
        # a4v Normal amplitude, external
        # a5p Tangential amplitude, internal
        # a5v Tangential amplitude, external
        # a6 Tangential concavity
        # a7 # Normal concavity
    """
    Plot the normal and tangential mesh size profiles vs normalized
    poloidal flux defined by the M3D-C1 sizefieldParam model.

    h(psi) = 1 / ( 1/(a_t*(1 - exp(-|psi/a1 - 1|^a_e)) + a_c)
                    + (1/lc) * 1/(1 + ((psi - psic)/Wc)^2) )

    where a_t, a_c, a_e are the tangential/normal and concavity
    amplitudes/exponents (different for the inner psi<a1 and outer
    psi>a1 branches), lc is the far-field coarsening length scale,
    and the last term smoothly coarsens the mesh around psi=psic
    over a width Wc.

    Parameters
    ----------
    filename : str or None
        Path to a sizefieldParam file containing 13 whitespace/newline
        separated floats, in the order:
        a1, a2, a3, a4p, a4v, a5p, a5v, a6, a7, lc1, lc2, Wc, psic
        If None, default parameters are used.

    Returns
    -------
    None
        Displays the mesh size plot.
    """

    # Reading from sizefieldParam file
    if filename is not None:
        path = Path(filename)
        try:
            A = np.fromstring(path.read_text(), sep=' ')[:13]
            (a1, a2, a3, a4p, a4v, a5p, a5v, a6, a7, lc1, lc2, Wc, psic) = A
        except (OSError, ValueError):
            print(f'Warning: no valid file found at {filename}')

    # Normal length
    psip = np.linspace(0, a1, 101)
    psiv = np.linspace(a1, 3, 101)

    h1p = 1. / (1. / (a4p * (1 - np.exp(-np.abs(psip/a1 - 1)**a2)) + a7)
                + 1/lc1 * (1. / (1 + ((psip - psic)/Wc)**2)))
    h1v = 1. / (1. / (a4v * (1 - np.exp(-np.abs(psiv/a1 - 1)**a3)) + a7)
                + 1/lc1 * (1. / (1 + ((psiv - psic)/Wc)**2)))

    # Tangential length
    h2p = 1. / (1. / (a5p * (1 - np.exp(-np.abs(psip/a1 - 1)**a2)) + a6)
                + 1/lc2 * (1. / (1 + ((psip - psic)/Wc)**2)))
    h2v = 1. / (1. / (a5v * (1 - np.exp(-np.abs(psiv/a1 - 1)**a3)) + a6)
                + 1/lc2 * (1. / (1 + ((psiv - psic)/Wc)**2)))

    print(f'Coefficients: {a1}  {a2}  {a3}  {a4p}  {a4v}  {a5p}  {a5v}  '
          f'{a6}  {a7}  {lc1}  {lc2}  {Wc}  {psic}')

    # Plotting
    fig, ax = plt.subplots(num=26)
    ax.plot(psip, h1p*1e2, 'r', label='Normal')
    ax.plot(psip, h2p*1e2, 'm', label='Tangential')
    ax.plot(psiv, h1v*1e2, 'r')
    ax.plot(psiv, h2v*1e2, 'm')
    ax.set_xlabel(r'$\Psi_N$')
    ax.set_ylabel('Mesh Size (cm)')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 6)
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot M3D-C1 sizefieldParam mesh size profiles.')
    parser.add_argument('--filename', type=str, default=None, help='Path to a sizefieldParam file.')
    args = parser.parse_args()

    plot_sizefieldParams(filename=args.filename)
