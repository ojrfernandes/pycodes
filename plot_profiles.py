#!/home/jfernandes/.venv/bin/python
import fpy
import argparse
import numpy as np
import matplotlib.pyplot as plt
from m3dc1.eval_field import eval_field
from m3dc1.flux_coordinates import flux_coordinates
from m3dc1.eigenfunction import check_sim_object

def plot_profiles(filename="C1.h5", time=1, fcoords='pest', points=121):
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

    # Initialize simulation object
    sims = check_sim_object(sim=None, time=time, filename=filename)

    if isinstance(sims[0].fc,fpy.flux_coordinates)==False or (fcoords!=None and (sims[0].fc.fcoords!=fcoords)):
        sims[0] = flux_coordinates(sim=sims[0], filename=filename, fcoords=fcoords, phit=0.0, points=points, quiet=True)
    else:
        if sims[0].fc.m != points:
            sims[0] = flux_coordinates(sim=sims[0], filename=filename, fcoords=fcoords, phit=0.0, points=points, quiet=True)
    fc = sims[0].fc

    # Initialize toroidal angle array
    torphi = np.zeros_like(fc.rpath)

    # Evaluate fields
    pi = eval_field(field_name='pi', R=fc.rpath[0], phi=torphi[0], Z=fc.zpath[0], coord='scalar', sim=sims[0], time=sims[0].timeslice, quiet=True)
    pe = eval_field(field_name='pe', R=fc.rpath[0], phi=torphi[0], Z=fc.zpath[0], coord='scalar', sim=sims[0], time=sims[0].timeslice, quiet=True)
    p = eval_field(field_name='p', R=fc.rpath[0], phi=torphi[0], Z=fc.zpath[0], coord='scalar', sim=sims[0], time=sims[0].timeslice, quiet=True)
    ti = eval_field(field_name='ti', R=fc.rpath[0], phi=torphi[0], Z=fc.zpath[0], coord='scalar', sim=sims[0], time=sims[0].timeslice, quiet=True)
    te = eval_field(field_name='te', R=fc.rpath[0], phi=torphi[0], Z=fc.zpath[0], coord='scalar', sim=sims[0], time=sims[0].timeslice, quiet=True)
    ni = eval_field(field_name='ni', R=fc.rpath[0], phi=torphi[0], Z=fc.zpath[0], coord='scalar', sim=sims[0], time=sims[0].timeslice, quiet=True)
    ne = eval_field(field_name='ne', R=fc.rpath[0], phi=torphi[0], Z=fc.zpath[0], coord='scalar', sim=sims[0], time=sims[0].timeslice, quiet=True)
    v_phi = eval_field(field_name='v', R=fc.rpath[0], phi=torphi[0], Z=fc.zpath[0], coord='phi', sim=sims[0], time=sims[0].timeslice, quiet=True)
    j_R = eval_field(field_name='j', R=fc.rpath[0], phi=torphi[0], Z=fc.zpath[0], coord='R', sim=sims[0], time=sims[0].timeslice, quiet=True)
    j_phi = eval_field(field_name='j', R=fc.rpath[0], phi=torphi[0], Z=fc.zpath[0], coord='phi', sim=sims[0], time=sims[0].timeslice, quiet=True)
    j_Z = eval_field(field_name='j', R=fc.rpath[0], phi=torphi[0], Z=fc.zpath[0], coord='Z', sim=sims[0], time=sims[0].timeslice, quiet=True)

    # Create plots
    fig, axs = plt.subplots(2, 3, figsize=(22, 10))

    # Plot q profile
    axs[0, 0].plot(fc.psi_norm[1:], np.abs(fc.q[1:]), color=colors[3], label='q')
    axs[0, 0].set_xlim(fc.psi_norm[1],fc.psi_norm[-1])
    axs[0, 0].set_ylim(0, np.abs(fc.q[-1]))
    axs[0, 0].set_xlabel('Normalized poloidal flux')
    axs[0, 0].set_ylabel('q')
    axs[0, 0].legend()
    axs[0, 0].grid()

    # Plot pressure profiles
    axs[0, 1].plot(fc.psi_norm, pe*1e-3, label='P$_{\,\mathrm{e}}$', color=colors[1])
    axs[0, 1].plot(fc.psi_norm, pi*1e-3, label='P$_{\,\mathrm{i}}$', color=colors[3])
    axs[0, 1].plot(fc.psi_norm, p*1e-3, label='P', color=colors[5])
    axs[0, 1].set_xlim(0,1)
    axs[0, 1].set_ylim(0,)
    axs[0, 1].set_xlabel('Normalized poloidal flux')
    axs[0, 1].set_ylabel('Pressure ( kPa )')
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Plot temperature profiles
    axs[0, 2].plot(fc.psi_norm, te, label='T$_{\,\mathrm{e}}$', color=colors[1])
    axs[0, 2].plot(fc.psi_norm, ti, label='T$_{\,\mathrm{i}}$', color=colors[3])
    axs[0, 2].set_xlim(0,1)
    axs[0, 2].set_ylim(0,)
    axs[0, 2].set_xlabel('Normalized poloidal flux')
    axs[0, 2].set_ylabel('Temperature ( eV )')
    axs[0, 2].legend()
    axs[0, 2].grid()

    # Plot density profiles
    axs[1, 0].plot(fc.psi_norm, ne, label='n$_{\,\mathrm{e}}$', color=colors[1])
    axs[1, 0].plot(fc.psi_norm, ni, label='n$_{\,\mathrm{i}}$', color=colors[3])
    axs[1, 0].set_xlim(0,1)
    axs[1, 0].set_ylim(0,)
    axs[1, 0].set_xlabel('Normalized poloidal flux')
    axs[1, 0].set_ylabel('Density ( m$^{-3}$ )')
    axs[1, 0].legend()
    axs[1, 0].grid()

    # Plot toroidal velocity profile
    axs[1, 1].plot(fc.psi_norm, v_phi/1e3, label='v$_{\,\mathrm{\phi}}$', color=colors[3])
    axs[1, 1].set_xlim(0,1)
    # axs[1, 1].set_ylim(0,)
    axs[1, 1].set_xlabel('Normalized poloidal flux')
    axs[1, 1].set_ylabel('Velocity ( km/s )')
    axs[1, 1].legend()
    axs[1, 1].grid()

    # Plot current density profiles
    axs[1, 2].plot(fc.psi_norm, j_phi/1e3, label='j$_{\,\mathrm{\phi}}$', color=colors[1])
    axs[1, 2].plot(fc.psi_norm, -np.sqrt(j_Z**2 + j_R**2)/1e3, label='j$_{\,\mathrm{R+Z}}$', color=colors[3])
    axs[1, 2].set_xlim(0,1)
    # axs[1, 2].set_ylim(-1000,)
    axs[1, 2].set_xlabel('Normalized poloidal flux')
    axs[1, 2].set_ylabel('Current density ( kA$\,/\,$m$^2$ )')
    axs[1, 2].legend()
    axs[1, 2].grid()

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot M3D-C1 equilibrium profiles.')
    parser.add_argument('--filename', type=str, default='C1.h5', help='Path to the M3D-C1 HDF5 file.')
    parser.add_argument('--time', type=int, default=1, help='Time slice to plot.')
    parser.add_argument('--fcoords', type=str, default='pest', help='Flux coordinate system to use.')
    parser.add_argument('--points', type=int, default=121, help='Number of points along the flux surface.')
    args = parser.parse_args()

    plot_profiles(filename=args.filename,
                  time=args.time,
                  fcoords=args.fcoords,
                  points=args.points)