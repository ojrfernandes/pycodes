#!/home/jfernandes/.venv/bin/python
import argparse
import os
import numpy as np

def flare_model_gen(coils: str, directory: str, n_tor: int, d_phase: int, sets: str, timeslice: int, boundary: str=None, amplitudes: list=[1.0,1.0,1.0], flare_phase: bool=True, phase_signal: list=[-1,1]) -> None:
    """
    Generate flare model files for different phase combinations of IL and IU sets.

    Parameters
    ----------
    coils : str
        Type of coils to use. Options are 'I' or 'CP'.
    directory : str
        Path to the directory where the model files will be saved.
    n_tor : int
        Toroidal mode number.
    d_phase : int
        Phase difference increment in degrees.
    sets : str
        Path to the directory containing IM, IL, IU sets.
    timeslice : int
        Time slice to use from the M3D-C1 data.
    boundary : str
        Path to the boundary file. Default is tcabr_first_wall.txt.
    amplitudes : list of float
        Amplitudes for L, M, U sets respectively. Default is [1.0, 1.0, 1.0].
    flare_phase : bool
        If True, use phase 000 simulations and adjust phase in FLARE post-processing. Default is True.
    phase_signal : list of int
        (only applies if flare_phase is True)
        Phase signal for L and U sets respectively. Default is [-1, 1].
    
    Returns
    -------
    None
        Creates directories and model files for each phase combination.
    """

    if boundary is None:
        boundary = "/home/jfernandes/machines_geo/input_geo/tcabr_first_wall.txt"

    n_models = int((360 / n_tor / d_phase) + 1)

    #check if directory and sets end in '/'
    if not directory.endswith('/'):
        directory += '/'
    if not sets.endswith('/'):
        sets += '/'

    #check if directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    if coils not in ['I', 'CP']:
        raise ValueError("Invalid coils type. Options are 'I' or 'CP'.")
    if coils == 'I':
        for i in range(n_models):
            for j in range(n_models):
                phase_IL = int(i * d_phase * phase_signal[0])
                phase_IU = int(j * d_phase * phase_signal[1])
                phase_IL_str = f'{np.abs(phase_IL):03d}'
                phase_IU_str = f'{np.abs(phase_IU):03d}'
                new_dir = f'{directory}dephase_IL_{phase_IL_str}_IU_{phase_IU_str}/'

                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                else:
                    print(f'Directory {new_dir} already exists. Skipping...')
                    continue

                with open(f'{new_dir}.boundary', 'w') as file:
                    file.write(
                    '[axisurf]\n' \
                    f'filename: {boundary}\n' \
                    'units: m')
                
                if flare_phase:
                    with open(f'{new_dir}.bfield', 'w') as file:
                        file.write(
                        '[equi2d_m3dc1]\n' \
                        f'filename: {sets}IM_set_000/C1.h5\n' \

                        '\n' \
                        
                        '[IL_set:m3dc1]\n' \
                        f'filename:   {sets}IL_set_000/C1.h5\n' \
                        f'timeslice:  {timeslice}\n' \
                        f'amplitude:  {amplitudes[0]}\n' \
                        f'phase:      {phase_IL}\n' \
                        
                        '\n' \
                        
                        '[IM_set:m3dc1]\n' \
                        f'filename:   {sets}IM_set_000/C1.h5\n' \
                        f'timeslice:  {timeslice}\n' \
                        f'amplitude:  {amplitudes[1]}\n' \
                        f'phase:      0.0\n' \
                        
                        '\n' \
                        
                        '[IU_set:m3dc1]\n' \
                        f'filename:   {sets}IU_set_000/C1.h5\n' \
                        f'timeslice:  {timeslice}\n' \
                        f'amplitude:  {amplitudes[2]}\n' \
                        f'phase:      {phase_IU}\n' \
                        )
                else:
                    with open(f'{new_dir}.bfield', 'w') as file:
                        file.write(
                        '[equi2d_m3dc1]\n' \
                        f'filename: {sets}IM_set_000/C1.h5\n' \

                        '\n' \
                        
                        '[IL_set:m3dc1]\n' \
                        f'filename:   {sets}IL_set_{phase_IL_str}/C1.h5\n' \
                        f'timeslice:  {timeslice}\n' \
                        f'amplitude:  {amplitudes[0]}\n' \
                        f'phase:      0.0\n' \
                        
                        '\n' \
                        
                        '[IM_set:m3dc1]\n' \
                        f'filename:   {sets}IM_set_000/C1.h5\n' \
                        f'timeslice:  {timeslice}\n' \
                        f'amplitude:  {amplitudes[1]}\n' \
                        f'phase:      0.0\n' \
                        
                        '\n' \
                        
                        '[IU_set:m3dc1]\n' \
                        f'filename:   {sets}IU_set_{phase_IU_str}/C1.h5\n' \
                        f'timeslice:  {timeslice}\n' \
                        f'amplitude:  {amplitudes[2]}\n' \
                        f'phase:      0.0' \
                        )

    if coils == 'CP':
        for i in range(n_models):
            for j in range(n_models):
                phase_CPL = int(i * d_phase * phase_signal[0])
                phase_CPU = int(j * d_phase * phase_signal[1])
                phase_CPL_str = f'{np.abs(phase_CPL):03d}'
                phase_CPU_str = f'{np.abs(phase_CPU):03d}'
                new_dir = f'{directory}dephase_CPL_{phase_CPL_str}_CPU_{phase_CPU_str}/'

                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                else:
                    print(f'Directory {new_dir} already exists. Skipping...')
                    continue

                with open(f'{new_dir}.boundary', 'w') as file:
                    file.write(
                    '[axisurf]\n' \
                    f'filename: {boundary}\n' \
                    'units: m')

                if flare_phase:
                    with open(f'{new_dir}.bfield', 'w') as file:
                        file.write(
                        '[equi2d_m3dc1]\n' \
                        f'filename: {sets}CPM_set_000/C1.h5\n' \

                        '\n' \
                        
                        '[CPL_set:m3dc1]\n' \
                        f'filename:   {sets}CPL_set_000/C1.h5\n' \
                        f'timeslice:  {timeslice}\n' \
                        f'amplitude:  {amplitudes[0]}\n' \
                        f'phase:      {phase_CPL}\n' \

                        '\n' \
                        
                        '[CPM_set:m3dc1]\n' \
                        f'filename:   {sets}CPM_set_000/C1.h5\n' \
                        f'timeslice:  {timeslice}\n' \
                        f'amplitude:  {amplitudes[1]}\n' \
                        f'phase:      0.0\n' \
                        
                        '\n' \
                        
                        '[CPU_set:m3dc1]\n' \
                        f'filename:   {sets}CPU_set_000/C1.h5\n' \
                        f'timeslice:  {timeslice}\n' \
                        f'amplitude:  {amplitudes[2]}\n' \
                        f'phase:      {phase_CPU}\n' \
                        )
                else: 
                    with open(f'{new_dir}.bfield', 'w') as file:
                        file.write(
                        '[equi2d_m3dc1]\n' \
                        f'filename: {sets}CPM_set_000/C1.h5\n' \

                        '\n' \
                        
                        '[CPL_set:m3dc1]\n' \
                        f'filename:   {sets}CPL_set_{phase_CPL_str}/C1.h5\n' \
                        f'timeslice:  {timeslice}\n' \
                        f'amplitude:  {amplitudes[0]}\n' \
                        f'phase:      0.0\n' \

                        '\n' \
                        
                        '[CPM_set:m3dc1]\n' \
                        f'filename:   {sets}CPM_set_000/C1.h5\n' \
                        f'timeslice:  {timeslice}\n' \
                        f'amplitude:  {amplitudes[1]}\n' \
                        f'phase:      0.0\n' \
                        
                        '\n' \
                        
                        '[CPU_set:m3dc1]\n' \
                        f'filename:   {sets}CPU_set_{phase_CPU_str}/C1.h5\n' \
                        f'timeslice:  {timeslice}\n' \
                        f'amplitude:  {amplitudes[2]}\n' \
                        f'phase:      0.0' \
                        )
    print(f'Successfully created model files in {directory}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate flare model files for different phase combinations.')
    parser.add_argument('coils', type=str, help="Type of coils to use. Options are 'I' or 'CP'.")
    parser.add_argument('directory', type=str, help='Directory to save the generated model files.')
    parser.add_argument('n_tor', type=int, help='Toroidal mode number.')
    parser.add_argument('d_phase', type=int, help='Phase difference increment in degrees.')
    parser.add_argument('sets', type=str, help='Path to the directory containing IM, IL, IU sets.')
    parser.add_argument('timeslice', type=int, help='Time slice to use from the M3D-C1 data.')
    parser.add_argument('--boundary', type=str, default=None, help='Path to the boundary file. Default is tcabr_first_wall.txt')
    parser.add_argument('--amplitudes', type=float, nargs=3, default=[1.0, 1.0, 1.0], help='Amplitudes for IL, IM, IU sets respectively. Default is [1.0, 1.0, 1.0]')
    parser.add_argument('--flare_phase', action='store_true', help='If set, use phase 000 simulations and adjust phase in FLARE post-processing.')
    parser.add_argument('--phase_signal', type=int, nargs=2, default=[-1, 1], help='(only applies if phase_flare is set)\nPhase signal for IL and IU sets respectively. Default is [-1, 1]')

    args = parser.parse_args()

    flare_model_gen(
        args.coils,
        args.directory,
        args.n_tor,
        args.d_phase,
        args.sets,
        args.timeslice,
        boundary=args.boundary,
        amplitudes=args.amplitudes,
        flare_phase=args.flare_phase,
        phase_signal=args.phase_signal)
