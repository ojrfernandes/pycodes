import aug_sfutils as sf
import numpy as np
import matplotlib.pyplot as plt


class vtaData:
    """
    Class for loading, storing, and analysing VTA (Thomson Scattering) data
    from AUG shotfiles.
    """

    PREFIXES = {
        'e': 'edge',
        'c': 'core',
    }

    def __init__(self, shot: int, edition: int = 0):
        self.shot    = shot
        self.edition = edition
        self._sfread = sf.SFREAD(shot, 'vta')
        if not self._sfread.status:
            raise ValueError(f"VTA shotfile for shot {shot} could not be opened.")
        self.equ = sf.EQU(shot, diag='EQI', ed=edition)

        self.R = {p: np.asarray(self._sfread.getobject(f'R_{name}'))
                  for p, name in self.PREFIXES.items()}
        self.Z = {p: np.asarray(self._sfread.getobject(f'Z_{name}'))
                  for p, name in self.PREFIXES.items()}

        self.time = {p: np.asarray(self._sfread.gettimebase(f'Te_{p}'))
                     for p in self.PREFIXES}

        self._signals = {}
        for p in self.PREFIXES:
            self._signals[p] = {
                'Te':     np.asarray(self._sfread.getobject(f'Te_{p}')),
                'Te_low': np.asarray(self._sfread.getobject(f'Telow_{p}')),
                'Te_upp': np.asarray(self._sfread.getobject(f'Teupp_{p}')),
                'Ne':     np.asarray(self._sfread.getobject(f'Ne_{p}')),
                'Ne_low': np.asarray(self._sfread.getobject(f'Nelow_{p}')),
                'Ne_upp': np.asarray(self._sfread.getobject(f'Neupp_{p}')),
            }


    #  Helpers

    def available_objects(self) -> list:
        """Return the list of all objects available in the VTA shotfile."""
        return self._sfread.getlist()

    def time_index(self, t: float, prefix: str) -> int:
        """Return the index in the time base closest to *t* for a given prefix."""
        return int(np.argmin(np.abs(self.time[prefix] - t)))

    def time_range(self) -> dict:
        """Return the min/max time for each region."""
        return {p: (self.time[p].min(), self.time[p].max())
                for p in self.PREFIXES}


    #  Profile retrieval

    def get_profile(self, t: float, prefix: str) -> dict:
        """
        Extract a radial profile at time *t* for 'e' (edge) or 'c' (core).

        Returns a dict with keys:
            rho, Te, Te_err, Ne, Ne_err
        """
        if prefix not in self.PREFIXES:
            raise ValueError(f"prefix must be one of {list(self.PREFIXES)}")

        idx   = self.time_index(t, prefix)
        R_sel = self.R[prefix][idx]
        Z     = self.Z[prefix]

        # Map (R, Z) → rho_pol via equilibrium reconstruction
        R_arr = np.full_like(Z, R_sel)
        rho   = np.asarray(
            sf.rz2rho(self.equ, R_arr, Z, t_in=t, coord_out='rho_pol')
        ).squeeze()

        sig = self._signals[prefix]
        return {
            'rho':    rho,
            'Te':     sig['Te'][idx],
            'Te_err': [sig['Te'][idx] - sig['Te_low'][idx],
                       sig['Te_upp'][idx] - sig['Te'][idx]],
            'Ne':     sig['Ne'][idx],
            'Ne_err': [sig['Ne'][idx] - sig['Ne_low'][idx],
                       sig['Ne_upp'][idx] - sig['Ne'][idx]],
            'time':   self.time[prefix][idx],
            'prefix': prefix,
        }

    def get_profiles(self, t: float) -> dict:
        """Return both edge and core profiles at time *t*."""
        return {p: self.get_profile(t, p) for p in self.PREFIXES}


    #  Data manipulation                                                 

    def combine_profiles(self, t: float) -> dict:
        """
        Merge core and edge profiles into a single sorted radial profile.
        Useful for fitting or further analysis.
        """
        profiles = self.get_profiles(t)
        rho_all  = np.concatenate([profiles['c']['rho'], profiles['e']['rho']])
        sort_idx = np.argsort(rho_all)

        def concat_err(key):
            low = np.concatenate([profiles['c'][key][0], profiles['e'][key][0]])
            upp = np.concatenate([profiles['c'][key][1], profiles['e'][key][1]])
            return [low[sort_idx], upp[sort_idx]]

        return {
            'rho':    rho_all[sort_idx],
            'Te':     np.concatenate([profiles['c']['Te'], profiles['e']['Te']])[sort_idx],
            'Te_err': concat_err('Te_err'),
            'Ne':     np.concatenate([profiles['c']['Ne'], profiles['e']['Ne']])[sort_idx],
            'Ne_err': concat_err('Ne_err'),
        }

    def clean_by_error(self, profile: dict, threshold: float = 0.5) -> dict:
        """
        Remove datapoints where either the lower or upper error exceeds
        'threshold' * |value| for Te or Ne.

        Parameters
        ----------
        profile   : dict — output of get_profile(), get_profiles(), or combine_profiles()
        threshold : float — fractional error threshold (default 0.5 = 50%)

        Returns
        -------
        A cleaned copy of the profile dict with bad points masked out.
        """
        cleaned = profile.copy()
        combined_mask = np.ones(len(profile['rho']), dtype=bool)

        # First loop: build the combined mask across both quantities
        for qty in ('Te', 'Ne'):
            val     = profile[qty]
            err_low = profile[f'{qty}_err'][0]
            err_upp = profile[f'{qty}_err'][1]

            mask = (
                (err_low <= threshold * np.abs(val)) &
                (err_upp <= threshold * np.abs(val))
            )
            combined_mask &= mask

        # Second loop: apply the combined mask to everything
        cleaned['rho'] = profile['rho'][combined_mask]
        for qty in ('Te', 'Ne'):
            cleaned[qty]          = profile[qty][combined_mask]
            cleaned[f'{qty}_err'] = [profile[f'{qty}_err'][0][combined_mask],
                                     profile[f'{qty}_err'][1][combined_mask]]

        return cleaned


    #  Plotting

    def _plot_clean(self, profile: dict, clean: bool = False,
                    threshold: float = 0.5) -> dict:
        """
        Helper function to optionally clean a profile before protting.
        """
        return self.clean_by_error(profile, threshold) if clean else profile

    def _plot_quantity(self, ax, rho, val, err, label, error_style, **kwargs):
        """
        Plot a single quantity with either errorbars or a filled band.

        Parameters
        ----------
        error_style : 'bars' — classic errorbars
                      'fill' — shaded band between lower and upper limits
        """

        # Sort by rho for monotonic plotting
        sort_idx = np.argsort(rho)
        rho      = rho[sort_idx]
        val      = val[sort_idx]

        err_low, err_upp = err

        if error_style == 'bars':
            ax.errorbar(rho, val, yerr=[err_low, err_upp], fmt='o',
                        label=label, **kwargs)

        elif error_style == 'fill':
            color     = kwargs.pop('color', None)
            alpha     = kwargs.pop('alpha', 0.3)
            fmt       = kwargs.pop('fmt',   'o')
            linewidth = kwargs.pop('linewidth', 1.5)

            # Plot central values
            line, = ax.plot(rho, val, fmt, color=color,
                            linewidth=linewidth, label=label, **kwargs)

            # Shade
            ax.fill_between(rho,
                            val - err_low,
                            val + err_upp,
                            alpha=alpha,
                            color=line.get_color())
        else:
            raise ValueError(f"error_style must be 'bars' or 'fill', got '{error_style}'")                                                            

    def plot_profiles(self, t: float,
                      ax: np.ndarray | None = None,
                      fig: plt.Figure | None = None,
                      clean: bool = False,
                      threshold: float = 0.5,
                      figsize: tuple = (10, 6),
                      dpi: int = 100,
                      error_style: str = 'bars',
                      **kwargs) -> tuple:
        """
        Plot Te and Ne profiles for both edge and core at time *t*.

        Parameters
        ----------
        t   : float — requested time (s)
        ax  : optional 2×2 array of Axes; created if None
        fig : optional Figure handle
        clean :     bool - apply error based cleaning before plotting (default False)
        threshold : float - fractional error threshold for cleaning (default 0.5)
        figsize   : tuple - figure size (default (10, 6))
        dpi       : int   - figure resolution (default 100)
        error_style : str - style for displaying errors ('bars' or 'fill')
        **kwargs   : additional keyword arguments for the errorbar plots

        Returns
        -------
        (fig, ax)
        """
        raw_profiles = self.get_profiles(t)
        profiles = {p: self._plot_clean(prof, clean=clean, threshold=threshold)
                    for p, prof in raw_profiles.items()}

        if ax is None:
            fig, ax = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        fig = fig or ax.flat[0].get_figure() 

        plot_cfg = [
            (0, 'c', 'Te', r'$T_e$ (eV)',       'core'),
            (1, 'c', 'Ne', r'$n_e$ (m$^{-3}$)', 'core'),
            (2, 'e', 'Te', r'$T_e$ (eV)',       'edge'),
            (3, 'e', 'Ne', r'$n_e$ (m$^{-3}$)', 'edge'),
        ]

        for flat_idx, prefix, qty, ylabel, label in plot_cfg:
            row, col = divmod(flat_idx, 2)
            p = profiles[prefix]
            self._plot_quantity(
                ax[row, col],
                p['rho'], p[qty], p[f'{qty}_err'],
                label=f'VTA ({label}), t≈{p["time"]:.3f} s',
                error_style=error_style,
                **kwargs
            )
            ax[row, col].set_xlabel(r'$\rho_p$')
            ax[row, col].set_ylabel(ylabel)
            ax[row, col].legend()

        fig.suptitle(f'VTA profiles — shot {self.shot}, t ≈ {t:.3f} s')
        fig.tight_layout()
        return fig, ax

    def plot_combined(self, t: float, clean: bool = False, threshold: float = 0.5,
                      figsize: tuple = (10, 4), dpi: int = 100, error_style: str = 'bars', **kwargs) -> tuple:
        """
        Plot a single merged core + edge profile for Te and Ne at time *t*.
 
        Parameters
        ----------
        t   : float — requested time (s)
        clean     : bool - apply error cleaning before plotting (default False)
        threshold : float - fractional error threshold for cleaning (default 0.5)
        figsize   : tuple - figure size (default (10, 4))
        dpi       : int   - figure resolution (default 100)
        error_style : str - style for displaying errors ('bars' or 'fill')
        **kwargs   : additional keyword arguments for the errorbar plots
        """
        combined = self.combine_profiles(t)
        combined = self._plot_clean(combined, clean=clean, threshold=threshold)

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

        for ax, qty, ylabel in zip(
            axes,
            ['Te', 'Ne'],
            [r'$T_e$ (eV)', r'$n_e$ (m$^{-3}$)'],
        ):
            self._plot_quantity(
                ax,
                combined['rho'], combined[qty], combined[f'{qty}_err'],
                label=f'VTA (combined), t≈{combined["time"]:.3f} s',
                error_style=error_style,
                **kwargs
            )
            ax.set_xlabel(r'$\rho_p$')
            ax.set_ylabel(ylabel)

        fig.suptitle(f'VTA combined profile — shot {self.shot}, t ≈ {t:.3f} s')
        fig.tight_layout()
        return fig, axes

    def __repr__(self) -> str:
        tr = self.time_range()
        return (f"vtaData(shot={self.shot}, edition={self.edition}, "
                f"t_edge=[{tr['e'][0]:.3f}, {tr['e'][1]:.3f}] s, "
                f"t_core=[{tr['c'][0]:.3f}, {tr['c'][1]:.3f}] s)")
