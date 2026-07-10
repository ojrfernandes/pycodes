#!/usr/bin/env python3
"""
Convert physical (SI) quantities to M3D-C1's normalized unit system, and
back. Covers the C1input SCALAR namelist quantities you have to convert by
hand (eta_wall, eta_zone(i), etc.) -- NOT profile/data files (profile_ne,
profile_te, current.dat, ...), which M3D-C1 reads in stated physical units
and normalizes internally; see ~/docs/M3DC1_SI_unit_conversion_guide.txt
for that distinction and the source citations behind every formula below.

All 17 conversion factors are derived from first principles in SI (not
transcribed/converted from the Gaussian-cgs forms in doc/units.tex) and
cross-checked to reproduce that doc's default numeric table exactly:

    tau0 = L0 * sqrt(mu0 * mu_i * m_p * n0) / B0        (Alfven time)
    pressure_unit = B0^2 / mu0
    <dimension>   = <combination of L0, B0, n0, tau0, mu0, m_p, e>

Usage as a CLI:
    ./m3dc1_units.py resistivity 7.4e-7
    ./m3dc1_units.py resistivity 2.6997e-7 --to-si
    ./m3dc1_units.py temperature 1500 --n0-norm 5e13   # non-default n0_norm

Usage as a module:
    from m3dc1_units import M3DC1Units
    u = M3DC1Units.from_c1input(l0_norm=100., b0_norm=1.e4, n0_norm=1e14)
    u.to_normalized(7.4e-7, 'resistivity')   # -> 2.6997e-7
    u.to_si(2.6997e-7, 'resistivity')        # -> 7.4e-7
"""

import argparse

# Physical constants (SI, CODATA)
MU0 = 4.0e-7 * 3.141592653589793   # permeability of free space, H/m
MP = 1.67262192369e-27             # proton mass, kg
QE = 1.602176634e-19               # elementary charge, C (J per eV)

# (SI unit label, human-readable description) per dimension, for CLI output
UNIT_LABELS = {
    'length':               ('m',        'length'),
    'time':                 ('s',        'time'),
    'bfield':                ('T',        'magnetic field'),
    'density':              ('m^-3',     'number density'),
    'velocity':              ('m/s',      'velocity'),
    'current':               ('A',        'current'),
    'current_density':      ('A/m^2',    'current density'),
    'efield':                 ('V/m',      'electric field'),
    'pressure':               ('Pa',       'pressure'),
    'energy':                 ('J',        'energy'),
    'force':                  ('N',        'force'),
    'resistivity':            ('Ohm*m',    'resistivity'),
    'temperature':            ('eV',       'temperature'),
    'diffusivity':            ('m^2/s',    'diffusivity'),
    'thermal_conductivity':   ('1/(m*s)',  'thermal conductivity'),
    'viscosity':              ('kg/(m*s)', 'viscosity'),
}


class M3DC1Units:
    """Normalized-unit system for one choice of L0, B0, n0, ion_mass."""

    def __init__(self, L0=1.0, B0=1.0, n0=1e20, ion_mass=1.0):
        """
        L0: length norm, meters
        B0: field norm, Tesla
        n0: density norm, m^-3
        ion_mass: main ion atomic mass number (dimensionless, C1input ion_mass)
        """
        self.L0 = L0
        self.B0 = B0
        self.n0 = n0
        self.mu_i = ion_mass
        self.tau0 = L0 * (MU0 * ion_mass * MP * n0) ** 0.5 / B0

    @classmethod
    def from_c1input(cls, l0_norm=100., b0_norm=1.e4, n0_norm=1e14, ion_mass=1.):
        """
        Build directly from a C1input's own units:
          l0_norm in cm, b0_norm in Gauss, n0_norm in cm^-3 (as C1input has them).
        Defaults match M3D-C1's own code defaults (L0=1m, B0=1T, n0=1e20 m^-3).
        """
        return cls(L0=l0_norm / 100., B0=b0_norm / 1.e4, n0=n0_norm * 1.e6,
                   ion_mass=ion_mass)

    @property
    def factors(self):
        """SI value corresponding to 1.0 in normalized units, per dimension."""
        L0, B0, n0, tau0, mu_i = self.L0, self.B0, self.n0, self.tau0, self.mu_i
        pressure = B0 ** 2 / MU0
        return {
            'length':               L0,
            'time':                 tau0,
            'bfield':                B0,
            'density':              n0,
            'velocity':              L0 / tau0,
            'current':               B0 * L0 / MU0,
            'current_density':      B0 / (MU0 * L0),
            'efield':                 L0 * B0 / tau0,
            'pressure':               pressure,
            'energy':                 L0 ** 3 * pressure,
            'force':                  L0 ** 2 * pressure,
            'resistivity':            MU0 * L0 ** 2 / tau0,
            'temperature':            pressure / n0 / QE,   # eV
            'diffusivity':            L0 ** 2 / tau0,
            'thermal_conductivity':   n0 * L0 ** 2 / tau0,
            'viscosity':              mu_i * MP * n0 * L0 ** 2 / tau0,
        }

    def to_normalized(self, value, dimension):
        return value / self.factors[dimension]

    def to_si(self, value, dimension):
        return value * self.factors[dimension]


DEFAULT = M3DC1Units()  # L0=1m, B0=1T, n0=1e20 m^-3, ion_mass=1 (code defaults)


def _cli():
    dims = list(UNIT_LABELS.keys())
    parser = argparse.ArgumentParser(
        description="Convert a physical (SI) quantity to M3D-C1 normalized "
                     "units, or back, for the C1input scalar parameters you "
                     "have to convert by hand.")
    parser.add_argument("dimension", choices=dims, help="physical dimension")
    parser.add_argument("value", type=float, help="numeric value to convert")
    parser.add_argument("--to-si", action="store_true",
                         help="convert FROM normalized TO SI (default: SI -> normalized)")
    parser.add_argument("--l0-norm", type=float, default=100.,
                         help="l0_norm from C1input, cm (default 100)")
    parser.add_argument("--b0-norm", type=float, default=1.e4,
                         help="b0_norm from C1input, Gauss (default 1e4)")
    parser.add_argument("--n0-norm", type=float, default=1e14,
                         help="n0_norm from C1input, cm^-3 (default 1e14)")
    parser.add_argument("--ion-mass", type=float, default=1.,
                         help="ion_mass from C1input (default 1 = hydrogen)")
    args = parser.parse_args()

    units = M3DC1Units.from_c1input(args.l0_norm, args.b0_norm, args.n0_norm,
                                     args.ion_mass)
    si_unit, desc = UNIT_LABELS[args.dimension]

    if args.to_si:
        result = units.to_si(args.value, args.dimension)
        print(f"{args.value:g} (normalized {desc}) = {result:.6g} {si_unit}")
    else:
        result = units.to_normalized(args.value, args.dimension)
        print(f"{args.value:g} {si_unit} ({desc}) = {result:.6g} (normalized)")


if __name__ == "__main__":
    _cli()
