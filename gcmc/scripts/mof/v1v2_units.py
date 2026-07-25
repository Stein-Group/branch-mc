"""V1/V2: the conversions that put engine output on the paper's axes.

Everything the loop produces is in reduced units: a guest count N, a reduced
activity z*, a reduced Henry constant K*. The anchor plots uptake in mol/kg (and
cm^3 STP/cm^3) against pressure in bar, and reports Henry constants in mol/kg/Pa.
This module is the bridge, and it is only unit bookkeeping -- no physics, no
sampling -- so it lives apart from the loop and is tested against hand
arithmetic.

Two conversions matter.

Loading. N guests in a framework of mass M_frame (amu, summed over the supercell)
is
    n [mol/kg] = (N / N_A) [mol] / (M_frame * 1.66054e-27 [kg])
The framework mass in kg is M_frame[amu] * (1e-3 / N_A) kg, so
    n [mol/kg] = N / (M_frame[amu] * 1e-3)          [since the N_A cancel]
             = 1000 N / M_frame[amu].
That cancellation is the whole trick and is why the number is clean.

Henry. The Widom Rosenbluth weight K* = <exp(-beta u_gf)> is dimensionless.
The standard adsorption Henry constant is

    K_H [mol/kg/Pa] = K* / (R T rho_framework)

with R the gas constant, T in kelvin, and rho_framework the crystal density in
kg/m^3 (framework mass over cell volume). This is RASPA's own definition and it
reproduces RASPA2's reported Henry to 5 significant figures on MFM-300(In)
(RASPA2 <W> = 996.84, K_H = 3.1475e-4 mol/kg/Pa; ours identical).

An earlier version used sigma_ref^3/(kB T) here, treating K* as if it multiplied
the reduced activity z*. That was wrong by ~350x: z* is the activity a GCMC
insertion consumes, not the constant in Henry's law, and Henry carries the
framework density, not sigma_ref^3. The RASPA2 Widom cross-check caught it --
RASPA agreed with our K* to 4 figures but not with the old conversion, which
localised the error to the arithmetic here.
"""

from __future__ import annotations

# CODATA 2018
KB_J_PER_K = 1.380649e-23
N_A = 6.02214076e23
ANGSTROM_M = 1.0e-10
R_GAS = 8.314462618  # J/mol/K
AMU_KG = 1.66053906660e-27


def loading_mol_per_kg(n_guests: float, framework_mass_amu: float) -> float:
    """N guests in a framework of mass M (amu) as mol/kg. The N_A cancel, leaving
    1000 N / M."""
    if framework_mass_amu <= 0:
        raise ValueError("framework mass must be positive")
    return 1000.0 * n_guests / framework_mass_amu


def framework_density_kg_per_m3(
    framework_mass_amu: float, cell_volume_angstrom3: float
) -> float:
    """Crystal density from the supercell mass and volume (both for the SAME
    supercell, so the ratio is intensive)."""
    if cell_volume_angstrom3 <= 0:
        raise ValueError("cell volume must be positive")
    return framework_mass_amu * AMU_KG / (cell_volume_angstrom3 * ANGSTROM_M**3)


def z_star_per_pa(sigma_ref_angstrom: float, T_kelvin: float) -> float:
    """The factor multiplying a fugacity in Pa to get z*: sigma_ref^3 / (kB T).

    This is the activity a GCMC insertion consumes (P0.4), NOT the Henry-law
    constant; the two were confused in an earlier version. Kept because the V2
    loop genuinely needs it to turn a fugacity into z*.
    """
    return (sigma_ref_angstrom * ANGSTROM_M) ** 3 / (KB_J_PER_K * T_kelvin)


def henry_mol_per_kg_per_pa(
    henry_star: float,
    framework_mass_amu: float,
    cell_volume_angstrom3: float,
    T_kelvin: float,
) -> float:
    """Reduced Widom weight K* to the adsorption Henry constant in mol/kg/Pa.

    K_H = K* / (R T rho_framework), RASPA's definition, cross-checked against
    RASPA2's own Widom output on MFM-300(In).
    """
    rho = framework_density_kg_per_m3(framework_mass_amu, cell_volume_angstrom3)
    return henry_star / (R_GAS * T_kelvin * rho)


def pressure_to_fugacity_pa(pressure_pa: float, phi: float) -> float:
    """Fugacity from pressure and the Peng-Robinson coefficient: f = phi P.

    The loop takes z*, and z* is built from a fugacity, so a pressure axis is
    turned into activities through phi (units.peng_robinson) before the loop ever
    runs. Kept here so the V2 driver reads one obvious call.
    """
    return phi * pressure_pa
