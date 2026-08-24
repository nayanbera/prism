#!/usr/bin/env python3
"""
Synthetic ASAXS test data: Au-core / SiO2-shell nanoparticles in water.
Monodisperse — single particle size, no size distribution.

Physics
-------
  Au core   : R_core = 55 Å  (fixed, monodisperse)
  SiO2 shell: shell thickness Δ = 165 Å → R_total = 220 Å
  Energies  : 20 points in [10.919, 11.919] keV chosen for equidistant f'(Au L3)

Decomposition identity (for validation)
  I(q,E) = I_MM(q) + 2f'(E)·I_RM(q) + [f'²(E)+f''²(E)]·I_RR(q)
"""

import numpy as np
from pathlib import Path

try:
    import xraydb
except ImportError:
    raise SystemExit("xraydb required:  pip install xraydb")

OUT_DIR = Path(__file__).parent / "au_sio2_mono"

# ── Parameters ────────────────────────────────────────────────────────────────
R_CORE     = 55.0    # Å  Au-core radius (monodisperse)
R_TOTAL    = 220.0   # Å  total outer radius
N_ENERGIES = 20      # number of ASAXS energy points
E_MIN      = 10.919  # keV
E_MAX      = 11.919  # keV  (Au L3 edge)
PHI        = 1e-4    # particle volume fraction
NOISE      = 0.005   # 0.5% relative Gaussian noise

# ── Electron densities in e⁻/Å³ ─────────────────────────────────────────────
rho_Au    = (19.32  / 196.97) * 6.022e23 * 79  * 1e-24
rho_SiO2  = (2.196  / 60.08 ) * 6.022e23 * 30  * 1e-24
rho_H2O   = (1.000  / 18.015) * 6.022e23 * 10  * 1e-24
V_atom_Au = 196.97  / (19.32  * 6.022e23) * 1e24

drho_Au   = rho_Au   - rho_H2O
drho_SiO2 = rho_SiO2 - rho_H2O
DELTA     = R_TOTAL - R_CORE

# ── q grid ────────────────────────────────────────────────────────────────────
q = np.geomspace(0.003, 0.145, 500)

# ── Form factor (monodisperse — no distribution integral) ─────────────────────
def sphere_f0(q, R):
    x = q * R
    return np.where(np.abs(x) < 1e-8, 1.0,
                    3.0 * (np.sin(x) - x * np.cos(x)) / x**3)

def sphere_V(R):
    return (4.0/3.0) * np.pi * R**3

V_core = sphere_V(R_CORE)
V_tot  = sphere_V(R_TOTAL)

f0_core = sphere_f0(q, R_CORE)   # (Nq,)
f0_tot  = sphere_f0(q, R_TOTAL)  # (Nq,)

F_non = (drho_Au - drho_SiO2) * V_core * f0_core + drho_SiO2 * V_tot * f0_tot
F_res = V_core / V_atom_Au * f0_core

I_MM_raw = F_non**2
I_RM_raw = F_non * F_res
I_RR_raw = F_res**2

# ── Convert to cm⁻¹ ──────────────────────────────────────────────────────────
r_e_cm = 2.818e-13
N_p_cm = PHI / V_tot * 1e24
SCALE  = N_p_cm * r_e_cm**2

I_MM = SCALE * I_MM_raw
I_RM = SCALE * I_RM_raw
I_RR = SCALE * I_RR_raw

# ── Choose energies with equidistant f' spacing ───────────────────────────────
E_dense   = np.linspace(E_MIN, E_MAX, 3000)
Z_Au      = 79
fp_dense  = np.array([xraydb.f1_chantler(Z_Au, e * 1000) for e in E_dense])
fpp_dense = np.array([abs(xraydb.f2_chantler(Z_Au, e * 1000)) for e in E_dense])

sort_idx   = np.argsort(fp_dense)
fp_sorted  = fp_dense[sort_idx]
E_sorted   = E_dense[sort_idx]

fp_targets = np.linspace(fp_sorted[0], fp_sorted[-1], N_ENERGIES)
energies   = sorted(np.interp(fp_targets, fp_sorted, E_sorted))

fp_vals    = np.array([xraydb.f1_chantler(Z_Au, e * 1000) for e in energies])
fpp_vals   = np.array([abs(xraydb.f2_chantler(Z_Au, e * 1000)) for e in energies])

print("Selected energies (equidistant f' spacing):")
print(f"  {'E (keV)':>10}  {'fp (e)':>8}  {'fpp (e)':>8}")
print(f"  {'-'*34}")
for e, fp, fpp in zip(energies, fp_vals, fpp_vals):
    print(f"  {e:10.4f}  {fp:8.3f}  {fpp:8.3f}")

# ── Generate I(q,E) and save .dat files ──────────────────────────────────────
rng = np.random.default_rng(42)
OUT_DIR.mkdir(parents=True, exist_ok=True)

for E, fp, fpp in zip(energies, fp_vals, fpp_vals):
    I_tot  = I_MM + 2*fp * I_RM + (fp**2 + fpp**2) * I_RR
    sigma  = np.maximum(NOISE * I_tot, 1e-6 * I_tot.max())
    I_obs  = np.maximum(I_tot + rng.normal(0, 1, len(q)) * sigma, 1e-8)

    fname  = OUT_DIR / f"Au_SiO2_mono_{E:.4f}keV.dat"
    header = (
        f"Synthetic Au-core/SiO2-shell ASAXS test data  [MONODISPERSE]\n"
        f"  R_core = {R_CORE} A,  R_total = {R_TOTAL} A  (no size distribution)\n"
        f"  phi = {PHI:.2e},  noise = {NOISE*100:.1f}%\n"
        f"  Energy={E:.4f} keV,  f'(Au L3) = {fp:.4f} e,  f'' = {fpp:.4f} e\n"
        f"  Columns: q(1/A)  I(cm-1)  sigma(cm-1)"
    )
    np.savetxt(fname, np.column_stack([q, I_obs, sigma]),
               header=header, fmt="%.6e")

print(f"\nSaved {N_ENERGIES} files to:  {OUT_DIR}/")
print(f"\nReference partial structure factors at q = 0.01 Å⁻¹:")
q01 = 0.01
print(f"  I_MM = {np.interp(q01, q, I_MM):.4e} cm⁻¹")
print(f"  I_RM = {np.interp(q01, q, I_RM):.4e} cm⁻¹")
print(f"  I_RR = {np.interp(q01, q, I_RR):.4e} cm⁻¹")
print(f"  I_RM/I_MM = {np.interp(q01, q, I_RM)/np.interp(q01, q, I_MM):.4f}")

np.savetxt(OUT_DIR / "ground_truth_partials.dat",
           np.column_stack([q, I_MM, I_RM, I_RR]),
           header="q(1/A)  I_MM(cm-1)  I_RM(cm-1)  I_RR(cm-1)\n"
                  f"  R_core={R_CORE}A (monodisperse), R_total={R_TOTAL}A, phi={PHI:.2e}",
           fmt="%.6e")
print("Ground-truth partials saved: ground_truth_partials.dat")
