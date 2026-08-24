#!/usr/bin/env python3
"""
General ASAXS synthetic data generator for core-shell nanoparticles.

Usage
-----
  python generate_asaxs_data.py <config_file.py>

The config file must define a dict named CONFIG.  See config_au_sio2_poly.py
for a fully-annotated example.

Output
------
  <output.directory>/
    <name>_<E:.4f>keV.dat          — 3-column (q  I  sigma) per energy
    ground_truth_partials.dat       — 4-column (q  I_MM  I_RM  I_RR)

Physics
-------
  I(q, E) = I_MM(q) + 2f'(E)·I_RM(q) + [f'²(E) + f''²(E)]·I_RR(q)

  For a core-shell particle of core radius R_core and total radius R_total:
    F_non(q, R) = (Δρ_core − Δρ_shell)·V_core·f₀(q, R)
                + Δρ_shell·V_total·f₀(q, R_total)
    F_res(q, R) = (V_core / V_atom_res)·f₀(q, R)   [resonant electrons]

  I_XX = N_p · r_e² · ∫ P(R) F_X(q,R) F_X(q,R) dR

Noise models
------------
  poisson  : σ(q) = √(I(q)·I(q_min)/N_peak)   — photon counting statistics
             Relative noise at q_min = 1/√N_peak; grows as signal falls.
  relative : σ(q) = relative_noise · I(q)       — flat percentage noise
"""

import sys
import importlib.util
import numpy as np
from pathlib import Path

try:
    import xraydb
except ImportError:
    raise SystemExit("xraydb is required:  pip install xraydb")


# ─── Config loader ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("_cfg", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "CONFIG"):
        raise ValueError(f"{path}: must define a dict named CONFIG")
    return mod.CONFIG


# ─── Electron density helper ──────────────────────────────────────────────────

def electron_density(mass_density, molar_mass, n_electrons):
    """Electron density in e⁻/Å³."""
    N_A = 6.022e23
    return (mass_density / molar_mass) * N_A * n_electrons * 1e-24


def atom_volume(mass_density, molar_mass):
    """Volume per formula unit in Å³."""
    N_A = 6.022e23
    return molar_mass / (mass_density * N_A) * 1e24


# ─── Form-factor helpers ──────────────────────────────────────────────────────

def sphere_f0_1d(q, R):
    """Normalised sphere form factor for scalar R → shape (Nq,)."""
    x = q * R
    return np.where(np.abs(x) < 1e-8, 1.0,
                    3.0 * (np.sin(x) - x * np.cos(x)) / x**3)


def sphere_f0_2d(q, R_arr):
    """Normalised sphere form factor → shape (Nq, NR)."""
    x = q[:, None] * R_arr[None, :]
    return np.where(np.abs(x) < 1e-8, 1.0,
                    3.0 * (np.sin(x) - x * np.cos(x)) / x**3)


def sphere_V(R):
    return (4.0 / 3.0) * np.pi * R**3


# ─── Partial structure factors ────────────────────────────────────────────────

def compute_partials(q, cfg):
    """Return I_MM, I_RM, I_RR in cm⁻¹."""
    core    = cfg["core"]
    shell   = cfg["shell"]
    solvent = cfg["solvent"]
    exp     = cfg["experiment"]

    rho_core    = electron_density(core["mass_density"],    core["molar_mass"],    core["n_electrons"])
    rho_shell   = electron_density(shell["mass_density"],   shell["molar_mass"],   shell["n_electrons"])
    rho_solvent = electron_density(solvent["mass_density"], solvent["molar_mass"], solvent["n_electrons"])
    V_atom_res  = atom_volume(core["mass_density"], core["molar_mass"])

    drho_core  = rho_core  - rho_solvent
    drho_shell = rho_shell - rho_solvent
    delta      = shell["radius_outer"] - core["radius_mean"]  # constant shell thickness

    sigma_rel = core.get("sigma_rel", 0.0)
    phi       = exp["volume_fraction"]

    if sigma_rel == 0.0:
        # ── Monodisperse ──────────────────────────────────────────────────────
        R_c   = core["radius_mean"]
        R_t   = shell["radius_outer"]
        Vc    = sphere_V(R_c)
        Vt    = sphere_V(R_t)
        f0c   = sphere_f0_1d(q, R_c)
        f0t   = sphere_f0_1d(q, R_t)
        F_non = (drho_core - drho_shell) * Vc * f0c + drho_shell * Vt * f0t
        F_res = Vc / V_atom_res * f0c
        I_MM_raw = F_non**2
        I_RM_raw = F_non * F_res
        I_RR_raw = F_res**2
        V_particle = Vt

    else:
        # ── Polydisperse lognormal ────────────────────────────────────────────
        sigma_ln = np.sqrt(np.log(1 + sigma_rel**2))
        mu_ln    = np.log(core["radius_mean"]) - sigma_ln**2 / 2
        sigma_R  = core["radius_mean"] * sigma_rel
        R_vals   = np.linspace(max(1.0, core["radius_mean"] - 6 * sigma_R),
                               core["radius_mean"] + 6 * sigma_R, 600)
        P_R      = (np.exp(-(np.log(R_vals) - mu_ln)**2 / (2 * sigma_ln**2))
                    / (R_vals * sigma_ln * np.sqrt(2 * np.pi)))
        P_R     /= np.trapezoid(P_R, R_vals)

        R_tot   = R_vals + delta
        Vc      = sphere_V(R_vals)
        Vt      = sphere_V(R_tot)
        f0c     = sphere_f0_2d(q, R_vals)   # (Nq, NR)
        f0t     = sphere_f0_2d(q, R_tot)    # (Nq, NR)
        F_non   = (drho_core - drho_shell) * Vc[None, :] * f0c + drho_shell * Vt[None, :] * f0t
        F_res   = Vc[None, :] / V_atom_res * f0c
        I_MM_raw = np.trapezoid(P_R[None, :] * F_non**2,      R_vals, axis=1)
        I_RM_raw = np.trapezoid(P_R[None, :] * F_non * F_res, R_vals, axis=1)
        I_RR_raw = np.trapezoid(P_R[None, :] * F_res**2,      R_vals, axis=1)
        V_particle = np.mean(sphere_V(R_vals + delta))

    # Convert to cm⁻¹
    r_e_cm = 2.818e-13
    N_p_cm = phi / V_particle * 1e24
    scale  = N_p_cm * r_e_cm**2
    return scale * I_MM_raw, scale * I_RM_raw, scale * I_RR_raw


# ─── Equidistant-f' energy selection ─────────────────────────────────────────

def select_energies(cfg):
    exp    = cfg["experiment"]
    Z      = exp["resonant_Z"]
    E_min  = exp["E_min_keV"]
    E_max  = exp["E_max_keV"]
    N      = exp["n_energies"]

    E_dense   = np.linspace(E_min, E_max, 3000)
    fp_dense  = np.array([xraydb.f1_chantler(Z, e * 1000) for e in E_dense])
    sort_idx  = np.argsort(fp_dense)
    fp_sorted = fp_dense[sort_idx]
    E_sorted  = E_dense[sort_idx]
    fp_tgt    = np.linspace(fp_sorted[0], fp_sorted[-1], N)
    energies  = sorted(np.interp(fp_tgt, fp_sorted, E_sorted))

    fp_vals  = np.array([xraydb.f1_chantler(Z, e * 1000) for e in energies])
    fpp_vals = np.array([abs(xraydb.f2_chantler(Z, e * 1000)) for e in energies])
    return energies, fp_vals, fpp_vals


# ─── Noise ───────────────────────────────────────────────────────────────────

def make_noise(I_tot, cfg, rng):
    nc = cfg["noise"]
    if nc["model"] == "poisson":
        # σ(q) = √(I(q) · I(q_min) / N_peak)
        # → at q_min: σ/I = 1/√N_peak; grows with falling signal
        I_ref = I_tot[0]
        N_pk  = nc["N_peak"]
        sigma = np.sqrt(np.maximum(I_tot * I_ref / N_pk, 1e-30))
    else:
        rel   = nc.get("relative", 0.005)
        sigma = rel * I_tot
    sigma = np.maximum(sigma, 1e-8)
    I_obs = np.maximum(I_tot + rng.normal(0, 1, len(I_tot)) * sigma, 1e-8)
    return I_obs, sigma


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} <config_file.py>")

    cfg     = load_config(sys.argv[1])
    out_cfg = cfg["output"]
    out_dir = Path(__file__).parent / out_cfg["directory"]
    name    = out_cfg.get("name", cfg.get("system", {}).get("name", "particle"))

    # ── q grid ────────────────────────────────────────────────────────────────
    qg  = cfg["q_grid"]
    if qg.get("spacing", "log") == "linear":
        q = np.linspace(qg["q_min"], qg["q_max"], qg["n_points"])
    else:
        q = np.geomspace(qg["q_min"], qg["q_max"], qg["n_points"])

    # ── Partial structure factors ─────────────────────────────────────────────
    print("Computing partial structure factors …")
    I_MM, I_RM, I_RR = compute_partials(q, cfg)

    # ── Energy selection ──────────────────────────────────────────────────────
    print("Selecting energies …")
    energies, fp_vals, fpp_vals = select_energies(cfg)

    print(f"\nEnergies (equidistant f' spacing, Z={cfg['experiment']['resonant_Z']}):")
    print(f"  {'E (keV)':>10}  {'fp (e)':>8}  {'fpp (e)':>8}")
    print(f"  {'-'*34}")
    for e, fp, fpp in zip(energies, fp_vals, fpp_vals):
        print(f"  {e:10.4f}  {fp:8.3f}  {fpp:8.3f}")

    # ── Noise summary ─────────────────────────────────────────────────────────
    nc = cfg["noise"]
    if nc["model"] == "poisson":
        I_ref = I_MM[0] + 2*fp_vals[0]*I_RM[0] + (fp_vals[0]**2+fpp_vals[0]**2)*I_RR[0]
        print(f"\nNoise model: Poisson,  N_peak = {nc['N_peak']:.0f} photons at q_min")
        print(f"  → σ/I at q_min = {1/np.sqrt(nc['N_peak'])*100:.2f}%")
        if I_ref > 0:
            I_last  = I_MM[-1] + 2*fp_vals[0]*I_RM[-1] + (fp_vals[0]**2+fpp_vals[0]**2)*I_RR[-1]
            sig_frac = np.sqrt(I_ref / (I_last * nc["N_peak"])) * 100
            print(f"  → σ/I at q_max ≈ {sig_frac:.1f}%")
    else:
        print(f"\nNoise model: relative {nc.get('relative',0.005)*100:.2f}%")

    # ── Save files ────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for E, fp, fpp in zip(energies, fp_vals, fpp_vals):
        I_tot        = I_MM + 2*fp * I_RM + (fp**2 + fpp**2) * I_RR
        I_obs, sigma = make_noise(I_tot, cfg, rng)

        fname  = out_dir / f"{name}_{E:.4f}keV.dat"
        hdr    = (
            f"{cfg.get('system', {}).get('description', name)}\n"
            f"  Core: Z={cfg['core']['Z']}, R_mean={cfg['core']['radius_mean']} A"
            f", sigma_rel={cfg['core'].get('sigma_rel',0)*100:.0f}%\n"
            f"  Shell: R_outer={cfg['shell']['radius_outer']} A\n"
            f"  phi={cfg['experiment']['volume_fraction']:.2e}"
            f",  noise={nc['model']}"
            + (f",  N_peak={nc['N_peak']}" if nc['model']=='poisson' else f",  rel={nc.get('relative',0):.3f}") + "\n"
            f"  Energy={E:.4f} keV,  f'(Z{cfg['experiment']['resonant_Z']}) = {fp:.4f} e"
            f",  f'' = {fpp:.4f} e\n"
            f"  Columns: q(1/A)  I(cm-1)  sigma(cm-1)"
        )
        np.savetxt(fname, np.column_stack([q, I_obs, sigma]),
                   header=hdr, fmt="%.6e")

    # ── Ground-truth partials ─────────────────────────────────────────────────
    np.savetxt(out_dir / "ground_truth_partials.dat",
               np.column_stack([q, I_MM, I_RM, I_RR]),
               header=(f"q(1/A)  I_MM(cm-1)  I_RM(cm-1)  I_RR(cm-1)\n"
                       f"  {cfg.get('system',{}).get('description', name)}"),
               fmt="%.6e")

    print(f"\nSaved {len(energies)} files + ground_truth_partials.dat")
    print(f"Output directory: {out_dir}/")

    q01 = 0.01
    print(f"\nReference partials at q = 0.01 Å⁻¹:")
    print(f"  I_MM = {np.interp(q01, q, I_MM):.4e} cm⁻¹")
    print(f"  I_RM = {np.interp(q01, q, I_RM):.4e} cm⁻¹")
    print(f"  I_RR = {np.interp(q01, q, I_RR):.4e} cm⁻¹")


if __name__ == "__main__":
    main()
