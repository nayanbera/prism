"""
Config: Au-core / SiO2-shell / water, lognormal polydisperse, Poissonian noise
Run:  python generate_asaxs_data.py config_au_sio2_poly.py
"""

CONFIG = {
    "system": {
        "name": "Au-SiO2-poly",
        "description": "Au core / SiO2 shell in water — lognormal polydisperse σ=20%",
    },

    # ── Core material ──────────────────────────────────────────────────────────
    # mass_density : g/cm³
    # molar_mass   : g/mol  (per formula unit, e.g. 1 atom for elements)
    # n_electrons  : electrons per formula unit
    # Z            : atomic number of resonant element (for f'/f'' from xraydb)
    # radius_mean  : Å  (mean core radius for lognormal distribution)
    # sigma_rel    : fractional lognormal width — 0.0 for monodisperse
    "core": {
        "Z":            79,       # Au
        "mass_density": 19.32,    # g/cm³
        "molar_mass":   196.97,   # g/mol
        "n_electrons":  79,
        "radius_mean":  55.0,     # Å
        "sigma_rel":    0.20,     # 20% lognormal polydispersity
    },

    # ── Shell material ─────────────────────────────────────────────────────────
    # radius_outer : total outer radius (Å) — shell thickness = radius_outer − core.radius_mean
    # Shell thickness is kept constant across the size distribution.
    "shell": {
        "mass_density":  2.196,   # g/cm³  (amorphous SiO2)
        "molar_mass":   60.08,    # g/mol  (Si + 2O)
        "n_electrons":  30,       # 14 (Si) + 2×8 (O)
        "radius_outer": 220.0,    # Å  → shell thickness = 165 Å
    },

    # ── Solvent ────────────────────────────────────────────────────────────────
    "solvent": {
        "mass_density": 1.000,    # g/cm³
        "molar_mass":  18.015,    # g/mol
        "n_electrons": 10,        # 2 (H×2) + 8 (O)
    },

    # ── Experiment ────────────────────────────────────────────────────────────
    # resonant_Z   : atomic number of the element scanned through its absorption edge
    # E_min / E_max: energy range in keV (should bracket the edge of interest)
    # n_energies   : number of energy points; chosen to be equidistant in f'
    "experiment": {
        "volume_fraction": 1e-4,
        "resonant_Z":  79,        # Au
        "E_min_keV":   10.919,    # keV  (Au L3 edge ≈ 11.919 keV)
        "E_max_keV":   11.919,
        "n_energies":  20,
    },

    # ── q grid ────────────────────────────────────────────────────────────────
    "q_grid": {
        "q_min":   0.003,   # Å⁻¹
        "q_max":   0.145,   # Å⁻¹
        "n_points": 500,
        "spacing": "log",   # "log" (geomspace) or "linear"
    },

    # ── Noise model ────────────────────────────────────────────────────────────
    # model = "poisson"  : σ(q) = √(I(q)·I(q_min)/N_peak)
    #   → relative noise at q_min = 1/√N_peak  (e.g. 10 000 → 1%)
    #   → grows as √(I_qmin/I(q)) at higher q (realistic photon statistics)
    # model = "relative" : σ(q) = relative · I(q)  (flat percentage, simpler)
    "noise": {
        "model":    "poisson",
        "N_peak":   10000,     # photons at q_min → 1% noise at the peak
        # "model":  "relative",
        # "relative": 0.005,
    },

    # ── Output ────────────────────────────────────────────────────────────────
    "output": {
        "directory": "au_sio2_poly",
        "name":      "Au_SiO2_poly",   # prefix for data file names
    },
}
