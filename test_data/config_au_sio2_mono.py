"""
Config: Au-core / SiO2-shell / water, monodisperse, Poissonian noise
Run:  python generate_asaxs_data.py config_au_sio2_mono.py
"""

CONFIG = {
    "system": {
        "name": "Au-SiO2-mono",
        "description": "Au core / SiO2 shell in water — monodisperse",
    },

    "core": {
        "Z":            79,       # Au
        "mass_density": 19.32,
        "molar_mass":   196.97,
        "n_electrons":  79,
        "radius_mean":  55.0,     # Å
        "sigma_rel":    0.0,      # monodisperse
    },

    "shell": {
        "mass_density":  2.196,
        "molar_mass":   60.08,
        "n_electrons":  30,
        "radius_outer": 220.0,    # Å
    },

    "solvent": {
        "mass_density": 1.000,
        "molar_mass":  18.015,
        "n_electrons": 10,
    },

    "experiment": {
        "volume_fraction": 1e-4,
        "resonant_Z":  79,
        "E_min_keV":   10.919,
        "E_max_keV":   11.919,
        "n_energies":  20,
    },

    "q_grid": {
        "q_min":   0.003,
        "q_max":   0.145,
        "n_points": 500,
        "spacing": "log",
    },

    "noise": {
        "model":  "poisson",
        "N_peak": 10000,
    },

    "output": {
        "directory": "au_sio2_mono",
        "name":      "Au_SiO2_mono",
    },
}
