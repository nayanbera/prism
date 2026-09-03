"""Synthetic ASAXS data generator tab — parameterizable core-shell model."""

import traceback
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QSpinBox, QComboBox, QPushButton, QDoubleSpinBox,
    QScrollArea, QSplitter, QTextEdit, QFileDialog, QLineEdit,
    QRadioButton, QButtonGroup, QCheckBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import pyqtgraph as pg

from .core.crosshair import add_crosshair

try:
    import xraydb
    _HAS_XRAYDB = True
except ImportError:
    _HAS_XRAYDB = False


# ─── Color helper (blue → red by energy) ─────────────────────────────────────

def _energy_color(t: float) -> tuple[int, int, int]:
    stops = [(0.0, (0, 0, 255)), (0.33, (0, 200, 255)),
             (0.5, (0, 220, 0)), (0.67, (255, 220, 0)), (1.0, (255, 0, 0))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]; t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(int(c0[k] + f * (c1[k] - c0[k])) for k in range(3))
    return (255, 0, 0)


# ─── Material presets ─────────────────────────────────────────────────────────

_ELEMENT_PRESETS = {
    'Au (gold)':      {'Z': 79, 'mass_density': 19.32,  'molar_mass': 196.97, 'n_electrons': 79},
    'Pt (platinum)':  {'Z': 78, 'mass_density': 21.45,  'molar_mass': 195.08, 'n_electrons': 78},
    'Pd (palladium)': {'Z': 46, 'mass_density': 12.02,  'molar_mass': 106.42, 'n_electrons': 46},
    'Ag (silver)':    {'Z': 47, 'mass_density': 10.49,  'molar_mass': 107.87, 'n_electrons': 47},
    'Fe (iron)':      {'Z': 26, 'mass_density':  7.87,  'molar_mass':  55.85, 'n_electrons': 26},
    'Cu (copper)':    {'Z': 29, 'mass_density':  8.96,  'molar_mass':  63.55, 'n_electrons': 29},
    'Ni (nickel)':    {'Z': 28, 'mass_density':  8.91,  'molar_mass':  58.69, 'n_electrons': 28},
    'Co (cobalt)':    {'Z': 27, 'mass_density':  8.90,  'molar_mass':  58.93, 'n_electrons': 27},
    'Se (selenium)':  {'Z': 34, 'mass_density':  4.81,  'molar_mass':  78.97, 'n_electrons': 34},
    'Br (bromine)':   {'Z': 35, 'mass_density':  3.12,  'molar_mass':  79.90, 'n_electrons': 35},
    'Zn (zinc)':      {'Z': 30, 'mass_density':  7.13,  'molar_mass':  65.38, 'n_electrons': 30},
}

_MATERIAL_PRESETS = {
    'SiO₂ (silica)':     {'mass_density': 2.196, 'molar_mass':  60.08, 'n_electrons': 30},
    'H₂O (water)':       {'mass_density': 1.000, 'molar_mass':  18.015,'n_electrons': 10},
    'D₂O (heavy water)': {'mass_density': 1.105, 'molar_mass':  20.03, 'n_electrons': 10},
    'Polystyrene':        {'mass_density': 1.050, 'molar_mass': 104.15, 'n_electrons': 56},
    'PMMA':               {'mass_density': 1.190, 'molar_mass': 100.12, 'n_electrons': 54},
    'Toluene':            {'mass_density': 0.867, 'molar_mass':  92.14, 'n_electrons': 50},
    'Ethanol':            {'mass_density': 0.789, 'molar_mass':  46.07, 'n_electrons': 26},
    'Air':                {'mass_density': 0.00129,'molar_mass': 28.97, 'n_electrons': 15},
}


# ─── Molecular formula parser ────────────────────────────────────────────────

def _parse_formula(formula: str) -> dict[str, int]:
    """Parse a molecular formula like 'Ca3(PO4)2' → {'Ca':3,'P':2,'O':8}."""
    import re

    def _helper(s: str) -> dict[str, int]:
        result = {}
        i = 0
        while i < len(s):
            if s[i] == '(':
                j = i + 1; depth = 1
                while j < len(s) and depth:
                    if s[j] == '(': depth += 1
                    elif s[j] == ')': depth -= 1
                    j += 1
                inner = _helper(s[i + 1:j - 1])
                m = re.match(r'\d+', s[j:])
                mul = int(m.group()) if m else 1
                j  += len(m.group()) if m else 0
                for sym, cnt in inner.items():
                    result[sym] = result.get(sym, 0) + cnt * mul
                i = j
            else:
                m = re.match(r'([A-Z][a-z]?)(\d*)', s[i:])
                if not m:
                    raise ValueError(f'Cannot parse formula near: {s[i:]!r}')
                sym, num = m.group(1), m.group(2)
                result[sym] = result.get(sym, 0) + (int(num) if num else 1)
                i += len(m.group())
        return result

    return _helper(formula.strip())


def _formula_to_Mne(formula: str) -> tuple[float, int]:
    """Return (molar_mass g/mol, total electrons) for a molecular formula string."""
    if not _HAS_XRAYDB:
        raise RuntimeError('xraydb not installed — run: pip install xraydb')
    counts = _parse_formula(formula)
    M, ne = 0.0, 0
    for sym, cnt in counts.items():
        try:
            ne += xraydb.atomic_number(sym) * cnt
            M  += xraydb.atomic_mass(sym) * cnt
        except Exception:
            raise ValueError(f'Unknown element symbol: {sym!r}')
    return M, ne


def _element_Z(symbol: str) -> int | None:
    if not _HAS_XRAYDB:
        return None
    try:
        return xraydb.atomic_number(symbol)
    except Exception:
        return None


# ─── Physics ──────────────────────────────────────────────────────────────────

def _elec_density(rho, M, ne):
    return (rho / M) * 6.022e23 * ne * 1e-24

def _atom_volume(rho, M):
    return M / (rho * 6.022e23) * 1e24

def _sphere_f0_1d(q, R):
    x = q * R
    return np.where(np.abs(x) < 1e-8, 1.0, 3*(np.sin(x) - x*np.cos(x)) / x**3)

def _sphere_f0_2d(q, R_arr):
    x = q[:, None] * R_arr[None, :]
    return np.where(np.abs(x) < 1e-8, 1.0, 3*(np.sin(x) - x*np.cos(x)) / x**3)

def _sphere_V(R):
    return (4/3) * np.pi * R**3

def _lognormal_grid(mean, sig_rel, n=300):
    """Grid + weights for a lognormal distribution with given mean and σ_rel."""
    sln = np.sqrt(np.log(1 + sig_rel**2))
    mu  = np.log(mean) - sln**2 / 2
    sR  = mean * sig_rel
    xv  = np.linspace(max(1.0, mean - 6*sR), mean + 6*sR, n)
    P   = np.exp(-(np.log(xv) - mu)**2 / (2*sln**2)) / (xv * sln * np.sqrt(2*np.pi))
    P  /= np.trapz(P, xv)
    return xv, P


def _compute_partials(q, cfg, log):
    core = cfg['core']; shell = cfg['shell']; solv = cfg['solvent']
    rho_c = _elec_density(core['mass_density'], core['molar_mass'], core['n_electrons'])
    rho_s = _elec_density(shell['mass_density'], shell['molar_mass'], shell['n_electrons'])
    rho_v = _elec_density(solv['mass_density'], solv['molar_mass'], solv['n_electrons'])
    V_at  = _atom_volume(core['mass_density'], core['molar_mass'])
    dr_c  = rho_c - rho_v;  dr_s = rho_s - rho_v
    sig_c = core.get('sigma_rel', 0.0)
    sig_s = shell.get('sigma_rel', 0.0)
    Rc0   = core['radius_mean']
    Rt0   = shell['radius_outer']
    delta = Rt0 - Rc0
    phi   = cfg['experiment']['volume_fraction']
    log(f'  ρ_core={rho_c:.4f}  ρ_shell={rho_s:.4f}  ρ_solv={rho_v:.4f} e/Å³')

    if sig_c == 0.0 and sig_s == 0.0:
        # ── Fully monodisperse ────────────────────────────────────────────────
        Vc = _sphere_V(Rc0); Vt = _sphere_V(Rt0)
        f0c = _sphere_f0_1d(q, Rc0); f0t = _sphere_f0_1d(q, Rt0)
        Fn = (dr_c - dr_s) * Vc * f0c + dr_s * Vt * f0t
        Fr = Vc / V_at * f0c
        I_MM_r, I_RM_r, I_RR_r = Fn**2, Fn*Fr, Fr**2
        Vp = Vt;  log('  Mode: monodisperse')

    elif sig_c > 0.0 and sig_s == 0.0:
        # ── Polydisperse core, fixed shell thickness ───────────────────────────
        Rv, PR = _lognormal_grid(Rc0, sig_c, n=600)
        Rtv = Rv + delta
        Vc = _sphere_V(Rv); Vt = _sphere_V(Rtv)
        f0c = _sphere_f0_2d(q, Rv); f0t = _sphere_f0_2d(q, Rtv)
        Fn  = (dr_c - dr_s) * Vc[None,:] * f0c + dr_s * Vt[None,:] * f0t
        Fr  = Vc[None,:] / V_at * f0c
        I_MM_r = np.trapz(PR[None,:] * Fn**2,   Rv, axis=1)
        I_RM_r = np.trapz(PR[None,:] * Fn * Fr, Rv, axis=1)
        I_RR_r = np.trapz(PR[None,:] * Fr**2,   Rv, axis=1)
        Vp = np.mean(_sphere_V(Rv + delta))
        log(f'  Mode: core lognormal σ_rel={sig_c:.2f}')

    elif sig_c == 0.0 and sig_s > 0.0:
        # ── Monodisperse core, polydisperse R_outer ───────────────────────────
        # Fr doesn't depend on Rt → I_RR = Fr², I_RM = Fr*<Fn>, I_MM via moments
        Vc  = _sphere_V(Rc0)
        f0c = _sphere_f0_1d(q, Rc0)
        Fr  = Vc / V_at * f0c                              # (NQ,) — constant
        Fn0 = (dr_c - dr_s) * Vc * f0c                    # (NQ,) core contribution

        Rtv, Ptv = _lognormal_grid(Rt0, sig_s, n=300)
        # Ensure Rt > Rc (clip distribution below Rc+1)
        mask = Rtv > Rc0 + 1.0
        Rtv, Ptv = Rtv[mask], Ptv[mask]
        Ptv /= np.trapz(Ptv, Rtv)

        Vtv    = _sphere_V(Rtv)                            # (Nt,)
        f0t    = _sphere_f0_2d(q, Rtv)                    # (NQ, Nt)
        st     = dr_s * Vtv[None,:] * f0t                 # (NQ, Nt) shell amplitude
        mean_st    = np.trapz(Ptv[None,:] * st,      Rtv, axis=1)  # (NQ,)
        mean_st_sq = np.trapz(Ptv[None,:] * st**2,   Rtv, axis=1)  # (NQ,)

        I_MM_r = Fn0**2 + 2*Fn0*mean_st + mean_st_sq
        I_RM_r = Fr * (Fn0 + mean_st)
        I_RR_r = Fr**2
        Vp = np.trapz(Ptv * _sphere_V(Rtv), Rtv)
        log(f'  Mode: R_outer lognormal σ_rel={sig_s:.2f}')

    else:
        # ── Both polydisperse — 2-D integral (60 × 60 grid) ──────────────────
        Rv, PRv = _lognormal_grid(Rc0, sig_c, n=60)
        Rtv, Ptv = _lognormal_grid(Rt0, sig_s, n=60)
        # Mask Rt < Rc (unphysical)
        valid = Rtv > Rv.min() + 1.0
        Rtv, Ptv = Rtv[valid], Ptv[valid]
        Ptv /= np.trapz(Ptv, Rtv)

        Vc  = _sphere_V(Rv)                                # (Nv,)
        Vt  = _sphere_V(Rtv)                               # (Nt,)
        f0c = _sphere_f0_2d(q, Rv)                        # (NQ, Nv)
        f0t = _sphere_f0_2d(q, Rtv)                       # (NQ, Nt)

        # Fn[q,i,j] = core_amp[q,i] + shell_amp[q,j]
        core_amp  = (dr_c - dr_s) * Vc[None,:] * f0c     # (NQ, Nv)
        shell_amp = dr_s * Vt[None,:] * f0t               # (NQ, Nt)
        Fn = core_amp[:,:,None] + shell_amp[:,None,:]      # (NQ, Nv, Nt)
        Fr = (Vc / V_at)[None,:] * f0c                    # (NQ, Nv)  — no Rt dep.

        W = PRv[None,:,None] * Ptv[None,None,:]            # (1, Nv, Nt)
        I_MM_r = np.trapz(np.trapz(W * Fn**2,         Rtv, axis=2), Rv, axis=1)
        # I_RM = <Fn * Fr>: Fr has no Nt dep → integrate Fn over Rt first
        mean_Fn = np.trapz(Ptv[None,None,:] * Fn,     Rtv, axis=2)  # (NQ, Nv)
        I_RM_r  = np.trapz(PRv[None,:] * mean_Fn * Fr, Rv, axis=1)
        I_RR_r  = np.trapz(PRv[None,:] * Fr**2,        Rv, axis=1)
        Vp = np.trapz(Ptv * _sphere_V(Rtv), Rtv)
        log(f'  Mode: core σ_rel={sig_c:.2f}  R_outer σ_rel={sig_s:.2f}  (2-D grid)')

    sc = phi / Vp * 1e24 * (2.818e-13)**2
    return sc * I_MM_r, sc * I_RM_r, sc * I_RR_r


def _compute_partials_multi(q, cfg, elem_results, log):
    """Compute all ASAXS partials for N elements with per-element geometry.

    Each elem in elem_results must have 'location': 'core' or 'shell'.
    Returns (I_MM, [I_R1M,...], [I_R1R1,...], {(i,j): I_RiRj,...})  all in cm⁻¹.
    """
    core = cfg['core']; shell = cfg['shell']; solv = cfg['solvent']
    rho_c = _elec_density(core['mass_density'], core['molar_mass'], core['n_electrons'])
    rho_s = _elec_density(shell['mass_density'], shell['molar_mass'], shell['n_electrons'])
    rho_v = _elec_density(solv['mass_density'], solv['molar_mass'], solv['n_electrons'])
    dr_c = rho_c - rho_v;  dr_s = rho_s - rho_v
    sig_c = core.get('sigma_rel', 0.0);  sig_s = shell.get('sigma_rel', 0.0)
    Rc0 = core['radius_mean'];  Rt0 = shell['radius_outer'];  delta = Rt0 - Rc0
    phi = cfg['experiment']['volume_fraction']
    n_elem = len(elem_results)
    log(f'  ρ_core={rho_c:.4f}  ρ_shell={rho_s:.4f}  ρ_solv={rho_v:.4f} e/Å³')

    # Volume per atom for each element: core element uses core material,
    # shell element uses shell material (sets the resonant atom number density)
    V_at = []
    for elem in elem_results:
        if elem.get('location', 'core') == 'core':
            V_at.append(_atom_volume(core['mass_density'], core['molar_mass']))
        else:
            V_at.append(_atom_volume(shell['mass_density'], shell['molar_mass']))

    if sig_c == 0.0 and sig_s == 0.0:
        # ── Monodisperse ──────────────────────────────────────────────────────
        Vc = _sphere_V(Rc0);  Vt = _sphere_V(Rt0)
        f0c = _sphere_f0_1d(q, Rc0);  f0t = _sphere_f0_1d(q, Rt0)
        Fn = (dr_c - dr_s) * Vc * f0c + dr_s * Vt * f0t
        Fr = []
        for i, elem in enumerate(elem_results):
            if elem.get('location', 'core') == 'core':
                Fr.append(Vc / V_at[i] * f0c)
            else:
                Fr.append((Vt * f0t - Vc * f0c) / V_at[i])
        I_MM   = Fn**2
        I_RiM  = [Fn * Fr[i] for i in range(n_elem)]
        I_RiRi = [Fr[i]**2 for i in range(n_elem)]
        I_RiRj = {(i, j): Fr[i] * Fr[j]
                  for i in range(n_elem) for j in range(i + 1, n_elem)}
        Vp = Vt
        log('  Mode: monodisperse')

    elif sig_c > 0.0 and sig_s == 0.0:
        # ── Polydisperse core, fixed shell thickness ───────────────────────────
        Rv, PR = _lognormal_grid(Rc0, sig_c, n=600)
        Rtv = Rv + delta
        Vc = _sphere_V(Rv);  Vt = _sphere_V(Rtv)
        f0c = _sphere_f0_2d(q, Rv);  f0t = _sphere_f0_2d(q, Rtv)
        Fn = (dr_c - dr_s) * Vc[None, :] * f0c + dr_s * Vt[None, :] * f0t  # (NQ, Nv)
        Fr = []
        for i, elem in enumerate(elem_results):
            if elem.get('location', 'core') == 'core':
                Fr.append(Vc[None, :] / V_at[i] * f0c)                        # (NQ, Nv)
            else:
                Fr.append((Vt[None, :] * f0t - Vc[None, :] * f0c) / V_at[i]) # (NQ, Nv)
        I_MM   = np.trapz(PR[None, :] * Fn**2,                  Rv, axis=1)
        I_RiM  = [np.trapz(PR[None, :] * Fn * Fr[i],            Rv, axis=1) for i in range(n_elem)]
        I_RiRi = [np.trapz(PR[None, :] * Fr[i]**2,              Rv, axis=1) for i in range(n_elem)]
        I_RiRj = {(i, j): np.trapz(PR[None, :] * Fr[i] * Fr[j], Rv, axis=1)
                  for i in range(n_elem) for j in range(i + 1, n_elem)}
        Vp = np.mean(_sphere_V(Rv + delta))
        log(f'  Mode: core lognormal σ_rel={sig_c:.2f}')

    elif sig_c == 0.0 and sig_s > 0.0:
        # ── Monodisperse core, polydisperse R_outer ───────────────────────────
        Vc = _sphere_V(Rc0);  f0c = _sphere_f0_1d(q, Rc0)
        Fn_core = (dr_c - dr_s) * Vc * f0c                    # (NQ,)
        Rtv, Ptv = _lognormal_grid(Rt0, sig_s, n=300)
        mask = Rtv > Rc0 + 1.0
        Rtv, Ptv = Rtv[mask], Ptv[mask]
        Ptv /= np.trapz(Ptv, Rtv)
        Vtv = _sphere_V(Rtv);  f0t = _sphere_f0_2d(q, Rtv)   # (NQ, Nt)
        shell_amp = dr_s * Vtv[None, :] * f0t                  # (NQ, Nt)
        mean_sh  = np.trapz(Ptv[None, :] * shell_amp,    Rtv, axis=1)  # (NQ,)
        mean_sh2 = np.trapz(Ptv[None, :] * shell_amp**2, Rtv, axis=1)  # (NQ,)
        I_MM = Fn_core**2 + 2 * Fn_core * mean_sh + mean_sh2

        # Per-element resonant amplitudes: 'core' → 1D, 'shell' → 2D (NQ, Nt)
        Fr = []   # list of (kind, array)
        for i, elem in enumerate(elem_results):
            if elem.get('location', 'core') == 'core':
                Fr.append(('core', Vc / V_at[i] * f0c))                             # (NQ,)
            else:
                Fr.append(('shell', (Vtv[None, :] * f0t - Vc * f0c[:, None]) / V_at[i]))  # (NQ, Nt)

        def _avg1(fi_pair):
            k, a = fi_pair
            return a if k == 'core' else np.trapz(Ptv[None, :] * a, Rtv, axis=1)

        def _avg_prod(fi_pair, fj_pair):
            ki, ai = fi_pair;  kj, aj = fj_pair
            if ki == 'core' and kj == 'core':
                return ai * aj
            if ki == 'core':   # aj is 2D
                return ai * _avg1(fj_pair)
            if kj == 'core':   # ai is 2D
                return _avg1(fi_pair) * aj
            return np.trapz(Ptv[None, :] * ai * aj, Rtv, axis=1)

        I_RiM = []
        for i in range(n_elem):
            ki, fi = Fr[i]
            if ki == 'core':
                I_RiM.append(fi * (Fn_core + mean_sh))
            else:
                mean_fi       = np.trapz(Ptv[None, :] * fi,             Rtv, axis=1)
                mean_sh_fi    = np.trapz(Ptv[None, :] * shell_amp * fi, Rtv, axis=1)
                I_RiM.append(Fn_core * mean_fi + mean_sh_fi)

        I_RiRi = []
        for i in range(n_elem):
            ki, fi = Fr[i]
            if ki == 'core':
                I_RiRi.append(fi**2)
            else:
                I_RiRi.append(np.trapz(Ptv[None, :] * fi**2, Rtv, axis=1))

        I_RiRj = {(i, j): _avg_prod(Fr[i], Fr[j])
                  for i in range(n_elem) for j in range(i + 1, n_elem)}
        Vp = np.trapz(Ptv * _sphere_V(Rtv), Rtv)
        log(f'  Mode: R_outer lognormal σ_rel={sig_s:.2f}')

    else:
        # ── Both polydisperse — 2-D integral ──────────────────────────────────
        Rv, PRv = _lognormal_grid(Rc0, sig_c, n=60)
        Rtv, Ptv = _lognormal_grid(Rt0, sig_s, n=60)
        valid = Rtv > Rv.min() + 1.0
        Rtv, Ptv = Rtv[valid], Ptv[valid]
        Ptv /= np.trapz(Ptv, Rtv)
        Vc = _sphere_V(Rv);  Vt = _sphere_V(Rtv)
        f0c = _sphere_f0_2d(q, Rv);  f0t = _sphere_f0_2d(q, Rtv)
        core_amp  = (dr_c - dr_s) * Vc[None, :] * f0c    # (NQ, Nv)
        shell_amp = dr_s * Vt[None, :] * f0t              # (NQ, Nt)
        Fn = core_amp[:, :, None] + shell_amp[:, None, :] # (NQ, Nv, Nt)
        W  = PRv[None, :, None] * Ptv[None, None, :]      # (1, Nv, Nt)

        def _intW(arr3d):
            return np.trapz(np.trapz(W * arr3d, Rtv, axis=2), Rv, axis=1)

        I_MM = _intW(Fn**2)

        # Per-element resonant amplitudes as (NQ, Nv, Nt) via broadcasting
        Fr3d = []
        for i, elem in enumerate(elem_results):
            if elem.get('location', 'core') == 'core':
                # (NQ, Nv) → broadcast to (NQ, Nv, Nt)
                fr = Vc[None, :] / V_at[i] * f0c          # (NQ, Nv)
                Fr3d.append(fr[:, :, None])                # broadcast over Nt
            else:
                # (Vt[Rt]*f0t[q,Rt] - Vc[Rv]*f0c[q,Rv]) — true (NQ, Nv, Nt)
                Fr3d.append(
                    (Vt[None, None, :] * f0t[:, None, :] - Vc[None, :, None] * f0c[:, :, None]) / V_at[i]
                )

        I_MM   = _intW(Fn**2)
        I_RiM  = [_intW(Fn * Fr3d[i]) for i in range(n_elem)]
        I_RiRi = [_intW(Fr3d[i]**2)   for i in range(n_elem)]
        I_RiRj = {(i, j): _intW(Fr3d[i] * Fr3d[j])
                  for i in range(n_elem) for j in range(i + 1, n_elem)}
        Vp = np.trapz(Ptv * _sphere_V(Rtv), Rtv)
        log(f'  Mode: core σ_rel={sig_c:.2f}  R_outer σ_rel={sig_s:.2f}  (2-D grid)')

    sc = phi / Vp * 1e24 * (2.818e-13)**2
    return (sc * I_MM,
            [sc * x for x in I_RiM],
            [sc * x for x in I_RiRi],
            {k: sc * x for k, x in I_RiRj.items()})


def _select_energies(cfg, log=None):
    """Single-element wrapper kept for backward compat."""
    exp = cfg['experiment']
    Z = exp['resonant_Z']; E0 = exp['E_min_keV']; E1 = exp['E_max_keV']; N = exp['n_energies']
    Ed   = np.linspace(E0, E1, 3000)
    fpd  = np.array([xraydb.f1_chantler(Z, e*1000) for e in Ed])
    idx  = np.argsort(fpd)
    fp_s = fpd[idx]; E_s = Ed[idx]
    tgt  = np.linspace(fp_s[0], fp_s[-1], N)
    Es   = sorted(np.interp(tgt, fp_s, E_s))
    fp   = np.array([xraydb.f1_chantler(Z, e*1000) for e in Es])
    fpp  = np.array([abs(xraydb.f2_chantler(Z, e*1000)) for e in Es])
    if log:
        log(f'  {"E (keV)":>10}  {"f′ (e)":>8}  {"f″ (e)":>8}')
        log(f'  {"-"*34}')
        for e, f, ff in zip(Es, fp, fpp):
            log(f'  {e:10.4f}  {f:8.3f}  {ff:8.3f}')
    return Es, fp, fpp


def _select_energies_multi(cfg, log=None):
    """Select energies and compute f'/f'' for all active resonant elements.

    Each element has its own E_min/E_max/N range (equidistant in Δf').
    The per-element grids are combined into a sorted union, and f'/f'' for
    every element is evaluated on the full combined grid.

    Returns (energies_array, elem_results) where elem_results is a list of
    {'Z', 'fp', 'fpp', 'scale'} dicts.
    """
    elements = cfg.get('resonant_elements', [])
    if not elements:
        exp = cfg['experiment']
        elements = [{'Z': exp.get('resonant_Z', 79), 'scale': 1.0,
                     'E_min_keV': exp['E_min_keV'], 'E_max_keV': exp['E_max_keV'],
                     'n_energies': exp['n_energies']}]

    # Build per-element equidistant-Δf' grids and combine
    all_Es = []
    for elem in elements:
        Z = elem['Z']
        E0 = elem.get('E_min_keV', 10.0)
        E1 = elem.get('E_max_keV', 11.0)
        N  = int(elem.get('n_energies', 20))
        Ed  = np.linspace(E0, E1, 3000)
        fpd = np.array([xraydb.f1_chantler(Z, e*1000) for e in Ed])
        idx = np.argsort(fpd)
        tgt = np.linspace(fpd[idx[0]], fpd[idx[-1]], N)
        Es_i = sorted(np.interp(tgt, fpd[idx], Ed[idx]))
        all_Es.extend(Es_i)

    # Sorted union with duplicates removed (within 1 eV)
    all_Es = sorted(set(round(e, 6) for e in all_Es))
    Es = np.array(all_Es)

    if log:
        hdrs = ['E (keV)'] + [f"f'_{i+1}" for i in range(len(elements))] + \
               [f'f"_{i+1}' for i in range(len(elements))]
        log('  ' + '  '.join(f'{h:>9}' for h in hdrs))
        log('  ' + '-' * (11 * len(hdrs)))

    elem_results = []
    fp_table  = []
    fpp_table = []
    for elem in elements:
        Z   = elem['Z']
        fp  = np.array([xraydb.f1_chantler(Z, e*1000) for e in Es])
        fpp = np.array([abs(xraydb.f2_chantler(Z, e*1000)) for e in Es])
        elem_results.append({'Z': Z, 'fp': fp, 'fpp': fpp, 'location': elem.get('location', 'core')})
        fp_table.append(fp); fpp_table.append(fpp)

    if log:
        for k, e in enumerate(Es):
            row = f'  {e:10.4f}' + ''.join(f'  {fp_table[i][k]:8.3f}' for i in range(len(elements))) \
                                 + ''.join(f'  {fpp_table[i][k]:8.3f}' for i in range(len(elements)))
            log(row)

    return Es, elem_results

def _make_noise(I_tot, cfg, rng):
    nc = cfg['noise']
    if nc['model'] == 'poisson':
        sigma = np.sqrt(np.maximum(I_tot * I_tot[0] / nc['N_peak'], 1e-30))
    else:
        sigma = nc.get('relative', 0.005) * I_tot
    sigma = np.maximum(sigma, 1e-8)
    return np.maximum(I_tot + rng.normal(0, 1, len(I_tot)) * sigma, 1e-8), sigma


# ─── Generator thread ─────────────────────────────────────────────────────────

class _GeneratorThread(QThread):
    log_line     = pyqtSignal(str)
    data_ready   = pyqtSignal(object, object, object)  # q, energies, list[I_obs]
    finished_ok  = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent); self._cfg = cfg

    def run(self):
        try:
            cfg     = self._cfg
            out_dir = Path(cfg['output']['directory'])
            name    = cfg['output'].get('name', 'particle')
            qg      = cfg['q_grid']
            q       = (np.geomspace if qg.get('spacing','log') == 'log' else np.linspace)(
                qg['q_min'], qg['q_max'], qg['n_points'])

            self.log_line.emit('\nSelecting energies (per-element equidistant Δf′ grids) …')
            energies, elem_results = _select_energies_multi(cfg, self.log_line.emit)
            n_elem = len(elem_results)

            self.log_line.emit('\nComputing partial structure factors (per-element geometry) …')
            I_MM, I_RiM, I_RiRi, I_RiRj = _compute_partials_multi(
                q, cfg, elem_results, self.log_line.emit)
            self.log_line.emit(
                f'  I_MM={I_MM[0]:.3e}' +
                ''.join(f'  I_R{i+1}M={I_RiM[i][0]:.3e}' for i in range(n_elem)) +
                ''.join(f'  I_R{i+1}R{i+1}={I_RiRi[i][0]:.3e}' for i in range(n_elem)) +
                '  cm⁻¹ (at q_min)')

            nc = cfg['noise']
            if nc['model'] == 'poisson':
                self.log_line.emit(
                    f'\nNoise: Poisson, N_peak={nc["N_peak"]:.0f}  '
                    f'→ σ/I(q_min) = {1/np.sqrt(nc["N_peak"])*100:.2f}%')
            else:
                self.log_line.emit(f'\nNoise: relative {nc.get("relative",0.005)*100:.2f}%')

            out_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(42)
            self.log_line.emit(f'\nWriting {len(energies)} files to {out_dir}/ …')

            I_obs_list = []
            for k, E in enumerate(energies):
                I_tot = I_MM.copy()
                # Self terms
                for i, er in enumerate(elem_results):
                    fp_i  = er['fp'][k]
                    fpp_i = er['fpp'][k]
                    I_tot += 2*fp_i * I_RiM[i] + (fp_i**2 + fpp_i**2) * I_RiRi[i]
                # Cross terms
                for i in range(n_elem):
                    for j in range(i+1, n_elem):
                        fp_i, fpp_i = elem_results[i]['fp'][k], elem_results[i]['fpp'][k]
                        fp_j, fpp_j = elem_results[j]['fp'][k], elem_results[j]['fpp'][k]
                        I_tot += 2*(fp_i*fp_j + fpp_i*fpp_j) * I_RiRj[(i,j)]

                I_obs, sigma = _make_noise(I_tot, cfg, rng)
                I_obs_list.append(I_obs)

                core = cfg['core']
                fp_strs = '  '.join(f"f'{i+1}={elem_results[i]['fp'][k]:.3f}" for i in range(n_elem))
                hdr = (
                    f"Synthetic ASAXS — {n_elem} resonant element(s)\n"
                    f"  Core Z={core['Z']}, R_mean={core['radius_mean']:.1f} A"
                    f", sig_rel={core.get('sigma_rel',0)*100:.0f}%\n"
                    f"  Shell R_outer={cfg['shell']['radius_outer']:.1f} A"
                    f",  phi={cfg['experiment']['volume_fraction']:.2e}"
                    f",  noise={nc['model']}\n"
                    f"  Energy={E:.4f} keV  {fp_strs}\n"
                    f"  Columns: q(1/A)  I(cm-1)  sigma(cm-1)"
                )
                np.savetxt(out_dir / f"{name}_{E:.4f}keV.dat",
                           np.column_stack([q, I_obs, sigma]), header=hdr, fmt="%.6e")
                fp1 = elem_results[0]['fp'][k]; fpp1 = elem_results[0]['fpp'][k]
                self.log_line.emit(
                    f'  [{k+1:2d}/{len(energies)}]  {E:.4f} keV  '
                    + '  '.join(f"f'_{i+1}={elem_results[i]['fp'][k]:.3f}"
                                f" f\"_{i+1}={elem_results[i]['fpp'][k]:.3f}"
                                for i in range(n_elem)))

            # Ground-truth file — all partials
            from .core.decompose import partial_names_multi
            names_gt = partial_names_multi(n_elem)
            cols_gt  = [q, I_MM]
            for i in range(n_elem):
                cols_gt.append(I_RiM[i])
            for i in range(n_elem):
                cols_gt.append(I_RiRi[i])
            for i in range(n_elem):
                for j in range(i+1, n_elem):
                    cols_gt.append(I_RiRj[(i,j)])
            hdr_gt = 'q(1/A)  ' + '  '.join(f'{nm}(cm-1)' for nm in names_gt)
            locs_str = '  '.join(f'loc_{i+1}={elem_results[i]["location"]}' for i in range(n_elem))
            hdr_gt = f'Resonant elements: {[e["Z"] for e in elem_results]}  {locs_str}\n' + hdr_gt
            np.savetxt(out_dir / 'ground_truth_partials.dat',
                       np.column_stack(cols_gt), header=hdr_gt, fmt='%.6e')
            self.log_line.emit('\nground_truth_partials.dat saved.')
            self.log_line.emit(f'\nDone — {len(energies)} datasets in {out_dir}/')

            self.data_ready.emit(q, energies, I_obs_list)
            self.finished_ok.emit(str(out_dir))
        except Exception:
            self.error_signal.emit(traceback.format_exc())


# ─── Spin box helpers ─────────────────────────────────────────────────────────

def _dbl(val, lo=0.0, hi=1e9, decimals=4, step=0.01):
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi); sb.setDecimals(decimals); sb.setSingleStep(step); sb.setValue(val)
    return sb

def _int(val, lo=1, hi=99999):
    sb = QSpinBox(); sb.setRange(lo, hi); sb.setValue(val); return sb


# ─── Tab ──────────────────────────────────────────────────────────────────────

class SimDataTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread   = None
        self._curves   = []
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        main_split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(main_split)

        # ══ LEFT: scrollable parameter form ══════════════════════════════════
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form_w = QWidget(); form_lay = QVBoxLayout(form_w); form_lay.setSpacing(8)
        scroll.setWidget(form_w)
        main_split.addWidget(scroll)

        # ── Core material ─────────────────────────────────────────────────────
        core_grp  = QGroupBox('Core material')
        core_form = QFormLayout(core_grp)
        self._core_preset = QComboBox()
        self._core_preset.addItem('— custom —')
        for nm in _ELEMENT_PRESETS: self._core_preset.addItem(nm)
        btn_core = QPushButton('Apply'); btn_core.setFixedWidth(60)
        r = QHBoxLayout(); r.addWidget(self._core_preset); r.addWidget(btn_core)
        core_form.addRow('Element preset:', r)

        self._core_formula = QLineEdit()
        self._core_formula.setPlaceholderText('e.g.  Au')
        btn_core_fml = QPushButton('← fill M, ne')
        btn_core_fml.setFixedWidth(90)
        self._lbl_core_sld = QLabel()
        self._lbl_core_sld.setStyleSheet('color: #555;')
        fml_row_c = QHBoxLayout()
        fml_row_c.addWidget(self._core_formula); fml_row_c.addWidget(btn_core_fml)
        core_form.addRow('Formula:', fml_row_c)
        core_form.addRow('', self._lbl_core_sld)
        btn_core_fml.clicked.connect(self._fill_core_from_formula)

        self._core_Z   = _int(79, lo=1, hi=118)
        self._core_rho = _dbl(19.32, decimals=4)
        self._core_M   = _dbl(196.97, decimals=4)
        self._core_ne  = _int(79)
        self._core_R   = _dbl(55.0, lo=0.1, hi=1e5, decimals=2, step=1.0)
        self._core_sig = _dbl(0.20, lo=0.0, hi=2.0, decimals=3, step=0.01)
        self._lbl_core_sig = QLabel('= 20%')
        core_form.addRow('Atomic number Z:',  self._core_Z)
        core_form.addRow('ρ (g/cm³):',        self._core_rho)
        core_form.addRow('M (g/mol):',         self._core_M)
        core_form.addRow('e⁻/atom:',           self._core_ne)
        core_form.addRow('R_mean (Å):',        self._core_R)
        sig_row = QHBoxLayout()
        sig_row.addWidget(self._core_sig); sig_row.addWidget(self._lbl_core_sig)
        core_form.addRow('σ_rel (lognormal):', sig_row)
        core_form.addRow('', QLabel('<i>Set σ_rel = 0 for monodisperse</i>'))
        form_lay.addWidget(core_grp)
        btn_core.clicked.connect(self._apply_core_preset)
        self._core_sig.valueChanged.connect(
            lambda v: self._lbl_core_sig.setText('(monodisperse)' if v == 0 else f'= {v*100:.0f}%'))

        # ── Shell material ────────────────────────────────────────────────────
        shell_grp  = QGroupBox('Shell material')
        shell_form = QFormLayout(shell_grp)
        self._shell_preset = QComboBox()
        self._shell_preset.addItem('— custom —')
        for nm in _MATERIAL_PRESETS: self._shell_preset.addItem(nm)
        btn_shell = QPushButton('Apply'); btn_shell.setFixedWidth(60)
        r2 = QHBoxLayout(); r2.addWidget(self._shell_preset); r2.addWidget(btn_shell)
        shell_form.addRow('Material preset:', r2)

        self._shell_formula = QLineEdit()
        self._shell_formula.setPlaceholderText('e.g.  SiO2')
        btn_shell_fml = QPushButton('← fill M, ne')
        btn_shell_fml.setFixedWidth(90)
        self._lbl_shell_sld = QLabel()
        self._lbl_shell_sld.setStyleSheet('color: #555;')
        fml_row_s = QHBoxLayout()
        fml_row_s.addWidget(self._shell_formula); fml_row_s.addWidget(btn_shell_fml)
        shell_form.addRow('Formula:', fml_row_s)
        shell_form.addRow('', self._lbl_shell_sld)
        btn_shell_fml.clicked.connect(self._fill_shell_from_formula)

        self._shell_rho = _dbl(2.196, decimals=4)
        self._shell_M   = _dbl(60.08, decimals=4)
        self._shell_ne  = _int(30)
        self._shell_Rt  = _dbl(220.0, lo=0.1, hi=1e5, decimals=2, step=1.0)
        self._shell_sig = _dbl(0.0, lo=0.0, hi=2.0, decimals=3, step=0.01)
        self._lbl_shell_sig = QLabel('(monodisperse)')
        self._lbl_thick = QLabel()
        shell_form.addRow('ρ (g/cm³):',    self._shell_rho)
        shell_form.addRow('M (g/mol):',     self._shell_M)
        shell_form.addRow('e⁻/formula:',    self._shell_ne)
        shell_form.addRow('R_outer (Å):',   self._shell_Rt)
        shell_sig_row = QHBoxLayout()
        shell_sig_row.addWidget(self._shell_sig); shell_sig_row.addWidget(self._lbl_shell_sig)
        shell_form.addRow('σ_rel (R_outer):', shell_sig_row)
        shell_form.addRow('Shell thickness:', self._lbl_thick)
        form_lay.addWidget(shell_grp)
        btn_shell.clicked.connect(self._apply_shell_preset)
        self._core_R.valueChanged.connect(self._update_thickness)
        self._shell_Rt.valueChanged.connect(self._update_thickness)
        self._shell_sig.valueChanged.connect(self._update_thickness)
        self._shell_sig.valueChanged.connect(
            lambda v: self._lbl_shell_sig.setText('(monodisperse)' if v == 0 else f'= {v*100:.0f}%'))
        self._update_thickness()

        # ── Solvent ───────────────────────────────────────────────────────────
        solv_grp  = QGroupBox('Solvent')
        solv_form = QFormLayout(solv_grp)
        self._solv_preset = QComboBox()
        self._solv_preset.addItem('— custom —')
        for nm in _MATERIAL_PRESETS: self._solv_preset.addItem(nm)
        btn_solv = QPushButton('Apply'); btn_solv.setFixedWidth(60)
        r3 = QHBoxLayout(); r3.addWidget(self._solv_preset); r3.addWidget(btn_solv)
        solv_form.addRow('Material preset:', r3)

        self._solv_formula = QLineEdit()
        self._solv_formula.setPlaceholderText('e.g.  H2O')
        btn_solv_fml = QPushButton('← fill M, ne')
        btn_solv_fml.setFixedWidth(90)
        self._lbl_solv_sld = QLabel()
        self._lbl_solv_sld.setStyleSheet('color: #555;')
        fml_row_v = QHBoxLayout()
        fml_row_v.addWidget(self._solv_formula); fml_row_v.addWidget(btn_solv_fml)
        solv_form.addRow('Formula:', fml_row_v)
        solv_form.addRow('', self._lbl_solv_sld)
        btn_solv_fml.clicked.connect(self._fill_solv_from_formula)

        self._solv_rho = _dbl(1.000, decimals=4)
        self._solv_M   = _dbl(18.015, decimals=4)
        self._solv_ne  = _int(10)
        solv_form.addRow('ρ (g/cm³):', self._solv_rho)
        solv_form.addRow('M (g/mol):',  self._solv_M)
        solv_form.addRow('e⁻/formula:', self._solv_ne)
        form_lay.addWidget(solv_grp)
        btn_solv.clicked.connect(self._apply_solv_preset)

        # ── Energies ──────────────────────────────────────────────────────────
        en_grp = QGroupBox('Energies — per resonant element')
        en_lay = QVBoxLayout(en_grp)

        # Build element symbol list from xraydb (or static fallback)
        _sym_list = []
        if _HAS_XRAYDB:
            for _Z in range(3, 93):
                try:
                    _sym_list.append((xraydb.atomic_symbol(_Z), _Z))
                except Exception:
                    pass
        if not _sym_list:
            _FALLBACK = ['Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S',
                         'Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
                         'Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo',
                         'Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba',
                         'La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
                         'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi',
                         'Th','U']
            _sym_list = [(_s, _z) for _z, _s in enumerate(_FALLBACK, start=3)]

        _SHELLS_ALL = ['K','L1','L2','L3','M1','M2','M3','M4','M5','N1','N2','N3']
        _DEFAULT_ELEMS = [('Au', 79), ('Pt', 78), ('Se', 34)]
        _DEFAULT_EDGES = ['L3', 'L3', 'K']

        self._sim_elem_Z:          list = [0, 0, 0]
        self._sim_elem_chk:        list = []
        self._sim_elem_sym:        list = []   # QComboBox symbol
        self._sim_elem_edge_combo: list = []   # QComboBox edge
        self._sim_elem_lbl:        list = []   # QLabel edge energy
        self._sim_elem_emin:       list = []   # QDoubleSpinBox
        self._sim_elem_emax:       list = []   # QDoubleSpinBox
        self._sim_elem_en:         list = []   # QSpinBox N
        self._sim_elem_loc:        list = []   # QComboBox Core/Shell location

        for i in range(3):
            ew = QWidget()
            ev = QVBoxLayout(ew)
            ev.setContentsMargins(0, 2, 0, 4)
            ev.setSpacing(2)

            # Row 1: enable checkbox / label  +  symbol combo  +  edge combo  +  edge label
            r1 = QHBoxLayout()
            if i == 0:
                r1.addWidget(QLabel('Element 1:'))
                chk = None
            else:
                chk = QCheckBox(f'Element {i+1}:')
                chk.setChecked(False)
                r1.addWidget(chk)
            self._sim_elem_chk.append(chk)

            sym_cb = QComboBox()
            sym_cb.setMinimumWidth(95)
            for sym, Z in _sym_list:
                sym_cb.addItem(f'{sym} ({Z})')
            # Set default selection
            def_sym, def_Z = _DEFAULT_ELEMS[i]
            idx_def = sym_cb.findText(f'{def_sym} ({def_Z})')
            if idx_def >= 0:
                sym_cb.setCurrentIndex(idx_def)
            self._sim_elem_sym.append(sym_cb)
            r1.addWidget(sym_cb)

            edge_cb = QComboBox()
            edge_cb.addItems(_SHELLS_ALL)
            edge_cb.setCurrentText(_DEFAULT_EDGES[i])
            edge_cb.setFixedWidth(52)
            self._sim_elem_edge_combo.append(edge_cb)
            r1.addWidget(edge_cb)

            lbl_e = QLabel()
            lbl_e.setStyleSheet('color: #888; font-size: 11px;')
            lbl_e.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self._sim_elem_lbl.append(lbl_e)
            r1.addWidget(lbl_e, 1)
            ev.addLayout(r1)

            # Row 2: E_min – E_max  N  Amp  (indented)
            r2 = QHBoxLayout()
            r2.addSpacing(16)
            r2.addWidget(QLabel('E:'))
            emin = _dbl(10.419, lo=0.1, hi=1000.0, decimals=4, step=0.01)
            emin.setFixedWidth(78)
            emax = _dbl(10.919, lo=0.1, hi=1000.0, decimals=4, step=0.01)
            emax.setFixedWidth(78)
            r2.addWidget(emin); r2.addWidget(QLabel('–')); r2.addWidget(emax)
            r2.addWidget(QLabel('keV'))
            r2.addSpacing(8)
            r2.addWidget(QLabel('N:'))
            n_sb = _int(20, lo=2, hi=200)
            n_sb.setFixedWidth(52)
            r2.addWidget(n_sb)
            r2.addSpacing(8)
            r2.addWidget(QLabel('Location:'))
            loc_cb = QComboBox()
            loc_cb.addItems(['Core', 'Shell'])
            loc_cb.setCurrentText('Core' if i == 0 else 'Shell')
            loc_cb.setFixedWidth(75)
            r2.addWidget(loc_cb)
            r2.addStretch()
            self._sim_elem_emin.append(emin)
            self._sim_elem_emax.append(emax)
            self._sim_elem_en.append(n_sb)
            self._sim_elem_loc.append(loc_cb)
            ev.addLayout(r2)

            if i != 0:
                for w in (sym_cb, edge_cb, emin, emax, n_sb, loc_cb):
                    w.setEnabled(False)

            en_lay.addWidget(ew)

            sym_cb.currentIndexChanged.connect(
                lambda _, ei=i: self._on_sim_symbol_changed(ei))
            edge_cb.currentIndexChanged.connect(
                lambda _, ei=i: self._on_sim_edge_changed(ei))
            if chk is not None:
                chk.toggled.connect(
                    lambda checked, ei=i: self._on_sim_elem_toggled(checked, ei))

        note = QLabel(
            '<i>Each element gets its own energy range near its absorption edge.<br>'
            'Energies are spaced for equal Δf′ steps; ranges are combined into one<br>'
            'shared grid.  Location sets which region of the particle each element<br>'
            'occupies; Core/Shell pairs have distinct I(q) shapes for all partials.</i>')
        note.setWordWrap(True)
        en_lay.addWidget(note)

        self._btn_preview_E = QPushButton('Preview selected energies')
        en_lay.addWidget(self._btn_preview_E)
        form_lay.addWidget(en_grp)

        # Backward-compat aliases used by _build_config / _preview_energies
        self._exp_Z    = None          # removed — use _sim_elem_sym[0]
        self._lbl_edge = self._sim_elem_lbl[0]
        self._exp_Emin = self._sim_elem_emin[0]
        self._exp_Emax = self._sim_elem_emax[0]
        self._exp_N    = self._sim_elem_en[0]

        self._btn_preview_E.clicked.connect(self._preview_energies)
        for i in range(3):
            self._on_sim_symbol_changed(i)

        # ── Sample ────────────────────────────────────────────────────────────
        samp_grp  = QGroupBox('Sample')
        samp_form = QFormLayout(samp_grp)
        self._exp_phi = _dbl(1e-4, lo=1e-10, hi=1.0, decimals=6, step=1e-5)
        samp_form.addRow('Volume fraction φ:', self._exp_phi)
        form_lay.addWidget(samp_grp)

        # ── q grid ────────────────────────────────────────────────────────────
        q_grp  = QGroupBox('q grid')
        q_form = QFormLayout(q_grp)
        self._q_min = _dbl(0.003, lo=1e-5, hi=10.0, decimals=5, step=0.001)
        self._q_max = _dbl(0.145, lo=1e-5, hi=10.0, decimals=5, step=0.001)
        self._q_n   = _int(500, lo=10, hi=10000)
        self._q_log = QComboBox(); self._q_log.addItems(['Logarithmic', 'Linear'])
        q_form.addRow('q_min (Å⁻¹):', self._q_min)
        q_form.addRow('q_max (Å⁻¹):', self._q_max)
        q_form.addRow('N points:',     self._q_n)
        q_form.addRow('Spacing:',      self._q_log)
        form_lay.addWidget(q_grp)

        # ── Noise ─────────────────────────────────────────────────────────────
        noise_grp  = QGroupBox('Noise model')
        noise_form = QFormLayout(noise_grp)
        self._rb_poisson  = QRadioButton('Poissonian  σ(q) = √(I(q)·I(q_min)/N_peak)')
        self._rb_relative = QRadioButton('Relative     σ(q) = p · I(q)')
        self._rb_poisson.setChecked(True)
        bg = QButtonGroup(); bg.addButton(self._rb_poisson); bg.addButton(self._rb_relative)
        self._noise_bg = bg
        self._n_peak = _int(10000, lo=1, hi=10000000)
        self._n_rel  = _dbl(0.005, lo=1e-6, hi=1.0, decimals=4, step=0.001)
        noise_form.addRow(self._rb_poisson)
        noise_form.addRow('N_peak (photons at q_min):', self._n_peak)
        noise_form.addRow(self._rb_relative)
        noise_form.addRow('Relative σ/I:', self._n_rel)
        form_lay.addWidget(noise_grp)

        # ── Output ────────────────────────────────────────────────────────────
        out_grp  = QGroupBox('Output')
        out_form = QFormLayout(out_grp)
        self._out_dir  = QLineEdit(str(Path.home() / 'asaxs_sim_data'))
        self._out_name = QLineEdit('particle')
        btn_browse = QPushButton('Browse…'); btn_browse.setFixedWidth(70)
        dr = QHBoxLayout(); dr.addWidget(self._out_dir); dr.addWidget(btn_browse)
        out_form.addRow('Directory:', dr)
        out_form.addRow('File prefix:', self._out_name)
        form_lay.addWidget(out_grp)
        btn_browse.clicked.connect(self._browse_dir)

        # ── Settings save/load ────────────────────────────────────────────────
        settings_row = QHBoxLayout()
        self._btn_save_cfg = QPushButton('Save settings…')
        self._btn_load_cfg = QPushButton('Load settings…')
        settings_row.addWidget(self._btn_save_cfg)
        settings_row.addWidget(self._btn_load_cfg)
        form_lay.addLayout(settings_row)
        self._btn_save_cfg.clicked.connect(self._save_settings)
        self._btn_load_cfg.clicked.connect(self._load_settings)

        # ── Generate ──────────────────────────────────────────────────────────
        self._btn_gen = QPushButton('Generate data')
        self._btn_gen.setFixedHeight(36)
        self._btn_gen.setStyleSheet('font-weight: bold; font-size: 13px;')
        form_lay.addWidget(self._btn_gen)
        form_lay.addStretch()
        self._btn_gen.clicked.connect(self._generate)

        # ══ RIGHT: plot (top) + log (bottom) ═════════════════════════════════
        right_w   = QWidget()
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_split = QSplitter(Qt.Orientation.Vertical)
        right_lay.addWidget(right_split)
        main_split.addWidget(right_w)

        # ── Plot ──────────────────────────────────────────────────────────────
        plot_w   = QWidget()
        plot_lay = QVBoxLayout(plot_w)
        plot_lay.setContentsMargins(4, 4, 4, 0)
        self._pw = pg.PlotWidget()
        self._pw.setBackground('#1e1e1e')
        self._pw.showGrid(x=True, y=True, alpha=0.3)
        self._pw.setLogMode(x=True, y=True)
        self._pw.setLabel('bottom', 'q', units='Å⁻¹')
        self._pw.setLabel('left',   'I(q)', units='cm⁻¹')
        self._pw.addLegend(offset=(10, 10))
        self._coord_lbl = QLabel('x = —    y = —')
        self._coord_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_lay.addWidget(self._pw)
        plot_lay.addWidget(self._coord_lbl)
        self._proxy = add_crosshair(self._pw, x_fmt='.4g', y_fmt='.4e', label=self._coord_lbl)
        right_split.addWidget(plot_w)

        # ── Log ───────────────────────────────────────────────────────────────
        log_w   = QWidget()
        log_lay = QVBoxLayout(log_w)
        log_lay.setContentsMargins(4, 4, 4, 4)
        log_lay.addWidget(QLabel('Log'))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        mono = QFont('Courier New', 10); mono.setStyleHint(QFont.StyleHint.Monospace)
        self._log.setFont(mono)
        btn_cl = QPushButton('Clear'); btn_cl.setFixedWidth(70)
        btn_cl.clicked.connect(self._log.clear)
        log_lay.addWidget(self._log)
        log_lay.addWidget(btn_cl)
        right_split.addWidget(log_w)

        right_split.setSizes([500, 260])
        main_split.setSizes([400, 800])

    # ─── Formula fill handlers ────────────────────────────────────────────────

    def _fill_core_from_formula(self):
        formula = self._core_formula.text().strip()
        if not formula:
            return
        try:
            M, ne = _formula_to_Mne(formula)
            self._core_M.setValue(M)
            self._core_ne.setValue(ne)
            counts = _parse_formula(formula)
            if len(counts) == 1:
                sym = next(iter(counts))
                Z = _element_Z(sym)
                if Z:
                    self._core_Z.setValue(Z)
            rho = self._core_rho.value()
            sld = _elec_density(rho, M, ne)
            self._lbl_core_sld.setText(f'M = {M:.3f} g/mol,  ne = {ne},  ρ_e = {sld:.3f} e/Å³')
        except Exception as e:
            self._lbl_core_sld.setText(f'Error: {e}')

    def _fill_shell_from_formula(self):
        formula = self._shell_formula.text().strip()
        if not formula:
            return
        try:
            M, ne = _formula_to_Mne(formula)
            self._shell_M.setValue(M)
            self._shell_ne.setValue(ne)
            rho = self._shell_rho.value()
            sld = _elec_density(rho, M, ne)
            self._lbl_shell_sld.setText(f'M = {M:.3f} g/mol,  ne = {ne},  ρ_e = {sld:.3f} e/Å³')
        except Exception as e:
            self._lbl_shell_sld.setText(f'Error: {e}')

    def _fill_solv_from_formula(self):
        formula = self._solv_formula.text().strip()
        if not formula:
            return
        try:
            M, ne = _formula_to_Mne(formula)
            self._solv_M.setValue(M)
            self._solv_ne.setValue(ne)
            rho = self._solv_rho.value()
            sld = _elec_density(rho, M, ne)
            self._lbl_solv_sld.setText(f'M = {M:.3f} g/mol,  ne = {ne},  ρ_e = {sld:.3f} e/Å³')
        except Exception as e:
            self._lbl_solv_sld.setText(f'Error: {e}')

    # ─── Preset appliers ──────────────────────────────────────────────────────

    def _apply_core_preset(self):
        nm = self._core_preset.currentText()
        if nm in _ELEMENT_PRESETS:
            p = _ELEMENT_PRESETS[nm]
            self._core_Z.setValue(p['Z'])
            self._core_rho.setValue(p['mass_density'])
            self._core_M.setValue(p['molar_mass'])
            self._core_ne.setValue(p['n_electrons'])

    def _apply_shell_preset(self):
        nm = self._shell_preset.currentText()
        if nm in _MATERIAL_PRESETS:
            p = _MATERIAL_PRESETS[nm]
            self._shell_rho.setValue(p['mass_density']); self._shell_M.setValue(p['molar_mass'])
            self._shell_ne.setValue(p['n_electrons'])

    def _apply_solv_preset(self):
        nm = self._solv_preset.currentText()
        if nm in _MATERIAL_PRESETS:
            p = _MATERIAL_PRESETS[nm]
            self._solv_rho.setValue(p['mass_density']); self._solv_M.setValue(p['molar_mass'])
            self._solv_ne.setValue(p['n_electrons'])

    def _update_thickness(self):
        delta = self._shell_Rt.value() - self._core_R.value()
        sig   = self._shell_sig.value()
        if sig > 0:
            self._lbl_thick.setText(f'{delta:.1f} Å  (R_outer σ={sig*100:.0f}%)')
        else:
            self._lbl_thick.setText(f'{delta:.1f} Å')

    def _elem_Z_from_combo(self, elem_idx: int) -> int:
        """Return integer Z from the symbol combo for elem_idx."""
        txt = self._sim_elem_sym[elem_idx].currentText()  # e.g. "Au (79)"
        try:
            return int(txt.split('(')[1].rstrip(')'))
        except Exception:
            return 0

    def _on_sim_symbol_changed(self, elem_idx: int):
        """Called when symbol combo changes — update Z, reset edge combo, update E range."""
        Z = self._elem_Z_from_combo(elem_idx)
        self._sim_elem_Z[elem_idx] = Z
        lbl = self._sim_elem_lbl[elem_idx]
        edge_cb = self._sim_elem_edge_combo[elem_idx]

        if not _HAS_XRAYDB:
            lbl.setText('xraydb not installed')
            return

        # Rebuild edge combo with available edges for this Z
        try:
            edges = xraydb.xray_edges(Z)
        except Exception:
            edges = {}

        _SHELLS_ALL = ['K','L1','L2','L3','M1','M2','M3','M4','M5','N1','N2','N3']
        edge_cb.blockSignals(True)
        prev = edge_cb.currentText()
        edge_cb.clear()
        available = [sh for sh in _SHELLS_ALL if sh in edges and edges[sh].energy > 100]
        if not available:
            available = [sh for sh in _SHELLS_ALL if sh in edges]
        if not available:
            available = ['K']
        for sh in available:
            edge_cb.addItem(sh)
        if prev in available:
            edge_cb.setCurrentText(prev)
        else:
            # Default: L3 for heavy elements, K for light
            default = 'L3' if Z > 50 and 'L3' in available else available[0]
            edge_cb.setCurrentText(default)
        edge_cb.blockSignals(False)

        self._on_sim_edge_changed(elem_idx)

    def _on_sim_edge_changed(self, elem_idx: int):
        """Called when edge combo changes — update E_max default and label."""
        Z = self._sim_elem_Z[elem_idx]
        if not isinstance(Z, int) or Z == 0:
            Z = self._elem_Z_from_combo(elem_idx)
            self._sim_elem_Z[elem_idx] = Z
        shell = self._sim_elem_edge_combo[elem_idx].currentText()
        lbl   = self._sim_elem_lbl[elem_idx]

        if not _HAS_XRAYDB or Z == 0:
            return
        try:
            edge_eV = xraydb.xray_edge(Z, shell).energy
        except Exception:
            lbl.setText('edge not found')
            return

        edge_keV = edge_eV / 1000.0
        emax_sb = self._sim_elem_emax[elem_idx]
        emin_sb = self._sim_elem_emin[elem_idx]
        emax_sb.blockSignals(True); emin_sb.blockSignals(True)
        emax_sb.setValue(edge_keV)
        emin_sb.setValue(max(0.1, edge_keV - 0.5))
        emax_sb.blockSignals(False); emin_sb.blockSignals(False)
        lbl.setText(f'{shell} edge: {edge_keV:.4f} keV')

    def _on_sim_elem_toggled(self, checked: bool, elem_idx: int):
        """Enable/disable all widgets for element elem_idx."""
        for w in (self._sim_elem_sym[elem_idx],
                  self._sim_elem_edge_combo[elem_idx],
                  self._sim_elem_emin[elem_idx],
                  self._sim_elem_emax[elem_idx],
                  self._sim_elem_en[elem_idx],
                  self._sim_elem_loc[elem_idx]):
            w.setEnabled(checked)

    def _update_sim_edge_info(self, elem_idx: int = 0):
        self._on_sim_edge_changed(elem_idx)

    def _update_edge_info(self):
        self._on_sim_edge_changed(0)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, 'Output directory', self._out_dir.text())
        if d: self._out_dir.setText(d)

    def _preview_energies(self):
        if not _HAS_XRAYDB:
            self._log.append('xraydb not installed'); return
        cfg = self._build_config()
        elems = cfg['resonant_elements']
        self._log.append('<b>Energy preview (per element):</b>')
        for i, e in enumerate(elems):
            self._log.append(
                f'  Elem {i+1}: Z={e["Z"]}  '
                f'E: {e["E_min_keV"]:.4f}–{e["E_max_keV"]:.4f} keV  N={e["n_energies"]}')
        try:
            Es, elem_results = _select_energies_multi(cfg)
            self._log.append(f'  Combined grid: {len(Es)} energies')
            n = len(elem_results)
            fp_hdrs  = ''.join(f"  {('fp_'+str(i+1)):>9}"  for i in range(n))
            fpp_hdrs = ''.join(f"  {('fpp_'+str(i+1)):>9}" for i in range(n))
            hdr = f'  {"E (keV)":>10}' + fp_hdrs + fpp_hdrs
            self._log.append(hdr)
            self._log.append('  ' + '-' * len(hdr))
            for k, e in enumerate(Es):
                row = f'  {e:10.4f}'
                for er in elem_results: row += f'  {er["fp"][k]:9.3f}'
                for er in elem_results: row += f'  {er["fpp"][k]:9.3f}'
                self._log.append(row)
            for i, er in enumerate(elem_results):
                fp = er['fp']
                self._log.append(
                    f"  Elem {i+1} (Z={er['Z']}): "
                    f"f' range {fp.min():.3f}→{fp.max():.3f} e  Δf'={fp.max()-fp.min():.3f} e")
        except Exception as ex:
            self._log.append(f'  Error: {ex}')

    # ─── Build config ─────────────────────────────────────────────────────────

    def _build_config(self) -> dict:
        out_path = Path(self._out_dir.text()) / self._out_name.text()
        # Collect active resonant elements with per-element energy ranges
        res_elems = [{
            'Z':          self._sim_elem_Z[0],
            'location':   'core' if self._sim_elem_loc[0].currentText() == 'Core' else 'shell',
            'E_min_keV':  self._sim_elem_emin[0].value(),
            'E_max_keV':  self._sim_elem_emax[0].value(),
            'n_energies': self._sim_elem_en[0].value(),
        }]
        for i in range(1, 3):
            if self._sim_elem_chk[i] is not None and self._sim_elem_chk[i].isChecked():
                res_elems.append({
                    'Z':          self._sim_elem_Z[i],
                    'location':   'core' if self._sim_elem_loc[i].currentText() == 'Core' else 'shell',
                    'E_min_keV':  self._sim_elem_emin[i].value(),
                    'E_max_keV':  self._sim_elem_emax[i].value(),
                    'n_energies': self._sim_elem_en[i].value(),
                })
        Z0 = self._sim_elem_Z[0]
        return {
            'system': {'description': f'Z={self._core_Z.value()} core R_mean={self._core_R.value():.1f}A'},
            'core':   {'Z': self._core_Z.value(), 'mass_density': self._core_rho.value(),
                       'molar_mass': self._core_M.value(), 'n_electrons': self._core_ne.value(),
                       'radius_mean': self._core_R.value(), 'sigma_rel': self._core_sig.value()},
            'shell':  {'mass_density': self._shell_rho.value(), 'molar_mass': self._shell_M.value(),
                       'n_electrons': self._shell_ne.value(), 'radius_outer': self._shell_Rt.value(),
                       'sigma_rel': self._shell_sig.value()},
            'solvent':{'mass_density': self._solv_rho.value(), 'molar_mass': self._solv_M.value(),
                       'n_electrons': self._solv_ne.value()},
            'experiment': {'volume_fraction': self._exp_phi.value(),
                           'resonant_Z': Z0,
                           'E_min_keV': res_elems[0]['E_min_keV'],
                           'E_max_keV': res_elems[0]['E_max_keV'],
                           'n_energies': res_elems[0]['n_energies']},
            'resonant_elements': res_elems,
            'q_grid': {'q_min': self._q_min.value(), 'q_max': self._q_max.value(),
                       'n_points': self._q_n.value(),
                       'spacing': 'log' if self._q_log.currentIndex() == 0 else 'linear'},
            'noise':  {'model': 'poisson' if self._rb_poisson.isChecked() else 'relative',
                       'N_peak': self._n_peak.value(), 'relative': self._n_rel.value()},
            'output': {'directory': str(out_path), 'name': self._out_name.text()},
        }

    # ─── Settings save / load ────────────────────────────────────────────────

    def _save_settings(self):
        import json
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save simulation settings', '',
            'ASAXS sim settings (*.asaxs_sim);;JSON files (*.json);;All files (*)')
        if not path:
            return
        cfg = self._build_config()
        cfg['_out_dir_raw'] = self._out_dir.text()   # preserve raw dir separately
        try:
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=2)
            self._log.append(f'<b>Settings saved:</b> {path}')
        except Exception as e:
            self._log.append(f'<b>Save error:</b> {e}')

    def _load_settings(self):
        import json
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load simulation settings', '',
            'ASAXS sim settings (*.asaxs_sim);;JSON files (*.json);;All files (*)')
        if not path:
            return
        try:
            with open(path, 'r') as f:
                cfg = json.load(f)
            self._apply_config(cfg)
            self._log.append(f'<b>Settings loaded:</b> {path}')
        except Exception as e:
            self._log.append(f'<b>Load error:</b> {e}')

    def _apply_config(self, cfg: dict):
        """Populate all UI widgets from a saved config dict."""
        c = cfg.get('core', {})
        if 'Z'            in c: self._core_Z.setValue(c['Z'])
        if 'mass_density' in c: self._core_rho.setValue(c['mass_density'])
        if 'molar_mass'   in c: self._core_M.setValue(c['molar_mass'])
        if 'n_electrons'  in c: self._core_ne.setValue(c['n_electrons'])
        if 'radius_mean'  in c: self._core_R.setValue(c['radius_mean'])
        if 'sigma_rel'    in c: self._core_sig.setValue(c['sigma_rel'])

        s = cfg.get('shell', {})
        if 'mass_density' in s: self._shell_rho.setValue(s['mass_density'])
        if 'molar_mass'   in s: self._shell_M.setValue(s['molar_mass'])
        if 'n_electrons'  in s: self._shell_ne.setValue(s['n_electrons'])
        if 'radius_outer' in s: self._shell_Rt.setValue(s['radius_outer'])
        if 'sigma_rel'    in s: self._shell_sig.setValue(s['sigma_rel'])

        v = cfg.get('solvent', {})
        if 'mass_density' in v: self._solv_rho.setValue(v['mass_density'])
        if 'molar_mass'   in v: self._solv_M.setValue(v['molar_mass'])
        if 'n_electrons'  in v: self._solv_ne.setValue(v['n_electrons'])

        e = cfg.get('experiment', {})
        if 'volume_fraction' in e: self._exp_phi.setValue(e['volume_fraction'])

        elems = cfg.get('resonant_elements', [])
        for i in range(3):
            active = i < len(elems)
            if i > 0:
                chk = self._sim_elem_chk[i]
                if chk is not None:
                    chk.setChecked(active)
            if not active:
                continue
            elem = elems[i]
            Z = elem.get('Z', 79)
            # Restore symbol combo
            sym_cb = self._sim_elem_sym[i]
            for j in range(sym_cb.count()):
                txt = sym_cb.itemText(j)
                try:
                    if int(txt.split('(')[1].rstrip(')')) == Z:
                        sym_cb.setCurrentIndex(j)
                        break
                except Exception:
                    pass
            # Edge combo — don't auto-update; just set E_min/E_max/N directly
            self._sim_elem_emin[i].setValue(elem.get('E_min_keV', 10.0))
            self._sim_elem_emax[i].setValue(elem.get('E_max_keV', 11.0))
            self._sim_elem_en[i].setValue(int(elem.get('n_energies', 20)))
            loc = elem.get('location', 'core' if i == 0 else 'shell')
            self._sim_elem_loc[i].setCurrentText('Core' if loc == 'core' else 'Shell')
            # Trigger symbol change to update edge label
            self._on_sim_symbol_changed(i)

        qg = cfg.get('q_grid', {})
        if 'q_min'    in qg: self._q_min.setValue(qg['q_min'])
        if 'q_max'    in qg: self._q_max.setValue(qg['q_max'])
        if 'n_points' in qg: self._q_n.setValue(qg['n_points'])
        if 'spacing'  in qg:
            self._q_log.setCurrentIndex(0 if qg['spacing'] == 'log' else 1)

        n = cfg.get('noise', {})
        if n.get('model') == 'poisson':
            self._rb_poisson.setChecked(True)
        elif n.get('model') == 'relative':
            self._rb_relative.setChecked(True)
        if 'N_peak'   in n: self._n_peak.setValue(int(n['N_peak']))
        if 'relative' in n: self._n_rel.setValue(n['relative'])

        out = cfg.get('output', {})
        if '_out_dir_raw' in cfg:
            self._out_dir.setText(cfg['_out_dir_raw'])
        elif 'directory' in out:
            self._out_dir.setText(str(Path(out['directory']).parent))
        if 'name' in out:
            self._out_name.setText(out['name'])

    # ─── Generation ──────────────────────────────────────────────────────────

    def _generate(self):
        if not _HAS_XRAYDB:
            self._log.append('<b>Error:</b> xraydb not installed — run: pip install xraydb'); return
        if self._thread and self._thread.isRunning():
            return
        import json
        cfg = self._build_config()
        cfg['_out_dir_raw'] = self._out_dir.text()

        # Auto-save settings alongside the data files
        try:
            out_dir = Path(cfg['output']['directory'])
            out_dir.mkdir(parents=True, exist_ok=True)
            settings_path = out_dir / f"{cfg['output']['name']}_settings.asaxs_sim"
            with open(settings_path, 'w') as _f:
                json.dump(cfg, _f, indent=2)
        except Exception as _e:
            settings_path = None

        self._log.clear()
        elems = cfg['resonant_elements']
        self._log.append('<b>Generating ASAXS data …</b>')
        self._log.append(f'Core:    Z={cfg["core"]["Z"]}, R_mean={cfg["core"]["radius_mean"]:.1f} Å'
                         f', σ_rel={cfg["core"]["sigma_rel"]*100:.0f}%')
        self._log.append(f'Shell:   R_outer={cfg["shell"]["radius_outer"]:.1f} Å')
        elem_str = '  '.join(
            f'Z={e["Z"]} loc={e.get("location","core")} '
            f'E={e["E_min_keV"]:.4f}–{e["E_max_keV"]:.4f} keV N={e["n_energies"]}'
            for e in elems)
        self._log.append(f'Elements ({len(elems)}): {elem_str}')
        self._log.append(f'Output:  {cfg["output"]["directory"]}/')
        if settings_path:
            self._log.append(f'Settings: {settings_path.name}  (auto-saved)')
        self._log.append('')
        self._btn_gen.setEnabled(False)
        self._btn_gen.setText('Generating …')

        self._thread = _GeneratorThread(cfg, self)
        self._thread.log_line.connect(self._log.append)
        self._thread.data_ready.connect(self._plot_data)
        self._thread.finished_ok.connect(self._on_done)
        self._thread.error_signal.connect(self._on_error)
        self._thread.start()

    def _plot_data(self, q, energies, I_obs_list):
        # Clear old curves
        for c in self._curves: self._pw.removeItem(c)
        self._curves.clear()
        self._pw.getPlotItem().legend.clear()

        N = len(energies)
        E_min, E_max = energies[0], energies[-1]
        dE = E_max - E_min if E_max > E_min else 1.0

        for i, (E, I_obs) in enumerate(zip(energies, I_obs_list)):
            t   = (E - E_min) / dE
            r, g, b = _energy_color(t)
            pen = pg.mkPen(color=(r, g, b, 200), width=1.2)
            c   = self._pw.plot(q, I_obs, pen=pen, name=f'{E:.4f} keV')
            self._curves.append(c)

        self._pw.setTitle(f'Simulated I(q, E) — {N} datasets')

    def _on_done(self, out_dir):
        self._log.append(f'\n<b>✓ Complete.</b>  Files in: {out_dir}')
        self._btn_gen.setEnabled(True); self._btn_gen.setText('Generate data')

    def _on_error(self, tb):
        self._log.append(f'\n<b>Error:</b>\n{tb}')
        self._btn_gen.setEnabled(True); self._btn_gen.setText('Generate data')
