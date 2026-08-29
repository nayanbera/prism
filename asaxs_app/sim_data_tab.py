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

    Returns (energies_list, elem_results) where elem_results is a list of
    {'Z', 'fp', 'fpp', 'scale'} dicts — one per active element.
    Energies are spaced equidistant in f'(E) for element 1 (primary).
    """
    exp      = cfg['experiment']
    E0, E1, N = exp['E_min_keV'], exp['E_max_keV'], exp['n_energies']
    elements  = cfg.get('resonant_elements', [{'Z': exp['resonant_Z'], 'scale': 1.0}])

    # Select energies equidistant in Δf' for the primary element
    Z0  = elements[0]['Z']
    Ed  = np.linspace(E0, E1, 3000)
    fpd = np.array([xraydb.f1_chantler(Z0, e*1000) for e in Ed])
    idx = np.argsort(fpd)
    tgt = np.linspace(fpd[idx[0]], fpd[idx[-1]], N)
    Es  = sorted(np.interp(tgt, fpd[idx], Ed[idx]))

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
        elem_results.append({'Z': Z, 'fp': fp, 'fpp': fpp, 'scale': elem.get('scale', 1.0)})
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

            self.log_line.emit('Computing partial structure factors (base model) …')
            I_MM_base, I_RM_base, I_RR_base = _compute_partials(q, cfg, self.log_line.emit)
            self.log_line.emit(
                f'  I_MM={I_MM_base[0]:.3e}  I_RM={I_RM_base[0]:.3e}  I_RR={I_RR_base[0]:.3e} cm⁻¹ (at q_min)')

            self.log_line.emit('\nSelecting energies (equidistant Δf′ for elem 1) …')
            energies, elem_results = _select_energies_multi(cfg, self.log_line.emit)
            n_elem = len(elem_results)

            # Build per-element scaled partials
            # I_RiM  = scale_i × I_RM_base
            # I_RiRi = scale_i² × I_RR_base
            # I_RiRj = scale_i × scale_j × I_RR_base  (cross-terms, co-location assumed)
            scales = [e['scale'] for e in elem_results]
            I_RiM  = [scales[i] * I_RM_base  for i in range(n_elem)]
            I_RiRi = [scales[i]**2 * I_RR_base for i in range(n_elem)]
            I_RiRj = {}
            for i in range(n_elem):
                for j in range(i+1, n_elem):
                    I_RiRj[(i,j)] = scales[i] * scales[j] * I_RR_base

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
                I_tot = I_MM_base.copy()
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
            cols_gt  = [q, I_MM_base]
            for i in range(n_elem):
                cols_gt.append(I_RiM[i])
            for i in range(n_elem):
                cols_gt.append(I_RiRi[i])
            for i in range(n_elem):
                for j in range(i+1, n_elem):
                    cols_gt.append(I_RiRj[(i,j)])
            hdr_gt = 'q(1/A)  ' + '  '.join(f'{nm}(cm-1)' for nm in names_gt)
            scales_str = '  '.join(f'scale_{i+1}={scales[i]:.3f}' for i in range(n_elem))
            hdr_gt = f'Resonant elements: {[e["Z"] for e in elem_results]}  {scales_str}\n' + hdr_gt
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
        en_grp  = QGroupBox('Energies')
        en_form = QFormLayout(en_grp)

        # Three resonant-element rows
        self._sim_elem_Z:     list = []
        self._sim_elem_amp:   list = []
        self._sim_elem_chk:   list = []
        self._sim_elem_lbl:   list = []

        elem_widget = QWidget()
        elem_vlay   = QVBoxLayout(elem_widget)
        elem_vlay.setContentsMargins(0, 0, 0, 0)
        elem_vlay.setSpacing(3)
        for i in range(3):
            row = QHBoxLayout()
            if i == 0:
                row.addWidget(QLabel('Elem 1 (primary):'))
                chk = None
            else:
                chk = QCheckBox(f'Elem {i+1}:')
                chk.setChecked(False)
                row.addWidget(chk)
            self._sim_elem_chk.append(chk)

            row.addWidget(QLabel('Z:'))
            z_sb = _int(79 - i*5, lo=1, hi=118)
            z_sb.setFixedWidth(55)
            self._sim_elem_Z.append(z_sb)
            row.addWidget(z_sb)

            row.addWidget(QLabel('Amp:'))
            amp_sb = _dbl(1.0 if i == 0 else 0.5, lo=0.0, hi=100.0, decimals=3, step=0.05)
            amp_sb.setFixedWidth(70)
            self._sim_elem_amp.append(amp_sb)
            row.addWidget(amp_sb)

            lbl = QLabel()
            lbl.setStyleSheet('color: #666; font-size: 11px;')
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self._sim_elem_lbl.append(lbl)
            row.addWidget(lbl, 1)
            elem_vlay.addLayout(row)

            # Wire signals
            z_sb.valueChanged.connect(lambda v, ei=i: self._update_sim_edge_info(ei))
            if chk is not None:
                chk.toggled.connect(lambda checked, ei=i: self._sim_elem_Z[ei].setEnabled(checked)
                                    or self._sim_elem_amp[ei].setEnabled(checked))
                z_sb.setEnabled(False)
                amp_sb.setEnabled(False)
                chk.toggled.connect(lambda checked, ei=i: (
                    self._sim_elem_Z[ei].setEnabled(checked),
                    self._sim_elem_amp[ei].setEnabled(checked),
                ))

        en_form.addRow('Resonant elements:', elem_widget)

        # Backward-compat alias — _exp_Z points to elem-1 spinbox
        self._exp_Z    = self._sim_elem_Z[0]
        self._lbl_edge = self._sim_elem_lbl[0]

        self._exp_Emin = _dbl(10.919, lo=0.1, hi=1000.0, decimals=4, step=0.01)
        self._exp_Emax = _dbl(11.919, lo=0.1, hi=1000.0, decimals=4, step=0.01)
        self._exp_N    = _int(20, lo=2, hi=200)

        en_form.addRow('E_min (keV):',  self._exp_Emin)
        en_form.addRow('E_max (keV):',  self._exp_Emax)
        en_form.addRow('N energies:',   self._exp_N)

        note = QLabel(
            '<i>Energies are spaced for equal Δf′ steps across [E_min, E_max].<br>'
            'Amp = relative resonant amplitude (1.0 = same as primary element).<br>'
            'Cross-terms I_RiRj assume co-located resonant atoms.</i>')
        note.setWordWrap(True)
        en_form.addRow(note)

        self._btn_preview_E = QPushButton('Preview selected energies')
        en_form.addRow(self._btn_preview_E)
        form_lay.addWidget(en_grp)

        self._exp_Z.valueChanged.connect(lambda v: self._update_sim_edge_info(0))
        self._core_Z.valueChanged.connect(lambda v: self._exp_Z.setValue(v))
        self._btn_preview_E.clicked.connect(self._preview_energies)
        for i in range(3):
            self._update_sim_edge_info(i)

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
                    self._exp_Z.setValue(Z)
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
            self._exp_Z.setValue(p['Z'])

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

    def _update_sim_edge_info(self, elem_idx: int = 0):
        lbl = self._sim_elem_lbl[elem_idx]
        if not _HAS_XRAYDB:
            lbl.setText('xraydb not installed'); return
        Z = self._sim_elem_Z[elem_idx].value()
        try:
            edges = xraydb.xray_edges(Z)
            parts = []
            for sh in ['K', 'L3', 'L2', 'L1', 'M5']:
                if sh in edges:
                    parts.append(f'{sh}: {edges[sh].energy/1000:.3f} keV')
                    if len(parts) == 3: break
            lbl.setText('  '.join(parts))
        except Exception:
            lbl.setText('')

    def _update_edge_info(self):
        self._update_sim_edge_info(0)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, 'Output directory', self._out_dir.text())
        if d: self._out_dir.setText(d)

    def _preview_energies(self):
        if not _HAS_XRAYDB:
            self._log.append('xraydb not installed'); return
        cfg = self._build_config()
        elems = cfg['resonant_elements']
        self._log.append('<b>Energy preview:</b>')
        self._log.append(
            f'  {len(elems)} element(s): Z={[e["Z"] for e in elems]}  '
            f'E: {cfg["experiment"]["E_min_keV"]:.4f}–{cfg["experiment"]["E_max_keV"]:.4f} keV  '
            f'N={cfg["experiment"]["n_energies"]}')
        try:
            Es, elem_results = _select_energies_multi(cfg)
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
        # Collect active resonant elements
        res_elems = [{'Z': self._sim_elem_Z[0].value(),
                      'scale': self._sim_elem_amp[0].value()}]
        for i in range(1, 3):
            if self._sim_elem_chk[i] is not None and self._sim_elem_chk[i].isChecked():
                res_elems.append({'Z':    self._sim_elem_Z[i].value(),
                                  'scale': self._sim_elem_amp[i].value()})
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
                           'resonant_Z': self._sim_elem_Z[0].value(),
                           'E_min_keV': self._exp_Emin.value(), 'E_max_keV': self._exp_Emax.value(),
                           'n_energies': self._exp_N.value()},
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
        if 'E_min_keV'       in e: self._exp_Emin.setValue(e['E_min_keV'])
        if 'E_max_keV'       in e: self._exp_Emax.setValue(e['E_max_keV'])
        if 'n_energies'      in e: self._exp_N.setValue(e['n_energies'])

        elems = cfg.get('resonant_elements', [])
        if elems:
            self._sim_elem_Z[0].setValue(elems[0].get('Z', 79))
            self._sim_elem_amp[0].setValue(elems[0].get('scale', 1.0))
        for i in range(1, 3):
            chk = self._sim_elem_chk[i]
            if chk is None:
                continue
            if i < len(elems):
                chk.setChecked(True)
                self._sim_elem_Z[i].setValue(elems[i].get('Z', 79))
                self._sim_elem_amp[i].setValue(elems[i].get('scale', 0.5))
            else:
                chk.setChecked(False)

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
        elem_str = '  '.join(f'Z={e["Z"]} amp={e["scale"]:.2f}' for e in elems)
        self._log.append(f'Elements ({len(elems)}): {elem_str}')
        self._log.append(f'Energies:{cfg["experiment"]["E_min_keV"]:.4f}–{cfg["experiment"]["E_max_keV"]:.4f} keV'
                         f'  ({cfg["experiment"]["n_energies"]} points, equidistant Δf′ for elem 1)')
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
