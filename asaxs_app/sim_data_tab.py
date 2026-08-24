"""Synthetic ASAXS data generator tab — parameterizable core-shell model."""

import io
import sys
import traceback
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QSpinBox, QComboBox, QPushButton, QDoubleSpinBox,
    QScrollArea, QSplitter, QTextEdit, QFileDialog, QLineEdit,
    QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

try:
    import xraydb
    _HAS_XRAYDB = True
except ImportError:
    _HAS_XRAYDB = False


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


# ─── Physics (inline — mirrors generate_asaxs_data.py) ────────────────────────

def _elec_density(mass_density, molar_mass, n_electrons):
    N_A = 6.022e23
    return (mass_density / molar_mass) * N_A * n_electrons * 1e-24

def _atom_volume(mass_density, molar_mass):
    N_A = 6.022e23
    return molar_mass / (mass_density * N_A) * 1e24

def _sphere_f0_1d(q, R):
    x = q * R
    return np.where(np.abs(x) < 1e-8, 1.0,
                    3.0 * (np.sin(x) - x * np.cos(x)) / x**3)

def _sphere_f0_2d(q, R_arr):
    x = q[:, None] * R_arr[None, :]
    return np.where(np.abs(x) < 1e-8, 1.0,
                    3.0 * (np.sin(x) - x * np.cos(x)) / x**3)

def _sphere_V(R):
    return (4.0 / 3.0) * np.pi * R**3

def _compute_partials(q, cfg, log):
    core = cfg['core']; shell = cfg['shell']; solvent = cfg['solvent']
    rho_core    = _elec_density(core['mass_density'],    core['molar_mass'],    core['n_electrons'])
    rho_shell   = _elec_density(shell['mass_density'],   shell['molar_mass'],   shell['n_electrons'])
    rho_solvent = _elec_density(solvent['mass_density'], solvent['molar_mass'], solvent['n_electrons'])
    V_atom_res  = _atom_volume(core['mass_density'], core['molar_mass'])
    drho_core   = rho_core  - rho_solvent
    drho_shell  = rho_shell - rho_solvent
    delta       = shell['radius_outer'] - core['radius_mean']
    sigma_rel   = core.get('sigma_rel', 0.0)
    phi         = cfg['experiment']['volume_fraction']

    log(f'  ρ_core={rho_core:.4f}  ρ_shell={rho_shell:.4f}  ρ_solvent={rho_solvent:.4f} e/Å³')
    log(f'  Δρ_core={drho_core:.4f}  Δρ_shell={drho_shell:.4f} e/Å³,  shell thickness={delta:.1f} Å')

    if sigma_rel == 0.0:
        R_c = core['radius_mean']; R_t = shell['radius_outer']
        Vc = _sphere_V(R_c); Vt = _sphere_V(R_t)
        f0c = _sphere_f0_1d(q, R_c); f0t = _sphere_f0_1d(q, R_t)
        F_non = (drho_core - drho_shell) * Vc * f0c + drho_shell * Vt * f0t
        F_res = Vc / V_atom_res * f0c
        I_MM_raw = F_non**2; I_RM_raw = F_non * F_res; I_RR_raw = F_res**2
        V_particle = Vt
        log('  Mode: monodisperse')
    else:
        sigma_ln = np.sqrt(np.log(1 + sigma_rel**2))
        mu_ln    = np.log(core['radius_mean']) - sigma_ln**2 / 2
        sigma_R  = core['radius_mean'] * sigma_rel
        R_vals   = np.linspace(max(1.0, core['radius_mean'] - 6 * sigma_R),
                               core['radius_mean'] + 6 * sigma_R, 600)
        P_R      = (np.exp(-(np.log(R_vals) - mu_ln)**2 / (2 * sigma_ln**2))
                    / (R_vals * sigma_ln * np.sqrt(2 * np.pi)))
        P_R     /= np.trapezoid(P_R, R_vals)
        R_tot    = R_vals + delta
        Vc = _sphere_V(R_vals); Vt = _sphere_V(R_tot)
        f0c = _sphere_f0_2d(q, R_vals); f0t = _sphere_f0_2d(q, R_tot)
        F_non = (drho_core - drho_shell) * Vc[None, :] * f0c + drho_shell * Vt[None, :] * f0t
        F_res = Vc[None, :] / V_atom_res * f0c
        I_MM_raw = np.trapezoid(P_R[None, :] * F_non**2,      R_vals, axis=1)
        I_RM_raw = np.trapezoid(P_R[None, :] * F_non * F_res, R_vals, axis=1)
        I_RR_raw = np.trapezoid(P_R[None, :] * F_res**2,      R_vals, axis=1)
        V_particle = np.mean(_sphere_V(R_vals + delta))
        log(f'  Mode: lognormal polydisperse, σ_rel={sigma_rel:.2f}')

    r_e_cm = 2.818e-13
    N_p_cm = phi / V_particle * 1e24
    scale  = N_p_cm * r_e_cm**2
    return scale * I_MM_raw, scale * I_RM_raw, scale * I_RR_raw

def _select_energies(cfg, log):
    exp = cfg['experiment']
    Z = exp['resonant_Z']; E_min = exp['E_min_keV']; E_max = exp['E_max_keV']
    N = exp['n_energies']
    E_dense  = np.linspace(E_min, E_max, 3000)
    fp_dense = np.array([xraydb.f1_chantler(Z, e * 1000) for e in E_dense])
    sort_idx = np.argsort(fp_dense)
    fp_sorted = fp_dense[sort_idx]; E_sorted = E_dense[sort_idx]
    fp_tgt   = np.linspace(fp_sorted[0], fp_sorted[-1], N)
    energies = sorted(np.interp(fp_tgt, fp_sorted, E_sorted))
    fp_vals  = np.array([xraydb.f1_chantler(Z, e * 1000) for e in energies])
    fpp_vals = np.array([abs(xraydb.f2_chantler(Z, e * 1000)) for e in energies])
    log(f'  {"E (keV)":>10}  {"f′ (e)":>8}  {"f″ (e)":>8}')
    log(f'  {"-"*34}')
    for e, fp, fpp in zip(energies, fp_vals, fpp_vals):
        log(f'  {e:10.4f}  {fp:8.3f}  {fpp:8.3f}')
    return energies, fp_vals, fpp_vals

def _make_noise(I_tot, cfg, rng):
    nc = cfg['noise']
    if nc['model'] == 'poisson':
        I_ref = I_tot[0]; N_pk = nc['N_peak']
        sigma = np.sqrt(np.maximum(I_tot * I_ref / N_pk, 1e-30))
    else:
        sigma = nc.get('relative', 0.005) * I_tot
    sigma = np.maximum(sigma, 1e-8)
    I_obs = np.maximum(I_tot + rng.normal(0, 1, len(I_tot)) * sigma, 1e-8)
    return I_obs, sigma


# ─── Generator thread ─────────────────────────────────────────────────────────

class _GeneratorThread(QThread):
    log_line     = pyqtSignal(str)
    finished_ok  = pyqtSignal(str)   # output directory
    error_signal = pyqtSignal(str)

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self._cfg = cfg

    def _log(self, msg: str):
        self.log_line.emit(msg)

    def run(self):
        try:
            cfg = self._cfg
            out_dir = Path(cfg['output']['directory'])
            name    = cfg['output'].get('name', 'particle')

            qg = cfg['q_grid']
            q  = (np.geomspace if qg.get('spacing', 'log') == 'log' else np.linspace)(
                qg['q_min'], qg['q_max'], qg['n_points'])

            self._log('Computing partial structure factors …')
            I_MM, I_RM, I_RR = _compute_partials(q, cfg, self._log)
            self._log(f'  I_MM(q_min)={I_MM[0]:.3e}  I_RM(q_min)={I_RM[0]:.3e}  I_RR(q_min)={I_RR[0]:.3e} cm⁻¹')

            self._log('\nSelecting energies (equidistant in f′) …')
            energies, fp_vals, fpp_vals = _select_energies(cfg, self._log)

            nc = cfg['noise']
            if nc['model'] == 'poisson':
                N_pk = nc['N_peak']
                self._log(f'\nNoise: Poisson, N_peak={N_pk:.0f}  → σ/I at q_min = {1/np.sqrt(N_pk)*100:.2f}%')
                I_ref = I_MM[0] + 2*fp_vals[0]*I_RM[0] + (fp_vals[0]**2+fpp_vals[0]**2)*I_RR[0]
                if I_ref > 0:
                    I_last = I_MM[-1] + 2*fp_vals[0]*I_RM[-1] + (fp_vals[0]**2+fpp_vals[0]**2)*I_RR[-1]
                    self._log(f'         σ/I at q_max ≈ {np.sqrt(I_ref/max(I_last,1e-40)/N_pk)*100:.1f}%')
            else:
                self._log(f'\nNoise: relative {nc.get("relative",0.005)*100:.2f}%')

            out_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(42)
            self._log(f'\nWriting {len(energies)} files to {out_dir}/ …')

            for i, (E, fp, fpp) in enumerate(zip(energies, fp_vals, fpp_vals)):
                I_tot        = I_MM + 2*fp * I_RM + (fp**2 + fpp**2) * I_RR
                I_obs, sigma = _make_noise(I_tot, cfg, rng)
                core = cfg['core']
                hdr = (
                    f"Synthetic ASAXS data — {cfg.get('system',{}).get('description','')}\n"
                    f"  Core: Z={core['Z']}, R_mean={core['radius_mean']:.1f} A"
                    f", sigma_rel={core.get('sigma_rel',0)*100:.0f}%\n"
                    f"  Shell R_outer={cfg['shell']['radius_outer']:.1f} A\n"
                    f"  phi={cfg['experiment']['volume_fraction']:.2e}"
                    f",  noise={nc['model']}"
                    + (f",  N_peak={nc['N_peak']}" if nc['model']=='poisson'
                       else f",  rel={nc.get('relative',0):.3f}") + "\n"
                    f"  Energy={E:.4f} keV,  f'(Z{cfg['experiment']['resonant_Z']})={fp:.4f} e"
                    f",  f''={fpp:.4f} e\n"
                    f"  Columns: q(1/A)  I(cm-1)  sigma(cm-1)"
                )
                np.savetxt(out_dir / f"{name}_{E:.4f}keV.dat",
                           np.column_stack([q, I_obs, sigma]),
                           header=hdr, fmt="%.6e")
                self._log(f'  [{i+1:2d}/{len(energies)}]  {E:.4f} keV  f′={fp:.3f} e')

            np.savetxt(out_dir / "ground_truth_partials.dat",
                       np.column_stack([q, I_MM, I_RM, I_RR]),
                       header="q(1/A)  I_MM(cm-1)  I_RM(cm-1)  I_RR(cm-1)",
                       fmt="%.6e")
            self._log('\nground_truth_partials.dat saved.')
            self._log(f'\nDone — {len(energies)} datasets in {out_dir}/')
            self.finished_ok.emit(str(out_dir))

        except Exception:
            self.error_signal.emit(traceback.format_exc())


# ─── Helper: material group ───────────────────────────────────────────────────

def _dbl(val, lo=0.0, hi=1e9, decimals=4, step=0.01):
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setValue(val)
    return sb

def _int(val, lo=1, hi=99999):
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(val)
    return sb


class _MaterialGroup(QGroupBox):
    """Reusable group: preset combo + density/molar mass/n_electrons fields."""
    def __init__(self, title, presets: dict, defaults: dict, parent=None):
        super().__init__(title, parent)
        self._presets = presets
        form = QFormLayout(self)

        self._combo = QComboBox()
        self._combo.addItem('— custom —')
        for name in presets:
            self._combo.addItem(name)
        btn_apply = QPushButton('Apply')
        btn_apply.setFixedWidth(60)
        row = QHBoxLayout()
        row.addWidget(self._combo); row.addWidget(btn_apply)
        form.addRow('Preset:', row)

        self._spin_rho = _dbl(defaults['mass_density'], lo=0.0, hi=1e5, decimals=4, step=0.01)
        self._spin_M   = _dbl(defaults['molar_mass'],   lo=0.0, hi=1e5, decimals=4, step=0.01)
        self._spin_ne  = _int(defaults['n_electrons'], lo=1, hi=9999)
        form.addRow('ρ (g/cm³):', self._spin_rho)
        form.addRow('M (g/mol):',  self._spin_M)
        form.addRow('e⁻/formula:', self._spin_ne)

        btn_apply.clicked.connect(self._apply)

    def _apply(self):
        name = self._combo.currentText()
        if name in self._presets:
            p = self._presets[name]
            self._spin_rho.setValue(p['mass_density'])
            self._spin_M.setValue(p['molar_mass'])
            self._spin_ne.setValue(p['n_electrons'])

    def values(self):
        return {
            'mass_density': self._spin_rho.value(),
            'molar_mass':   self._spin_M.value(),
            'n_electrons':  self._spin_ne.value(),
        }


# ─── Tab ──────────────────────────────────────────────────────────────────────

class SimDataTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left: scrollable parameter form ───────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form_w = QWidget()
        form_lay = QVBoxLayout(form_w)
        form_lay.setSpacing(8)
        scroll.setWidget(form_w)
        splitter.addWidget(scroll)

        # ── Core ──────────────────────────────────────────────────────────────
        core_grp = QGroupBox('Core material')
        core_form = QFormLayout(core_grp)

        self._core_preset = QComboBox()
        self._core_preset.addItem('— custom —')
        for nm in _ELEMENT_PRESETS: self._core_preset.addItem(nm)
        btn_core = QPushButton('Apply'); btn_core.setFixedWidth(60)
        r = QHBoxLayout(); r.addWidget(self._core_preset); r.addWidget(btn_core)
        core_form.addRow('Element preset:', r)

        self._core_Z    = _int(79, lo=1, hi=118)
        self._core_rho  = _dbl(19.32, decimals=4)
        self._core_M    = _dbl(196.97, decimals=4)
        self._core_ne   = _int(79)
        self._core_R    = _dbl(55.0, lo=0.1, hi=1e5, decimals=2, step=1.0)
        self._core_sig  = _dbl(0.20, lo=0.0, hi=2.0, decimals=3, step=0.01)
        self._lbl_core_sig = QLabel('(0 = monodisperse)')
        core_form.addRow('Atomic number Z:',  self._core_Z)
        core_form.addRow('ρ (g/cm³):',        self._core_rho)
        core_form.addRow('M (g/mol):',         self._core_M)
        core_form.addRow('e⁻/atom:',           self._core_ne)
        core_form.addRow('R_mean (Å):',        self._core_R)
        sig_row = QHBoxLayout()
        sig_row.addWidget(self._core_sig)
        sig_row.addWidget(self._lbl_core_sig)
        core_form.addRow('σ_rel (lognormal):', sig_row)
        form_lay.addWidget(core_grp)

        btn_core.clicked.connect(self._apply_core_preset)
        self._core_sig.valueChanged.connect(
            lambda v: self._lbl_core_sig.setText('(0 = monodisperse)' if v == 0 else f'= {v*100:.0f}%'))

        # ── Shell ─────────────────────────────────────────────────────────────
        shell_grp = QGroupBox('Shell material')
        shell_form = QFormLayout(shell_grp)
        self._shell_preset = QComboBox()
        self._shell_preset.addItem('— custom —')
        for nm in _MATERIAL_PRESETS: self._shell_preset.addItem(nm)
        btn_shell = QPushButton('Apply'); btn_shell.setFixedWidth(60)
        r2 = QHBoxLayout(); r2.addWidget(self._shell_preset); r2.addWidget(btn_shell)
        shell_form.addRow('Material preset:', r2)
        self._shell_rho = _dbl(2.196, decimals=4)
        self._shell_M   = _dbl(60.08, decimals=4)
        self._shell_ne  = _int(30)
        self._shell_Rt  = _dbl(220.0, lo=0.1, hi=1e5, decimals=2, step=1.0)
        shell_form.addRow('ρ (g/cm³):', self._shell_rho)
        shell_form.addRow('M (g/mol):',  self._shell_M)
        shell_form.addRow('e⁻/formula:', self._shell_ne)
        shell_form.addRow('R_outer (Å):', self._shell_Rt)
        self._lbl_shell_thick = QLabel()
        shell_form.addRow('Shell thickness:', self._lbl_shell_thick)
        form_lay.addWidget(shell_grp)
        btn_shell.clicked.connect(self._apply_shell_preset)
        self._core_R.valueChanged.connect(self._update_thickness)
        self._shell_Rt.valueChanged.connect(self._update_thickness)
        self._update_thickness()

        # ── Solvent ───────────────────────────────────────────────────────────
        solv_grp = QGroupBox('Solvent')
        solv_form = QFormLayout(solv_grp)
        self._solv_preset = QComboBox()
        self._solv_preset.addItem('— custom —')
        for nm in _MATERIAL_PRESETS: self._solv_preset.addItem(nm)
        btn_solv = QPushButton('Apply'); btn_solv.setFixedWidth(60)
        r3 = QHBoxLayout(); r3.addWidget(self._solv_preset); r3.addWidget(btn_solv)
        solv_form.addRow('Material preset:', r3)
        self._solv_rho = _dbl(1.000, decimals=4)
        self._solv_M   = _dbl(18.015, decimals=4)
        self._solv_ne  = _int(10)
        solv_form.addRow('ρ (g/cm³):', self._solv_rho)
        solv_form.addRow('M (g/mol):',  self._solv_M)
        solv_form.addRow('e⁻/formula:', self._solv_ne)
        form_lay.addWidget(solv_grp)
        btn_solv.clicked.connect(self._apply_solv_preset)

        # ── Experiment ────────────────────────────────────────────────────────
        exp_grp = QGroupBox('Experiment')
        exp_form = QFormLayout(exp_grp)
        self._exp_phi     = _dbl(1e-4, lo=1e-10, hi=1.0, decimals=8, step=1e-5)
        self._exp_phi.setDecimals(6)
        self._exp_Z       = _int(79, lo=1, hi=118)
        self._exp_Emin    = _dbl(10.919, lo=0.1, hi=1000.0, decimals=4, step=0.01)
        self._exp_Emax    = _dbl(11.919, lo=0.1, hi=1000.0, decimals=4, step=0.01)
        self._exp_N       = _int(20, lo=2, hi=200)
        self._lbl_edge    = QLabel()
        exp_form.addRow('Volume fraction φ:', self._exp_phi)
        exp_form.addRow('Resonant element Z:', self._exp_Z)
        exp_form.addRow('Edge info:',           self._lbl_edge)
        exp_form.addRow('E_min (keV):',         self._exp_Emin)
        exp_form.addRow('E_max (keV):',         self._exp_Emax)
        exp_form.addRow('N energies:',          self._exp_N)
        form_lay.addWidget(exp_grp)
        self._exp_Z.valueChanged.connect(self._update_edge_info)
        self._core_Z.valueChanged.connect(lambda v: self._exp_Z.setValue(v))
        self._update_edge_info()

        # ── q grid ────────────────────────────────────────────────────────────
        q_grp = QGroupBox('q grid')
        q_form = QFormLayout(q_grp)
        self._q_min  = _dbl(0.003, lo=1e-5, hi=10.0, decimals=5, step=0.001)
        self._q_max  = _dbl(0.145, lo=1e-5, hi=10.0, decimals=5, step=0.001)
        self._q_n    = _int(500, lo=10, hi=10000)
        self._q_log  = QComboBox(); self._q_log.addItems(['Logarithmic', 'Linear'])
        q_form.addRow('q_min (Å⁻¹):', self._q_min)
        q_form.addRow('q_max (Å⁻¹):', self._q_max)
        q_form.addRow('N points:',     self._q_n)
        q_form.addRow('Spacing:',      self._q_log)
        form_lay.addWidget(q_grp)

        # ── Noise ─────────────────────────────────────────────────────────────
        noise_grp = QGroupBox('Noise model')
        noise_form = QFormLayout(noise_grp)
        self._rb_poisson  = QRadioButton('Poissonian  σ(q) = √(I(q)·I(q_min)/N_peak)')
        self._rb_relative = QRadioButton('Relative     σ(q) = p · I(q)')
        self._rb_poisson.setChecked(True)
        self._noise_grp = QButtonGroup(); self._noise_grp.addButton(self._rb_poisson)
        self._noise_grp.addButton(self._rb_relative)
        self._n_peak = _int(10000, lo=1, hi=10000000)
        self._n_rel  = _dbl(0.005, lo=1e-6, hi=1.0, decimals=4, step=0.001)
        noise_form.addRow(self._rb_poisson)
        noise_form.addRow('N_peak:', self._n_peak)
        noise_form.addRow(self._rb_relative)
        noise_form.addRow('Relative σ/I:', self._n_rel)
        form_lay.addWidget(noise_grp)

        # ── Output ────────────────────────────────────────────────────────────
        out_grp = QGroupBox('Output')
        out_form = QFormLayout(out_grp)
        self._out_dir  = QLineEdit(str(Path.home() / 'asaxs_sim_data'))
        self._out_name = QLineEdit('particle')
        btn_browse = QPushButton('Browse…'); btn_browse.setFixedWidth(70)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._out_dir); dir_row.addWidget(btn_browse)
        out_form.addRow('Directory:', dir_row)
        out_form.addRow('File prefix:', self._out_name)
        form_lay.addWidget(out_grp)
        btn_browse.clicked.connect(self._browse_dir)

        # ── Generate button ───────────────────────────────────────────────────
        self._btn_gen = QPushButton('Generate data')
        self._btn_gen.setFixedHeight(36)
        self._btn_gen.setStyleSheet('font-weight: bold; font-size: 13px;')
        form_lay.addWidget(self._btn_gen)
        form_lay.addStretch()
        self._btn_gen.clicked.connect(self._generate)

        # ── Right: log ────────────────────────────────────────────────────────
        log_w = QWidget()
        log_lay = QVBoxLayout(log_w)
        log_lay.addWidget(QLabel('Output log'))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        mono = QFont('Courier New', 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._log.setFont(mono)
        self._btn_clear_log = QPushButton('Clear log')
        self._btn_clear_log.setFixedWidth(90)
        log_lay.addWidget(self._log)
        log_lay.addWidget(self._btn_clear_log)
        splitter.addWidget(log_w)
        self._btn_clear_log.clicked.connect(self._log.clear)
        splitter.setSizes([420, 560])

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
            self._shell_rho.setValue(p['mass_density'])
            self._shell_M.setValue(p['molar_mass'])
            self._shell_ne.setValue(p['n_electrons'])

    def _apply_solv_preset(self):
        nm = self._solv_preset.currentText()
        if nm in _MATERIAL_PRESETS:
            p = _MATERIAL_PRESETS[nm]
            self._solv_rho.setValue(p['mass_density'])
            self._solv_M.setValue(p['molar_mass'])
            self._solv_ne.setValue(p['n_electrons'])

    def _update_thickness(self):
        t = self._shell_Rt.value() - self._core_R.value()
        self._lbl_shell_thick.setText(f'{t:.1f} Å')

    def _update_edge_info(self):
        if not _HAS_XRAYDB:
            self._lbl_edge.setText('xraydb not installed')
            return
        Z = self._exp_Z.value()
        try:
            edges = xraydb.xray_edges(Z)
            parts = []
            for shell in ['K', 'L3', 'L2', 'L1', 'M5']:
                if shell in edges:
                    e_keV = edges[shell].energy / 1000.0
                    parts.append(f'{shell}: {e_keV:.3f} keV')
                    if len(parts) == 3:
                        break
            self._lbl_edge.setText('  '.join(parts))
        except Exception:
            self._lbl_edge.setText('')

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, 'Output directory',
                                             self._out_dir.text())
        if d:
            self._out_dir.setText(d)

    # ─── Build config dict from UI ────────────────────────────────────────────

    def _build_config(self) -> dict:
        out_path = Path(self._out_dir.text()) / self._out_name.text()
        return {
            'system': {'description': f'Z={self._core_Z.value()} core, R_mean={self._core_R.value()} A'},
            'core': {
                'Z':            self._core_Z.value(),
                'mass_density': self._core_rho.value(),
                'molar_mass':   self._core_M.value(),
                'n_electrons':  self._core_ne.value(),
                'radius_mean':  self._core_R.value(),
                'sigma_rel':    self._core_sig.value(),
            },
            'shell': {
                'mass_density': self._shell_rho.value(),
                'molar_mass':   self._shell_M.value(),
                'n_electrons':  self._shell_ne.value(),
                'radius_outer': self._shell_Rt.value(),
            },
            'solvent': {
                'mass_density': self._solv_rho.value(),
                'molar_mass':   self._solv_M.value(),
                'n_electrons':  self._solv_ne.value(),
            },
            'experiment': {
                'volume_fraction': self._exp_phi.value(),
                'resonant_Z':      self._exp_Z.value(),
                'E_min_keV':       self._exp_Emin.value(),
                'E_max_keV':       self._exp_Emax.value(),
                'n_energies':      self._exp_N.value(),
            },
            'q_grid': {
                'q_min':   self._q_min.value(),
                'q_max':   self._q_max.value(),
                'n_points': self._q_n.value(),
                'spacing': 'log' if self._q_log.currentIndex() == 0 else 'linear',
            },
            'noise': {
                'model':    'poisson' if self._rb_poisson.isChecked() else 'relative',
                'N_peak':   self._n_peak.value(),
                'relative': self._n_rel.value(),
            },
            'output': {
                'directory': str(out_path),
                'name':      self._out_name.text(),
            },
        }

    # ─── Generation ──────────────────────────────────────────────────────────

    def _generate(self):
        if not _HAS_XRAYDB:
            self._log.append('<b>Error: xraydb is not installed.</b>  Run: pip install xraydb')
            return
        if self._thread and self._thread.isRunning():
            return

        cfg = self._build_config()
        self._log.clear()
        self._log.append(f'<b>Generating ASAXS data …</b>')
        self._log.append(f'Core:    Z={cfg["core"]["Z"]}, R_mean={cfg["core"]["radius_mean"]:.1f} Å'
                         f', σ_rel={cfg["core"]["sigma_rel"]*100:.0f}%')
        self._log.append(f'Shell:   R_outer={cfg["shell"]["radius_outer"]:.1f} Å')
        self._log.append(f'Solvent: ρ={cfg["solvent"]["mass_density"]:.3f} g/cm³')
        self._log.append(f'Energy:  {cfg["experiment"]["E_min_keV"]:.4f}–{cfg["experiment"]["E_max_keV"]:.4f} keV'
                         f',  {cfg["experiment"]["n_energies"]} points')
        self._log.append(f'Output:  {cfg["output"]["directory"]}/')
        self._log.append('')

        self._btn_gen.setEnabled(False)
        self._btn_gen.setText('Generating …')

        self._thread = _GeneratorThread(cfg, self)
        self._thread.log_line.connect(self._log.append)
        self._thread.finished_ok.connect(self._on_done)
        self._thread.error_signal.connect(self._on_error)
        self._thread.start()

    def _on_done(self, out_dir: str):
        self._log.append(f'\n<b>✓ Complete.</b>  Files in: {out_dir}')
        self._btn_gen.setEnabled(True)
        self._btn_gen.setText('Generate data')

    def _on_error(self, tb: str):
        self._log.append(f'\n<b>Error:</b>\n{tb}')
        self._btn_gen.setEnabled(True)
        self._btn_gen.setText('Generate data')
