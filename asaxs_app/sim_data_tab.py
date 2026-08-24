"""Synthetic ASAXS data generator tab — parameterizable core-shell model."""

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

def _compute_partials(q, cfg, log):
    core = cfg['core']; shell = cfg['shell']; solv = cfg['solvent']
    rho_c = _elec_density(core['mass_density'], core['molar_mass'], core['n_electrons'])
    rho_s = _elec_density(shell['mass_density'], shell['molar_mass'], shell['n_electrons'])
    rho_v = _elec_density(solv['mass_density'], solv['molar_mass'], solv['n_electrons'])
    V_at  = _atom_volume(core['mass_density'], core['molar_mass'])
    dr_c  = rho_c - rho_v;  dr_s = rho_s - rho_v
    delta = shell['radius_outer'] - core['radius_mean']
    sig   = core.get('sigma_rel', 0.0)
    phi   = cfg['experiment']['volume_fraction']
    log(f'  ρ_core={rho_c:.4f}  ρ_shell={rho_s:.4f}  ρ_solv={rho_v:.4f} e/Å³')

    if sig == 0.0:
        Rc = core['radius_mean']; Rt = shell['radius_outer']
        Vc = _sphere_V(Rc); Vt = _sphere_V(Rt)
        f0c = _sphere_f0_1d(q, Rc); f0t = _sphere_f0_1d(q, Rt)
        Fn = (dr_c - dr_s) * Vc * f0c + dr_s * Vt * f0t
        Fr = Vc / V_at * f0c
        I_MM_r, I_RM_r, I_RR_r = Fn**2, Fn*Fr, Fr**2
        Vp = Vt;  log('  Mode: monodisperse')
    else:
        sln = np.sqrt(np.log(1 + sig**2))
        mu  = np.log(core['radius_mean']) - sln**2 / 2
        sR  = core['radius_mean'] * sig
        Rv  = np.linspace(max(1.0, core['radius_mean'] - 6*sR), core['radius_mean'] + 6*sR, 600)
        PR  = np.exp(-(np.log(Rv) - mu)**2 / (2*sln**2)) / (Rv * sln * np.sqrt(2*np.pi))
        PR /= np.trapezoid(PR, Rv)
        Rt  = Rv + delta
        Vc  = _sphere_V(Rv); Vt = _sphere_V(Rt)
        f0c = _sphere_f0_2d(q, Rv); f0t = _sphere_f0_2d(q, Rt)
        Fn  = (dr_c - dr_s) * Vc[None,:] * f0c + dr_s * Vt[None,:] * f0t
        Fr  = Vc[None,:] / V_at * f0c
        I_MM_r = np.trapezoid(PR[None,:] * Fn**2,    Rv, axis=1)
        I_RM_r = np.trapezoid(PR[None,:] * Fn * Fr,  Rv, axis=1)
        I_RR_r = np.trapezoid(PR[None,:] * Fr**2,    Rv, axis=1)
        Vp = np.mean(_sphere_V(Rv + delta))
        log(f'  Mode: lognormal σ_rel={sig:.2f}')

    sc = phi / Vp * 1e24 * (2.818e-13)**2
    return sc * I_MM_r, sc * I_RM_r, sc * I_RR_r

def _select_energies(cfg, log=None):
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
            cfg = self._cfg
            out_dir = Path(cfg['output']['directory'])
            name    = cfg['output'].get('name', 'particle')
            qg      = cfg['q_grid']
            q       = (np.geomspace if qg.get('spacing','log') == 'log' else np.linspace)(
                qg['q_min'], qg['q_max'], qg['n_points'])

            self.log_line.emit('Computing partial structure factors …')
            I_MM, I_RM, I_RR = _compute_partials(q, cfg, self.log_line.emit)
            self.log_line.emit(f'  I_MM(q_min)={I_MM[0]:.3e}  I_RM(q_min)={I_RM[0]:.3e}  I_RR(q_min)={I_RR[0]:.3e} cm⁻¹')

            self.log_line.emit('\nSelecting energies (equidistant Δf′ across range) …')
            energies, fp_vals, fpp_vals = _select_energies(cfg, self.log_line.emit)

            nc = cfg['noise']
            if nc['model'] == 'poisson':
                self.log_line.emit(f'\nNoise: Poisson, N_peak={nc["N_peak"]:.0f}  → σ/I(q_min) = {1/np.sqrt(nc["N_peak"])*100:.2f}%')
            else:
                self.log_line.emit(f'\nNoise: relative {nc.get("relative",0.005)*100:.2f}%')

            out_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(42)
            self.log_line.emit(f'\nWriting {len(energies)} files to {out_dir}/ …')

            I_obs_list = []
            for i, (E, fp, fpp) in enumerate(zip(energies, fp_vals, fpp_vals)):
                I_tot        = I_MM + 2*fp*I_RM + (fp**2 + fpp**2)*I_RR
                I_obs, sigma = _make_noise(I_tot, cfg, rng)
                I_obs_list.append(I_obs)
                core = cfg['core']
                hdr = (
                    f"Synthetic ASAXS — Z={core['Z']} core, R_mean={core['radius_mean']:.1f} A"
                    f", sig_rel={core.get('sigma_rel',0)*100:.0f}%\n"
                    f"  Shell R_outer={cfg['shell']['radius_outer']:.1f} A"
                    f",  phi={cfg['experiment']['volume_fraction']:.2e}"
                    f",  noise={nc['model']}"
                    + (f",  N_peak={nc['N_peak']}" if nc['model']=='poisson'
                       else f",  rel={nc.get('relative',0):.3f}") + "\n"
                    f"  Energy={E:.4f} keV,  f'={fp:.4f} e,  f''={fpp:.4f} e\n"
                    f"  Columns: q(1/A)  I(cm-1)  sigma(cm-1)"
                )
                np.savetxt(out_dir / f"{name}_{E:.4f}keV.dat",
                           np.column_stack([q, I_obs, sigma]), header=hdr, fmt="%.6e")
                self.log_line.emit(f'  [{i+1:2d}/{len(energies)}]  {E:.4f} keV  f′={fp:.3f} e  f″={fpp:.3f} e')

            np.savetxt(out_dir / "ground_truth_partials.dat",
                       np.column_stack([q, I_MM, I_RM, I_RR]),
                       header="q(1/A)  I_MM(cm-1)  I_RM(cm-1)  I_RR(cm-1)", fmt="%.6e")
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
        self._shell_rho = _dbl(2.196, decimals=4)
        self._shell_M   = _dbl(60.08, decimals=4)
        self._shell_ne  = _int(30)
        self._shell_Rt  = _dbl(220.0, lo=0.1, hi=1e5, decimals=2, step=1.0)
        self._lbl_thick = QLabel()
        shell_form.addRow('ρ (g/cm³):',    self._shell_rho)
        shell_form.addRow('M (g/mol):',     self._shell_M)
        shell_form.addRow('e⁻/formula:',    self._shell_ne)
        shell_form.addRow('R_outer (Å):',   self._shell_Rt)
        shell_form.addRow('Shell thickness:', self._lbl_thick)
        form_lay.addWidget(shell_grp)
        btn_shell.clicked.connect(self._apply_shell_preset)
        self._core_R.valueChanged.connect(self._update_thickness)
        self._shell_Rt.valueChanged.connect(self._update_thickness)
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

        self._exp_Z    = _int(79, lo=1, hi=118)
        self._lbl_edge = QLabel()
        self._exp_Emin = _dbl(10.919, lo=0.1, hi=1000.0, decimals=4, step=0.01)
        self._exp_Emax = _dbl(11.919, lo=0.1, hi=1000.0, decimals=4, step=0.01)
        self._exp_N    = _int(20, lo=2, hi=200)

        en_form.addRow('Resonant element Z:', self._exp_Z)
        en_form.addRow('Edge energies:',       self._lbl_edge)
        en_form.addRow('E_min (keV):',         self._exp_Emin)
        en_form.addRow('E_max (keV):',         self._exp_Emax)
        en_form.addRow('N energies:',          self._exp_N)

        note = QLabel(
            '<i>Energies are spaced for equal Δf′ steps across [E_min, E_max].<br>'
            'This maximises the variation in f′ between measurements,<br>'
            'which is required for accurate ASAXS decomposition.</i>')
        note.setWordWrap(True)
        en_form.addRow(note)

        self._btn_preview_E = QPushButton('Preview selected energies')
        en_form.addRow(self._btn_preview_E)
        form_lay.addWidget(en_grp)

        self._exp_Z.valueChanged.connect(self._update_edge_info)
        self._core_Z.valueChanged.connect(lambda v: self._exp_Z.setValue(v))
        self._btn_preview_E.clicked.connect(self._preview_energies)
        self._update_edge_info()

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
        self._lbl_thick.setText(f'{self._shell_Rt.value() - self._core_R.value():.1f} Å')

    def _update_edge_info(self):
        if not _HAS_XRAYDB:
            self._lbl_edge.setText('xraydb not installed'); return
        Z = self._exp_Z.value()
        try:
            edges = xraydb.xray_edges(Z)
            parts = []
            for sh in ['K', 'L3', 'L2', 'L1', 'M5']:
                if sh in edges:
                    parts.append(f'{sh}: {edges[sh].energy/1000:.3f} keV')
                    if len(parts) == 3: break
            self._lbl_edge.setText('  '.join(parts))
        except Exception:
            self._lbl_edge.setText('')

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, 'Output directory', self._out_dir.text())
        if d: self._out_dir.setText(d)

    def _preview_energies(self):
        if not _HAS_XRAYDB:
            self._log.append('xraydb not installed'); return
        cfg = self._build_config()
        self._log.append('<b>Energy preview:</b>')
        self._log.append(f'  Z={cfg["experiment"]["resonant_Z"]},  '
                         f'E: {cfg["experiment"]["E_min_keV"]:.4f}–{cfg["experiment"]["E_max_keV"]:.4f} keV,  '
                         f'N={cfg["experiment"]["n_energies"]}')
        self._log.append('  Energies are chosen to be equidistant in f′(E) — this maximises')
        self._log.append('  the matrix condition number for the linear decomposition.')
        self._log.append(f'  {"E (keV)":>10}  {"f′ (e)":>8}  {"f″ (e)":>8}')
        self._log.append(f'  {"-"*34}')
        try:
            Es, fp, fpp = _select_energies(cfg)
            for e, f, ff in zip(Es, fp, fpp):
                self._log.append(f'  {e:10.4f}  {f:8.3f}  {ff:8.3f}')
            self._log.append(f'  f′ range: {fp.min():.3f} → {fp.max():.3f} e  (Δf′={fp.max()-fp.min():.3f} e)')
        except Exception as ex:
            self._log.append(f'  Error: {ex}')

    # ─── Build config ─────────────────────────────────────────────────────────

    def _build_config(self) -> dict:
        out_path = Path(self._out_dir.text()) / self._out_name.text()
        return {
            'system': {'description': f'Z={self._core_Z.value()} core R_mean={self._core_R.value():.1f}A'},
            'core':   {'Z': self._core_Z.value(), 'mass_density': self._core_rho.value(),
                       'molar_mass': self._core_M.value(), 'n_electrons': self._core_ne.value(),
                       'radius_mean': self._core_R.value(), 'sigma_rel': self._core_sig.value()},
            'shell':  {'mass_density': self._shell_rho.value(), 'molar_mass': self._shell_M.value(),
                       'n_electrons': self._shell_ne.value(), 'radius_outer': self._shell_Rt.value()},
            'solvent':{'mass_density': self._solv_rho.value(), 'molar_mass': self._solv_M.value(),
                       'n_electrons': self._solv_ne.value()},
            'experiment': {'volume_fraction': self._exp_phi.value(),
                           'resonant_Z': self._exp_Z.value(),
                           'E_min_keV': self._exp_Emin.value(), 'E_max_keV': self._exp_Emax.value(),
                           'n_energies': self._exp_N.value()},
            'q_grid': {'q_min': self._q_min.value(), 'q_max': self._q_max.value(),
                       'n_points': self._q_n.value(),
                       'spacing': 'log' if self._q_log.currentIndex() == 0 else 'linear'},
            'noise':  {'model': 'poisson' if self._rb_poisson.isChecked() else 'relative',
                       'N_peak': self._n_peak.value(), 'relative': self._n_rel.value()},
            'output': {'directory': str(out_path), 'name': self._out_name.text()},
        }

    # ─── Generation ──────────────────────────────────────────────────────────

    def _generate(self):
        if not _HAS_XRAYDB:
            self._log.append('<b>Error:</b> xraydb not installed — run: pip install xraydb'); return
        if self._thread and self._thread.isRunning():
            return
        cfg = self._build_config()
        self._log.clear()
        self._log.append('<b>Generating ASAXS data …</b>')
        self._log.append(f'Core:    Z={cfg["core"]["Z"]}, R_mean={cfg["core"]["radius_mean"]:.1f} Å'
                         f', σ_rel={cfg["core"]["sigma_rel"]*100:.0f}%')
        self._log.append(f'Shell:   R_outer={cfg["shell"]["radius_outer"]:.1f} Å')
        self._log.append(f'Energies:{cfg["experiment"]["E_min_keV"]:.4f}–{cfg["experiment"]["E_max_keV"]:.4f} keV'
                         f'  ({cfg["experiment"]["n_energies"]} points, equidistant Δf′)')
        self._log.append(f'Output:  {cfg["output"]["directory"]}/')
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
