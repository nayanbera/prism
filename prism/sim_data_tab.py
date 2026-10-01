"""Sim / Fit tab — N-layer multi-shell sphere model.

Physics lives in prism.core.multilayer_sphere (shared with XModFit2).
"""

import traceback
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QSpinBox, QComboBox, QPushButton, QDoubleSpinBox,
    QScrollArea, QSplitter, QTextEdit, QFileDialog, QLineEdit,
    QRadioButton, QButtonGroup, QCheckBox, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import pyqtgraph as pg

from .core.crosshair import add_crosshair
from .core.multilayer_sphere import (
    compute_multilayer as _compute_multilayer,
    elec_density as _elec_density,
    atom_volume as _atom_volume,
    sphere_V as _sphere_V,
    sphere_f0_2d as _sphere_f0_2d,
    gh_nodes as _gh_nodes,
    layer_grid as _layer_grid,
    rho_profile as _rho_profile,
)

try:
    import xraydb
    _HAS_XRAYDB = True
except ImportError:
    _HAS_XRAYDB = False

# ── Layer-table column indices ─────────────────────────────────────────────────
_C_FIT, _C_NAME, _C_FML, _C_RHO, _C_M, _C_NE, _C_T, _C_SIG = range(8)
_COL_HDRS = ['Fit', 'Name', 'Formula', 'ρ\n(g/cm³)', 'M\n(g/mol)',
             'n_e', 't / R\n(Å)', 'σ\n(Å)']
_COL_W    = [36, 78, 78, 84, 78, 52, 84, 78]

_SHAPES     = ['Sphere', 'Cylinder', 'Disk', 'Parallelepiped']
_DIST_TYPES = ['Monodisperse', 'Gaussian', 'Log-Normal']

# ── Energy colour (blue → red) ─────────────────────────────────────────────────
def _energy_color(t: float) -> tuple:
    stops = [(0.0,(0,0,255)),(0.33,(0,200,255)),(0.5,(0,220,0)),
             (0.67,(255,220,0)),(1.0,(255,0,0))]
    for i in range(len(stops)-1):
        t0,c0=stops[i]; t1,c1=stops[i+1]
        if t0<=t<=t1:
            f=(t-t0)/(t1-t0)
            return tuple(int(c0[k]+f*(c1[k]-c0[k])) for k in range(3))
    return (255,0,0)

# ── Material presets ───────────────────────────────────────────────────────────
_ELEMENT_PRESETS = {
    'Au (gold)':      {'Z':79,'mass_density':19.32,'molar_mass':196.97,'n_electrons':79},
    'Pt (platinum)':  {'Z':78,'mass_density':21.45,'molar_mass':195.08,'n_electrons':78},
    'Pd (palladium)': {'Z':46,'mass_density':12.02,'molar_mass':106.42,'n_electrons':46},
    'Ag (silver)':    {'Z':47,'mass_density':10.49,'molar_mass':107.87,'n_electrons':47},
    'Fe (iron)':      {'Z':26,'mass_density': 7.87,'molar_mass': 55.85,'n_electrons':26},
    'Cu (copper)':    {'Z':29,'mass_density': 8.96,'molar_mass': 63.55,'n_electrons':29},
    'Ni (nickel)':    {'Z':28,'mass_density': 8.91,'molar_mass': 58.69,'n_electrons':28},
    'Se (selenium)':  {'Z':34,'mass_density': 4.81,'molar_mass': 78.97,'n_electrons':34},
}
_MATERIAL_PRESETS = {
    'SiO₂ (silica)':     {'mass_density':2.196,'molar_mass': 60.08,'n_electrons':30},
    'H₂O (water)':       {'mass_density':1.000,'molar_mass': 18.015,'n_electrons':10},
    'D₂O (heavy water)': {'mass_density':1.105,'molar_mass': 20.03,'n_electrons':10},
    'Polystyrene':        {'mass_density':1.050,'molar_mass':104.15,'n_electrons':56},
    'PMMA':               {'mass_density':1.190,'molar_mass':100.12,'n_electrons':54},
    'Toluene':            {'mass_density':0.867,'molar_mass': 92.14,'n_electrons':50},
    'Ethanol':            {'mass_density':0.789,'molar_mass': 46.07,'n_electrons':26},
    'Air':                {'mass_density':0.00129,'molar_mass':28.97,'n_electrons':15},
}

# ── Molecular formula parser ───────────────────────────────────────────────────
def _parse_formula(formula: str) -> dict:
    import re
    def _h(s):
        res={}; i=0
        while i<len(s):
            if s[i]=='(':
                j=i+1; d=1
                while j<len(s) and d:
                    if s[j]=='(': d+=1
                    elif s[j]==')': d-=1
                    j+=1
                inner=_h(s[i+1:j-1])
                m=re.match(r'\d+',s[j:]); mul=int(m.group()) if m else 1
                j+=len(m.group()) if m else 0
                for sym,cnt in inner.items(): res[sym]=res.get(sym,0)+cnt*mul
                i=j
            else:
                m=re.match(r'([A-Z][a-z]?)(\d*)',s[i:])
                if not m: raise ValueError(f'Bad formula near {s[i:]!r}')
                sym,num=m.group(1),m.group(2)
                res[sym]=res.get(sym,0)+(int(num) if num else 1); i+=len(m.group())
        return res
    return _h(formula.strip())

def _formula_to_Mne(formula: str) -> tuple:
    if not _HAS_XRAYDB:
        raise RuntimeError('xraydb not installed')
    counts=_parse_formula(formula); M=0.0; ne=0
    for sym,cnt in counts.items():
        ne+=xraydb.atomic_number(sym)*cnt; M+=xraydb.atomic_mass(sym)*cnt
    return M, ne

def _element_Z(symbol: str):
    if not _HAS_XRAYDB: return None
    try: return xraydb.atomic_number(symbol)
    except Exception: return None



# ── Energy selection ───────────────────────────────────────────────────────────
def _select_energies_multi(cfg, log=None):
    """Select energies for all active resonant elements and compute f′/f″."""
    elements = cfg.get('resonant_elements', [])
    if not elements:
        exp = cfg['experiment']
        elements = [{'Z': exp.get('resonant_Z', 79), 'layer_idx': 0,
                     'E_min_keV': exp['E_min_keV'],
                     'E_max_keV': exp['E_max_keV'],
                     'n_energies': exp['n_energies']}]
    all_Es = []
    for elem in elements:
        Z=elem['Z']; E0=elem.get('E_min_keV',10.0); E1=elem.get('E_max_keV',11.0)
        N=int(elem.get('n_energies',20))
        Ed=np.linspace(E0,E1,3000)
        fpd=np.array([xraydb.f1_chantler(Z,e*1000) for e in Ed])
        idx=np.argsort(fpd)
        tgt=np.linspace(fpd[idx[0]],fpd[idx[-1]],N)
        all_Es.extend(sorted(np.interp(tgt,fpd[idx],Ed[idx])))
    all_Es=sorted(set(round(e,6) for e in all_Es))
    Es=np.array(all_Es)
    elem_results=[]
    for elem in elements:
        Z=elem['Z']
        fp =np.array([xraydb.f1_chantler(Z,e*1000) for e in Es])
        fpp=np.array([abs(xraydb.f2_chantler(Z,e*1000)) for e in Es])
        elem_results.append({'Z':Z,'fp':fp,'fpp':fpp,
                             'layer_idx': int(elem.get('layer_idx',0))})
    if log:
        hdrs=['E (keV)']+[f"f'_{i+1}" for i in range(len(elements))]+\
             [f'f"_{i+1}' for i in range(len(elements))]
        log('  '+'  '.join(f'{h:>9}' for h in hdrs))
        for k,e in enumerate(Es):
            row=f'  {e:10.4f}'+''.join(f'  {elem_results[i]["fp"][k]:9.3f}' for i in range(len(elements)))\
                               +''.join(f'  {elem_results[i]["fpp"][k]:9.3f}' for i in range(len(elements)))
            log(row)
    return Es, elem_results


def _make_noise(I_tot, cfg, rng):
    nc=cfg['noise']
    if nc['model']=='poisson':
        sigma=np.sqrt(np.maximum(I_tot*I_tot[0]/nc['N_peak'],1e-30))
    else:
        sigma=nc.get('relative',0.005)*I_tot
    sigma=np.maximum(sigma,1e-8)
    return np.maximum(I_tot+rng.normal(0,1,len(I_tot))*sigma,1e-8), sigma


# ── Generator thread ───────────────────────────────────────────────────────────
class _GeneratorThread(QThread):
    log_line     = pyqtSignal(str)
    data_ready   = pyqtSignal(object, object, object)
    finished_ok  = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent); self._cfg = cfg

    def run(self):
        try:
            cfg     = self._cfg
            shape   = cfg.get('shape','Sphere')
            if shape != 'Sphere':
                self.error_signal.emit(f'Shape "{shape}" not yet implemented.'); return

            qg = cfg['q_grid']
            q  = (np.geomspace if qg.get('spacing','log')=='log' else np.linspace)(
                qg['q_min'], qg['q_max'], qg['n_points'])

            self.log_line.emit('\nSelecting energies …')
            energies, elem_results = _select_energies_multi(cfg, self.log_line.emit)
            n_e = len(elem_results)

            self.log_line.emit('\nComputing multilayer partial structure factors …')
            layer_cfgs  = cfg['layers']
            solvent_cfg = cfg['solvent']
            phi         = cfg['experiment']['volume_fraction']
            dist_type   = cfg.get('dist_type','Monodisperse')
            K           = int(cfg.get('gh_K', 12))

            I_MM, I_RiM, I_RiRi, I_RiRj = _compute_multilayer(
                q, layer_cfgs, solvent_cfg, phi, elem_results,
                dist_type, K=K, log=self.log_line.emit)

            self.log_line.emit(
                f'  I_MM(q_min)={I_MM[0]:.3e} cm⁻¹'
                +''.join(f'  I_R{i+1}R{i+1}={I_RiRi[i][0]:.3e}' for i in range(n_e)))

            nc = cfg['noise']
            if nc['model']=='poisson':
                self.log_line.emit(f'\nNoise: Poisson  N_peak={nc["N_peak"]:.0f}'
                                   f'  σ/I(q_min)={1/np.sqrt(nc["N_peak"])*100:.2f}%')
            else:
                self.log_line.emit(f'\nNoise: relative {nc.get("relative",0.005)*100:.2f}%')

            out_dir = Path(cfg['output']['directory'])
            name    = cfg['output'].get('name','particle')
            out_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(42)
            self.log_line.emit(f'\nWriting {len(energies)} files to {out_dir}/ …')

            I_obs_list = []
            for k, E in enumerate(energies):
                I_tot = I_MM.copy()
                for i, er in enumerate(elem_results):
                    fp_i=er['fp'][k]; fpp_i=er['fpp'][k]
                    I_tot += 2*fp_i*I_RiM[i]+(fp_i**2+fpp_i**2)*I_RiRi[i]
                for i in range(n_e):
                    for jj in range(i+1, n_e):
                        fp_i=elem_results[i]['fp'][k]; fpp_i=elem_results[i]['fpp'][k]
                        fp_j=elem_results[jj]['fp'][k]; fpp_j=elem_results[jj]['fpp'][k]
                        I_tot += 2*(fp_i*fp_j+fpp_i*fpp_j)*I_RiRj[(i,jj)]

                I_obs, sigma = _make_noise(I_tot, cfg, rng)
                I_obs_list.append(I_obs)

                layer_str = '  '.join(
                    f"R_{i}={layer_cfgs[i]['thickness']:.1f}Å" for i in range(len(layer_cfgs)))
                hdr = (f"Synthetic ASAXS — {n_e} resonant element(s)\n"
                       f"  Shape: {shape}  {layer_str}\n"
                       f"  phi={phi:.2e}  dist={dist_type}  noise={nc['model']}\n"
                       f"  Energy={E:.4f} keV  "
                       +'  '.join(f"f'_{i+1}={elem_results[i]['fp'][k]:.3f}" for i in range(n_e))+'\n'
                       f"  Columns: q(1/A)  I(cm-1)  sigma(cm-1)")
                np.savetxt(out_dir/f'{name}_{E:.4f}keV.dat',
                           np.column_stack([q, I_obs, sigma]),
                           header=hdr, fmt='%.6e')
                self.log_line.emit(
                    f'  [{k+1:2d}/{len(energies)}]  {E:.4f} keV  '
                    +'  '.join(f"f'_{i+1}={elem_results[i]['fp'][k]:.3f}" for i in range(n_e)))

            # Ground-truth partials
            cols = [q, I_MM]+list(I_RiM)+list(I_RiRi)
            hdrs_gt = ['q(1/A)','I_MM(cm-1)']+[f'I_R{i+1}M(cm-1)' for i in range(n_e)]\
                     +[f'I_R{i+1}R{i+1}(cm-1)' for i in range(n_e)]
            for i in range(n_e):
                for jj in range(i+1, n_e):
                    cols.append(I_RiRj[(i,jj)]); hdrs_gt.append(f'I_R{i+1}R{jj+1}(cm-1)')
            np.savetxt(out_dir/'ground_truth_partials.dat',
                       np.column_stack(cols),
                       header='  '.join(hdrs_gt), fmt='%.6e')
            self.log_line.emit(f'\nDone — {len(energies)} datasets in {out_dir}/')
            self.data_ready.emit(q, energies, I_obs_list)
            self.finished_ok.emit(str(out_dir))
        except Exception:
            self.error_signal.emit(traceback.format_exc())


# ── Widget helpers ─────────────────────────────────────────────────────────────
def _dbl(val, lo=0.0, hi=1e9, decimals=4, step=0.01):
    sb=QDoubleSpinBox(); sb.setRange(lo,hi); sb.setDecimals(decimals)
    sb.setSingleStep(step); sb.setValue(val); return sb

def _int(val, lo=1, hi=99999):
    sb=QSpinBox(); sb.setRange(lo,hi); sb.setValue(val); return sb

def _fit_cell(parent=None):
    """Centred QCheckBox widget for a table Fit column."""
    w=QWidget(parent); lay=QHBoxLayout(w)
    lay.setContentsMargins(0,0,0,0); lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
    cb=QCheckBox(w); lay.addWidget(cb); return w

def _tbl_dbl(parent, val, lo=0.0, hi=1e9, dec=4, step=0.01):
    sb=QDoubleSpinBox(parent); sb.setRange(lo,hi); sb.setDecimals(dec)
    sb.setSingleStep(step); sb.setValue(val); sb.setFrame(False); return sb

def _tbl_int(parent, val, lo=1, hi=99999):
    sb=QSpinBox(parent); sb.setRange(lo,hi); sb.setValue(val)
    sb.setFrame(False); return sb


# ── Main tab class ─────────────────────────────────────────────────────────────
class SimDataTab(QWidget):
    """Simulation and fitting tab (tab 6)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._curves = []
        # Fit data storage (filled by MainWindow signal wiring)
        self._fit_datasets: list = []       # from Data tab
        self._fit_q_decomp  = None          # from Decomp tab
        self._fit_I_MM = self._fit_I_RM = self._fit_I_RR = None
        self._fit_s_MM = self._fit_s_RM = self._fit_s_RR = None
        # Element state
        self._sim_elem_Z:          list = [0, 0, 0]
        self._sim_elem_chk:        list = []
        self._sim_elem_sym:        list = []
        self._sim_elem_edge_combo: list = []
        self._sim_elem_lbl:        list = []
        self._sim_elem_emin:       list = []
        self._sim_elem_emax:       list = []
        self._sim_elem_en:         list = []
        self._sim_elem_loc:        list = []
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── LEFT: scrollable parameters ──────────────────────────────────────
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form_w = QWidget(); form_lay = QVBoxLayout(form_w); form_lay.setSpacing(6)
        scroll.setWidget(form_w); splitter.addWidget(scroll)

        # ── Model (shape) ─────────────────────────────────────────────────────
        mdl_grp  = QGroupBox('Model')
        mdl_form = QFormLayout(mdl_grp)
        self._shape_combo = QComboBox(); self._shape_combo.addItems(_SHAPES)
        self._shape_combo.setCurrentText('Sphere')
        self._shape_combo.currentTextChanged.connect(self._on_shape_changed)
        mdl_form.addRow('Shape:', self._shape_combo)
        self._shape_note = QLabel('')
        self._shape_note.setStyleSheet('color:#888; font-size:11px;')
        mdl_form.addRow('', self._shape_note)
        form_lay.addWidget(mdl_grp)

        # ── Layers table ──────────────────────────────────────────────────────
        lyr_grp = QGroupBox('Layers')
        lyr_lay = QVBoxLayout(lyr_grp)

        self._layer_table = QTableWidget(0, 8)
        self._layer_table.setHorizontalHeaderLabels(_COL_HDRS)
        self._layer_table.verticalHeader().setVisible(False)
        self._layer_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._layer_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                          | QAbstractItemView.EditTrigger.SelectedClicked)
        hdr = self._layer_table.horizontalHeader()
        for c, w in enumerate(_COL_W):
            hdr.resizeSection(c, w)
        hdr.setStretchLastSection(False)
        self._layer_table.setMinimumHeight(110)
        self._layer_table.cellChanged.connect(self._on_cell_changed)
        lyr_lay.addWidget(self._layer_table)

        btn_row = QHBoxLayout()
        self._btn_add_shell    = QPushButton('+ Add shell')
        self._btn_remove_shell = QPushButton('× Remove shell')
        self._btn_remove_shell.setEnabled(False)
        self._btn_add_shell.clicked.connect(self._insert_shell)
        self._btn_remove_shell.clicked.connect(self._remove_shell)
        btn_row.addWidget(self._btn_add_shell)
        btn_row.addWidget(self._btn_remove_shell)
        btn_row.addStretch()
        lyr_lay.addLayout(btn_row)

        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel('Thickness distribution:'))
        self._dist_combo = QComboBox(); self._dist_combo.addItems(_DIST_TYPES)
        self._dist_combo.setCurrentText('Log-Normal')
        dist_row.addWidget(self._dist_combo)
        dist_row.addSpacing(16)
        dist_row.addWidget(QLabel('K (GH pts):'))
        self._gh_K = _int(12, lo=3, hi=30); self._gh_K.setFixedWidth(52)
        dist_row.addWidget(self._gh_K)
        dist_row.addStretch()
        lyr_lay.addLayout(dist_row)

        lyr_note = QLabel(
            '<i>Core: t/R = core radius.  Shells: t/R = shell thickness.<br>'
            'σ applies to each layer independently (tensor-product Gauss-Hermite).<br>'
            'Solvent is always infinite — edit ρ/M/ne only.</i>')
        lyr_note.setWordWrap(True)
        lyr_note.setStyleSheet('color:#888; font-size:11px;')
        lyr_lay.addWidget(lyr_note)
        form_lay.addWidget(lyr_grp)

        # Populate default rows
        self._init_layer_table()

        # ── Resonant elements ─────────────────────────────────────────────────
        en_grp = QGroupBox('Resonant elements')
        en_lay = QVBoxLayout(en_grp)

        _sym_list = []
        if _HAS_XRAYDB:
            for _Z in range(3, 93):
                try: _sym_list.append((xraydb.atomic_symbol(_Z), _Z))
                except Exception: pass
        if not _sym_list:
            _FB = ['Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S',
                   'Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
                   'Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo',
                   'Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba',
                   'La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
                   'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Th','U']
            _sym_list=[(s,z) for z,s in enumerate(_FB, start=3)]

        _SHELLS_ALL=['K','L1','L2','L3','M1','M2','M3','M4','M5','N1','N2','N3']
        _DEF_ELEM=[('Au',79),('Pt',78),('Se',34)]; _DEF_EDGE=['L3','L3','K']

        for i in range(3):
            ew=QWidget(); ev=QVBoxLayout(ew)
            ev.setContentsMargins(0,2,0,4); ev.setSpacing(2)

            r1=QHBoxLayout()
            if i==0:
                r1.addWidget(QLabel('Element 1:')); chk=None
            else:
                chk=QCheckBox(f'Element {i+1}:'); chk.setChecked(False); r1.addWidget(chk)
            self._sim_elem_chk.append(chk)

            sym_cb=QComboBox(); sym_cb.setMinimumWidth(95)
            for sym,Z in _sym_list: sym_cb.addItem(f'{sym} ({Z})')
            ds,dz=_DEF_ELEM[i]; idx=sym_cb.findText(f'{ds} ({dz})')
            if idx>=0: sym_cb.setCurrentIndex(idx)
            self._sim_elem_sym.append(sym_cb); r1.addWidget(sym_cb)

            edge_cb=QComboBox(); edge_cb.addItems(_SHELLS_ALL)
            edge_cb.setCurrentText(_DEF_EDGE[i]); edge_cb.setFixedWidth(52)
            self._sim_elem_edge_combo.append(edge_cb); r1.addWidget(edge_cb)

            lbl_e=QLabel(); lbl_e.setStyleSheet('color:#888; font-size:11px;')
            lbl_e.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
            self._sim_elem_lbl.append(lbl_e); r1.addWidget(lbl_e,1)
            ev.addLayout(r1)

            r2=QHBoxLayout(); r2.addSpacing(16); r2.addWidget(QLabel('E:'))
            emin=_dbl(10.419,lo=0.1,hi=1000.0,decimals=4,step=0.01); emin.setFixedWidth(78)
            emax=_dbl(10.919,lo=0.1,hi=1000.0,decimals=4,step=0.01); emax.setFixedWidth(78)
            r2.addWidget(emin); r2.addWidget(QLabel('–')); r2.addWidget(emax)
            r2.addWidget(QLabel('keV'))
            r2.addSpacing(8); r2.addWidget(QLabel('N:'))
            n_sb=_int(20,lo=2,hi=200); n_sb.setFixedWidth(52); r2.addWidget(n_sb)
            r2.addSpacing(8); r2.addWidget(QLabel('Layer:'))
            loc_cb=QComboBox(); loc_cb.setMinimumWidth(90)
            r2.addWidget(loc_cb); r2.addStretch()
            self._sim_elem_emin.append(emin); self._sim_elem_emax.append(emax)
            self._sim_elem_en.append(n_sb); self._sim_elem_loc.append(loc_cb)
            ev.addLayout(r2)

            if i!=0:
                for w in (sym_cb,edge_cb,emin,emax,n_sb,loc_cb): w.setEnabled(False)
            en_lay.addWidget(ew)

            sym_cb.currentIndexChanged.connect(lambda _,ei=i: self._on_sim_symbol_changed(ei))
            edge_cb.currentIndexChanged.connect(lambda _,ei=i: self._on_sim_edge_changed(ei))
            if chk: chk.toggled.connect(lambda chk,ei=i: self._on_sim_elem_toggled(chk,ei))

        self._btn_preview_E=QPushButton('Preview selected energies')
        self._btn_preview_E.clicked.connect(self._preview_energies)
        en_lay.addWidget(self._btn_preview_E)
        form_lay.addWidget(en_grp)
        for i in range(3): self._on_sim_symbol_changed(i)
        self._sync_elem_layer_combos()

        # ── Experiment ────────────────────────────────────────────────────────
        samp_grp=QGroupBox('Experiment')
        samp_form=QFormLayout(samp_grp)
        self._exp_phi=_dbl(1e-4,lo=1e-10,hi=1.0,decimals=6,step=1e-5)
        samp_form.addRow('Volume fraction φ:',self._exp_phi)
        form_lay.addWidget(samp_grp)

        # ── q grid ────────────────────────────────────────────────────────────
        q_grp=QGroupBox('q grid')
        q_form=QFormLayout(q_grp)
        self._q_min=_dbl(0.003,lo=1e-5,hi=10.0,decimals=5,step=0.001)
        self._q_max=_dbl(0.145,lo=1e-5,hi=10.0,decimals=5,step=0.001)
        self._q_n  =_int(500,lo=10,hi=10000)
        self._q_log=QComboBox(); self._q_log.addItems(['Logarithmic','Linear'])
        q_form.addRow('q_min (Å⁻¹):',self._q_min)
        q_form.addRow('q_max (Å⁻¹):',self._q_max)
        q_form.addRow('N points:',self._q_n)
        q_form.addRow('Spacing:',self._q_log)
        form_lay.addWidget(q_grp)

        # ── Mode (Simulate / Fit) ─────────────────────────────────────────────
        mode_grp=QGroupBox('Mode')
        mode_lay=QVBoxLayout(mode_grp)
        self._rb_sim   =QRadioButton('Simulate data')
        self._rb_fit_iq=QRadioButton('Fit to I(q, E) from Data tab')
        self._rb_fit_dc=QRadioButton('Fit to decomposed components (I_MM / I_RM / I_RR)')
        self._rb_sim.setChecked(True)
        self._mode_bg=QButtonGroup()
        for rb in (self._rb_sim,self._rb_fit_iq,self._rb_fit_dc):
            self._mode_bg.addButton(rb); mode_lay.addWidget(rb)
        self._fit_info_lbl=QLabel('No data loaded yet.')
        self._fit_info_lbl.setStyleSheet('color:#888; font-size:11px;')
        self._fit_info_lbl.setWordWrap(True)
        mode_lay.addWidget(self._fit_info_lbl)
        self._rb_sim.toggled.connect(self._on_mode_changed)
        self._rb_fit_iq.toggled.connect(self._on_mode_changed)
        self._rb_fit_dc.toggled.connect(self._on_mode_changed)
        form_lay.addWidget(mode_grp)

        # ── Noise (visible in Simulate mode only) ─────────────────────────────
        self._noise_grp=QGroupBox('Noise model')
        noise_form=QFormLayout(self._noise_grp)
        self._rb_poisson =QRadioButton('Poissonian  σ(q) = √(I·I(q_min)/N_peak)')
        self._rb_relative=QRadioButton('Relative       σ(q) = p · I(q)')
        self._rb_poisson.setChecked(True)
        nbg=QButtonGroup(); nbg.addButton(self._rb_poisson); nbg.addButton(self._rb_relative)
        self._noise_bg=nbg
        self._n_peak=_int(10000,lo=1,hi=10000000)
        self._n_rel =_dbl(0.005,lo=1e-6,hi=1.0,decimals=4,step=0.001)
        noise_form.addRow(self._rb_poisson)
        noise_form.addRow('N_peak:',self._n_peak)
        noise_form.addRow(self._rb_relative)
        noise_form.addRow('Relative σ/I:',self._n_rel)
        form_lay.addWidget(self._noise_grp)

        # ── Output (visible in Simulate mode only) ────────────────────────────
        self._out_grp=QGroupBox('Output')
        out_form=QFormLayout(self._out_grp)
        self._out_dir =QLineEdit(str(Path.home()/'asaxs_sim_data'))
        self._out_name=QLineEdit('particle')
        btn_br=QPushButton('Browse…'); btn_br.setFixedWidth(70)
        dr=QHBoxLayout(); dr.addWidget(self._out_dir); dr.addWidget(btn_br)
        out_form.addRow('Directory:',dr)
        out_form.addRow('File prefix:',self._out_name)
        btn_br.clicked.connect(self._browse_dir)
        form_lay.addWidget(self._out_grp)

        # ── Settings + buttons ────────────────────────────────────────────────
        cfg_row=QHBoxLayout()
        self._btn_save_cfg=QPushButton('Save settings…')
        self._btn_load_cfg=QPushButton('Load settings…')
        self._btn_save_cfg.clicked.connect(self._save_settings)
        self._btn_load_cfg.clicked.connect(self._load_settings)
        cfg_row.addWidget(self._btn_save_cfg); cfg_row.addWidget(self._btn_load_cfg)
        form_lay.addLayout(cfg_row)

        self._btn_gen=QPushButton('Generate data')
        self._btn_gen.setFixedHeight(36)
        self._btn_gen.setStyleSheet('font-weight:bold; font-size:13px;')
        self._btn_gen.clicked.connect(self._generate)
        form_lay.addWidget(self._btn_gen)

        self._btn_fit=QPushButton('Run fit  (coming soon)')
        self._btn_fit.setFixedHeight(36)
        self._btn_fit.setStyleSheet('font-weight:bold; font-size:13px;')
        self._btn_fit.setEnabled(False)
        form_lay.addWidget(self._btn_fit)

        form_lay.addStretch()
        self._on_mode_changed()

        # ── RIGHT: plot + log ─────────────────────────────────────────────────
        right_w=QWidget(); right_lay=QVBoxLayout(right_w)
        right_lay.setContentsMargins(0,0,0,0)
        right_split=QSplitter(Qt.Orientation.Vertical)
        right_lay.addWidget(right_split); splitter.addWidget(right_w)

        plot_w=QWidget(); plot_lay=QVBoxLayout(plot_w)
        plot_lay.setContentsMargins(4,4,4,0)
        self._pw=pg.PlotWidget(); self._pw.setBackground('#1e1e1e')
        self._pw.showGrid(x=True,y=True,alpha=0.3)
        self._pw.setLogMode(x=True,y=True)
        self._pw.setLabel('bottom','q',units='Å⁻¹')
        self._pw.setLabel('left','I(q)',units='cm⁻¹')
        self._pw.addLegend(offset=(10,10))
        self._coord_lbl=QLabel('x = —    y = —')
        self._coord_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_lay.addWidget(self._pw); plot_lay.addWidget(self._coord_lbl)
        self._proxy=add_crosshair(self._pw,x_fmt='.4g',y_fmt='.4e',label=self._coord_lbl)
        right_split.addWidget(plot_w)

        log_w=QWidget(); log_lay=QVBoxLayout(log_w)
        log_lay.setContentsMargins(4,4,4,4)
        log_lay.addWidget(QLabel('Log'))
        self._log=QTextEdit(); self._log.setReadOnly(True)
        mono=QFont('Courier New',10); mono.setStyleHint(QFont.StyleHint.Monospace)
        self._log.setFont(mono)
        btn_cl=QPushButton('Clear'); btn_cl.setFixedWidth(70)
        btn_cl.clicked.connect(self._log.clear)
        log_lay.addWidget(self._log); log_lay.addWidget(btn_cl)
        right_split.addWidget(log_w)

        right_split.setSizes([500,260])
        splitter.setSizes([430,800])

    # ── Layer table management ─────────────────────────────────────────────────

    def _init_layer_table(self):
        """Populate default Core → Shell → Solvent rows."""
        t = self._layer_table
        t.blockSignals(True)
        # Core
        self._add_layer_row(0, 'Core',    'Au',  19.32, 196.97, 79, 55.0,  11.0,  False)
        # Shell
        self._add_layer_row(1, 'Shell',   'SiO2', 2.196, 60.08, 30, 165.0,  0.0,  False)
        # Solvent (last)
        self._add_layer_row(2, 'Solvent', 'H2O',  1.000, 18.015, 10,  0.0,   0.0,  True)
        t.blockSignals(False)

    def _add_layer_row(self, row_idx, name, formula, density, molar_mass,
                       n_electrons, thickness, sigma, is_solvent=False):
        t = self._layer_table
        t.insertRow(row_idx)

        t.setCellWidget(row_idx, _C_FIT,  _fit_cell(t))
        t.setItem(row_idx, _C_NAME, QTableWidgetItem(name))
        t.setItem(row_idx, _C_FML,  QTableWidgetItem(formula))
        t.setCellWidget(row_idx, _C_RHO, _tbl_dbl(t, density,     0.0, 30.0,  4, 0.01))
        t.setCellWidget(row_idx, _C_M,   _tbl_dbl(t, molar_mass,  0.0, 1e6,   3, 1.0))
        t.setCellWidget(row_idx, _C_NE,  _tbl_int(t, n_electrons))

        if is_solvent:
            lbl=QLabel('∞', t); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setCellWidget(row_idx, _C_T,   lbl)
            lbl2=QLabel('—', t); lbl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setCellWidget(row_idx, _C_SIG, lbl2)
        else:
            t.setCellWidget(row_idx, _C_T,   _tbl_dbl(t, thickness, 0.1, 1e6, 2, 1.0))
            t.setCellWidget(row_idx, _C_SIG, _tbl_dbl(t, sigma,     0.0, 1e6, 2, 0.5))

    def _insert_shell(self):
        t = self._layer_table; n = t.rowCount()
        ins = n - 1                           # insert before solvent
        prev = ins - 1                        # copy from last non-solvent row
        self._layer_table.blockSignals(True)
        prev_name = t.item(prev, _C_NAME).text() if t.item(prev, _C_NAME) else 'Shell'
        self._add_layer_row(
            ins,
            name        = f'Shell {ins}',
            formula     = t.item(prev, _C_FML).text() if t.item(prev, _C_FML) else '',
            density     = t.cellWidget(prev, _C_RHO).value(),
            molar_mass  = t.cellWidget(prev, _C_M).value(),
            n_electrons = t.cellWidget(prev, _C_NE).value(),
            thickness   = t.cellWidget(prev, _C_T).value(),
            sigma       = t.cellWidget(prev, _C_SIG).value(),
            is_solvent  = False,
        )
        self._layer_table.blockSignals(False)
        self._btn_remove_shell.setEnabled(True)
        self._sync_elem_layer_combos()

    def _remove_shell(self):
        t = self._layer_table; n = t.rowCount()
        row = t.currentRow()
        if row < 1 or row >= n-1:
            row = n - 2          # default: remove last shell
        if n <= 2: return        # only core + solvent left
        t.removeRow(row)
        self._btn_remove_shell.setEnabled(t.rowCount() > 2)
        self._sync_elem_layer_combos()

    def _on_cell_changed(self, row, col):
        if col == _C_FML:
            item = self._layer_table.item(row, _C_FML)
            if item: self._fill_from_formula(row, item.text())
        elif col == _C_NAME:
            self._sync_elem_layer_combos()

    def _fill_from_formula(self, row, formula):
        if not _HAS_XRAYDB or not formula.strip(): return
        try:
            M, ne = _formula_to_Mne(formula)
            self._layer_table.blockSignals(True)
            M_sb  = self._layer_table.cellWidget(row, _C_M)
            ne_sb = self._layer_table.cellWidget(row, _C_NE)
            if M_sb  and hasattr(M_sb,  'setValue'): M_sb.setValue(M)
            if ne_sb and hasattr(ne_sb, 'setValue'): ne_sb.setValue(int(ne))
            self._layer_table.blockSignals(False)
        except Exception:
            self._layer_table.blockSignals(False)

    def _get_layer_params(self) -> list:
        """Return list of layer dicts (all rows except last = solvent)."""
        t = self._layer_table; n = t.rowCount()
        layers = []
        for row in range(n-1):
            name_i = t.item(row, _C_NAME); fml_i = t.item(row, _C_FML)
            layers.append({
                'name':        name_i.text() if name_i else f'Layer{row+1}',
                'formula':     fml_i.text()  if fml_i  else '',
                'density':     t.cellWidget(row, _C_RHO).value(),
                'molar_mass':  t.cellWidget(row, _C_M).value(),
                'n_electrons': int(t.cellWidget(row, _C_NE).value()),
                'thickness':   t.cellWidget(row, _C_T).value(),
                'sigma':       t.cellWidget(row, _C_SIG).value(),
                'fit':         t.cellWidget(row, _C_FIT).findChild(QCheckBox).isChecked(),
            })
        return layers

    def _get_solvent_params(self) -> dict:
        t = self._layer_table; row = t.rowCount()-1
        name_i = t.item(row, _C_NAME); fml_i = t.item(row, _C_FML)
        return {
            'name':        name_i.text() if name_i else 'Solvent',
            'formula':     fml_i.text()  if fml_i  else '',
            'density':     t.cellWidget(row, _C_RHO).value(),
            'molar_mass':  t.cellWidget(row, _C_M).value(),
            'n_electrons': int(t.cellWidget(row, _C_NE).value()),
        }

    def _sync_elem_layer_combos(self):
        """Refresh resonant-element Layer combos to match current table rows."""
        t = self._layer_table; n = t.rowCount()
        names = []
        for row in range(n-1):
            it = t.item(row, _C_NAME)
            names.append(it.text() if it else f'Layer {row+1}')
        for loc_cb in self._sim_elem_loc:
            prev = loc_cb.currentText()
            loc_cb.blockSignals(True); loc_cb.clear(); loc_cb.addItems(names)
            if prev in names: loc_cb.setCurrentText(prev)
            loc_cb.blockSignals(False)

    def _layer_name_to_idx(self, name: str) -> int:
        t = self._layer_table; n = t.rowCount()
        for row in range(n-1):
            it = t.item(row, _C_NAME)
            if it and it.text() == name: return row
        return 0

    # ── Shape change ──────────────────────────────────────────────────────────
    def _on_shape_changed(self, shape):
        if shape != 'Sphere':
            self._shape_note.setText(f'{shape} — not yet implemented; Sphere will be used.')
        else:
            self._shape_note.setText('')

    # ── Mode change ───────────────────────────────────────────────────────────
    def _on_mode_changed(self):
        sim = self._rb_sim.isChecked()
        self._noise_grp.setVisible(sim)
        self._out_grp.setVisible(sim)
        self._btn_gen.setVisible(sim)
        self._btn_fit.setVisible(not sim)
        self._update_fit_info()

    def _update_fit_info(self):
        if self._rb_fit_iq.isChecked():
            n = len(self._fit_datasets)
            if n:
                self._fit_info_lbl.setText(f'{n} I(q,E) datasets loaded from Data tab.')
            else:
                self._fit_info_lbl.setText('No I(q,E) data — load datasets in tab 1 first.')
        elif self._rb_fit_dc.isChecked():
            if self._fit_q_decomp is not None:
                self._fit_info_lbl.setText(
                    f'Decomposed components available ({len(self._fit_q_decomp)} q-points).')
            else:
                self._fit_info_lbl.setText('No decomposed components — run decomposition in tab 3 first.')
        else:
            self._fit_info_lbl.setText('')

    # ── Slots called by MainWindow ────────────────────────────────────────────
    def set_datasets(self, datasets: list):
        """Receive energy-dependent datasets from the Data tab."""
        self._fit_datasets = datasets
        self._update_fit_info()

    def set_partials(self, q, I_MM, I_RM, I_RR, s_MM, s_RM, s_RR):
        """Receive decomposed components from the q-space decomp tab."""
        self._fit_q_decomp = q
        self._fit_I_MM, self._fit_I_RM, self._fit_I_RR = I_MM, I_RM, I_RR
        self._fit_s_MM, self._fit_s_RM, self._fit_s_RR = s_MM, s_RM, s_RR
        self._update_fit_info()

    # ── Resonant element handlers (same as before) ────────────────────────────
    def _elem_Z_from_combo(self, idx):
        txt = self._sim_elem_sym[idx].currentText()
        try: return int(txt.split('(')[1].rstrip(')'))
        except Exception: return 0

    def _on_sim_symbol_changed(self, ei):
        Z = self._elem_Z_from_combo(ei); self._sim_elem_Z[ei] = Z
        lbl = self._sim_elem_lbl[ei]; edge_cb = self._sim_elem_edge_combo[ei]
        if not _HAS_XRAYDB: lbl.setText('xraydb not installed'); return
        try: edges = xraydb.xray_edges(Z)
        except Exception: edges = {}
        _SHELLS=['K','L1','L2','L3','M1','M2','M3','M4','M5','N1','N2','N3']
        edge_cb.blockSignals(True); prev=edge_cb.currentText(); edge_cb.clear()
        avail=[sh for sh in _SHELLS if sh in edges and edges[sh].energy>100]
        if not avail: avail=[sh for sh in _SHELLS if sh in edges]
        if not avail: avail=['K']
        for sh in avail: edge_cb.addItem(sh)
        if prev in avail: edge_cb.setCurrentText(prev)
        else:
            default='L3' if Z>50 and 'L3' in avail else avail[0]
            edge_cb.setCurrentText(default)
        edge_cb.blockSignals(False)
        self._on_sim_edge_changed(ei)

    def _on_sim_edge_changed(self, ei):
        Z = self._sim_elem_Z[ei]
        if not isinstance(Z,int) or Z==0:
            Z=self._elem_Z_from_combo(ei); self._sim_elem_Z[ei]=Z
        shell = self._sim_elem_edge_combo[ei].currentText()
        lbl   = self._sim_elem_lbl[ei]
        if not _HAS_XRAYDB or Z==0: return
        try: edge_eV=xraydb.xray_edge(Z,shell).energy
        except Exception: lbl.setText('edge not found'); return
        edge_keV=edge_eV/1000.0
        emax_sb=self._sim_elem_emax[ei]; emin_sb=self._sim_elem_emin[ei]
        emax_sb.blockSignals(True); emin_sb.blockSignals(True)
        emax_sb.setValue(edge_keV); emin_sb.setValue(max(0.1,edge_keV-0.5))
        emax_sb.blockSignals(False); emin_sb.blockSignals(False)
        lbl.setText(f'{shell} edge: {edge_keV:.4f} keV')

    def _on_sim_elem_toggled(self, checked, ei):
        for w in (self._sim_elem_sym[ei], self._sim_elem_edge_combo[ei],
                  self._sim_elem_emin[ei], self._sim_elem_emax[ei],
                  self._sim_elem_en[ei],   self._sim_elem_loc[ei]):
            w.setEnabled(checked)

    # ── Config build ──────────────────────────────────────────────────────────
    def _build_config(self) -> dict:
        layers  = self._get_layer_params()
        solvent = self._get_solvent_params()
        res_elems = [{'Z': self._sim_elem_Z[0],
                      'layer_idx': self._layer_name_to_idx(self._sim_elem_loc[0].currentText()),
                      'E_min_keV': self._sim_elem_emin[0].value(),
                      'E_max_keV': self._sim_elem_emax[0].value(),
                      'n_energies': self._sim_elem_en[0].value()}]
        for i in range(1,3):
            if self._sim_elem_chk[i] and self._sim_elem_chk[i].isChecked():
                res_elems.append({'Z': self._sim_elem_Z[i],
                                  'layer_idx': self._layer_name_to_idx(self._sim_elem_loc[i].currentText()),
                                  'E_min_keV': self._sim_elem_emin[i].value(),
                                  'E_max_keV': self._sim_elem_emax[i].value(),
                                  'n_energies': self._sim_elem_en[i].value()})
        return {
            'shape':    self._shape_combo.currentText(),
            'layers':   layers,
            'solvent':  solvent,
            'dist_type': self._dist_combo.currentText(),
            'gh_K':     self._gh_K.value(),
            'resonant_elements': res_elems,
            'experiment': {'volume_fraction': self._exp_phi.value(),
                           'resonant_Z': self._sim_elem_Z[0],
                           'E_min_keV': res_elems[0]['E_min_keV'],
                           'E_max_keV': res_elems[0]['E_max_keV'],
                           'n_energies': res_elems[0]['n_energies']},
            'q_grid': {'q_min': self._q_min.value(), 'q_max': self._q_max.value(),
                       'n_points': self._q_n.value(),
                       'spacing': 'log' if self._q_log.currentIndex()==0 else 'linear'},
            'noise': {'model': 'poisson' if self._rb_poisson.isChecked() else 'relative',
                      'N_peak': self._n_peak.value(), 'relative': self._n_rel.value()},
            'output': {'directory': self._out_dir.text(), 'name': self._out_name.text()},
        }

    # ── Generation ────────────────────────────────────────────────────────────
    def _generate(self):
        if not _HAS_XRAYDB:
            self._log.append('<b>Error:</b> xraydb not installed'); return
        if self._thread and self._thread.isRunning(): return
        import json
        cfg = self._build_config()
        try:
            out_dir = Path(cfg['output']['directory'])
            out_dir.mkdir(parents=True, exist_ok=True)
            sp = out_dir/f"{cfg['output']['name']}_settings.asaxs_sim"
            with open(sp,'w') as f: json.dump(cfg,f,indent=2)
        except Exception: pass
        self._log.clear()
        n_e   = len(cfg['resonant_elements'])
        n_lyr = len(cfg['layers'])
        self._log.append('<b>Generating ASAXS data …</b>')
        self._log.append(f'Shape: {cfg["shape"]}  Layers: {n_lyr}  Elements: {n_e}')
        self._log.append(f'Dist: {cfg["dist_type"]}  K={cfg["gh_K"]}'
                         + ('  [Numba]' if _HAS_NUMBA else '  [NumPy]'))
        layer_str = '  '.join(f'{L["name"]}(t={L["thickness"]:.0f},σ={L["sigma"]:.0f}Å)'
                               for L in cfg['layers'])
        self._log.append(f'Layers: {layer_str}')
        self._btn_gen.setEnabled(False); self._btn_gen.setText('Generating …')
        self._thread = _GeneratorThread(cfg, self)
        self._thread.log_line.connect(self._log.append)
        self._thread.data_ready.connect(self._plot_data)
        self._thread.finished_ok.connect(self._on_done)
        self._thread.error_signal.connect(self._on_error)
        self._thread.start()

    def _plot_data(self, q, energies, I_obs_list):
        for c in self._curves: self._pw.removeItem(c)
        self._curves.clear(); self._pw.getPlotItem().legend.clear()
        N=len(energies); E0,E1=energies[0],energies[-1]; dE=E1-E0 if E1>E0 else 1.0
        for i,(E,I) in enumerate(zip(energies,I_obs_list)):
            r,g,b=_energy_color((E-E0)/dE)
            c=self._pw.plot(q,I,pen=pg.mkPen(color=(r,g,b,200),width=1.2),name=f'{E:.4f} keV')
            self._curves.append(c)
        self._pw.setTitle(f'Simulated I(q, E) — {N} datasets')

    def _on_done(self, out_dir):
        self._log.append(f'\n<b>✓ Complete.</b>  Files: {out_dir}')
        self._btn_gen.setEnabled(True); self._btn_gen.setText('Generate data')

    def _on_error(self, tb):
        self._log.append(f'\n<b>Error:</b>\n{tb}')
        self._btn_gen.setEnabled(True); self._btn_gen.setText('Generate data')

    # ── Preview energies ──────────────────────────────────────────────────────
    def _preview_energies(self):
        if not _HAS_XRAYDB: self._log.append('xraydb not installed'); return
        cfg = self._build_config(); elems = cfg['resonant_elements']
        self._log.append('<b>Energy preview:</b>')
        for i,e in enumerate(elems):
            self._log.append(f'  Elem {i+1}: Z={e["Z"]}  layer={e["layer_idx"]}'
                             f'  E={e["E_min_keV"]:.4f}–{e["E_max_keV"]:.4f} keV  N={e["n_energies"]}')
        try:
            Es, er = _select_energies_multi(cfg)
            self._log.append(f'  Combined: {len(Es)} energies')
            n = len(er)
            hdr=f'  {"E (keV)":>10}'+''.join(f"  {f'fp_{i+1}':>9}" for i in range(n))\
                                    +''.join(f"  {f'fpp_{i+1}':>9}" for i in range(n))
            self._log.append(hdr)
            for k,e in enumerate(Es):
                row=f'  {e:10.4f}'+''.join(f'  {er[i]["fp"][k]:9.3f}' for i in range(n))\
                                  +''.join(f'  {er[i]["fpp"][k]:9.3f}' for i in range(n))
                self._log.append(row)
        except Exception as ex:
            self._log.append(f'  Error: {ex}')

    # ── Settings save / load ──────────────────────────────────────────────────
    def _browse_dir(self):
        d=QFileDialog.getExistingDirectory(self,'Output directory',self._out_dir.text())
        if d: self._out_dir.setText(d)

    def _save_settings(self):
        import json
        path,_=QFileDialog.getSaveFileName(
            self,'Save settings','','ASAXS sim (*.asaxs_sim);;JSON (*.json);;All (*)')
        if not path: return
        cfg=self._build_config()
        try:
            with open(path,'w') as f: json.dump(cfg,f,indent=2)
            self._log.append(f'<b>Settings saved:</b> {path}')
        except Exception as e:
            self._log.append(f'<b>Save error:</b> {e}')

    def _load_settings(self):
        import json
        path,_=QFileDialog.getOpenFileName(
            self,'Load settings','','ASAXS sim (*.asaxs_sim);;JSON (*.json);;All (*)')
        if not path: return
        try:
            with open(path,'r') as f: cfg=json.load(f)
            self._apply_config(cfg)
            self._log.append(f'<b>Settings loaded:</b> {path}')
        except Exception as e:
            self._log.append(f'<b>Load error:</b> {e}')

    def _apply_config(self, cfg: dict):
        """Restore UI from a saved config dict."""
        if 'shape' in cfg:
            self._shape_combo.setCurrentText(cfg['shape'])
        if 'dist_type' in cfg:
            self._dist_combo.setCurrentText(cfg['dist_type'])
        if 'gh_K' in cfg:
            self._gh_K.setValue(int(cfg['gh_K']))

        # Rebuild layer table
        layers  = cfg.get('layers', [])
        solvent = cfg.get('solvent', {})
        if layers:
            t = self._layer_table
            t.blockSignals(True)
            while t.rowCount(): t.removeRow(0)
            for i, L in enumerate(layers):
                self._add_layer_row(i, L.get('name',f'Layer{i+1}'),
                                    L.get('formula',''), L.get('density',1.0),
                                    L.get('molar_mass',18.0), L.get('n_electrons',10),
                                    L.get('thickness',10.0), L.get('sigma',0.0))
            # Solvent
            row = len(layers)
            self._add_layer_row(row, solvent.get('name','Solvent'),
                                solvent.get('formula','H2O'),
                                solvent.get('density',1.0),
                                solvent.get('molar_mass',18.015),
                                solvent.get('n_electrons',10),
                                0.0, 0.0, is_solvent=True)
            t.blockSignals(False)
            self._btn_remove_shell.setEnabled(t.rowCount() > 2)
            self._sync_elem_layer_combos()

        e = cfg.get('experiment', {})
        if 'volume_fraction' in e: self._exp_phi.setValue(e['volume_fraction'])

        elems = cfg.get('resonant_elements', [])
        for i in range(3):
            active = i < len(elems)
            if i > 0 and self._sim_elem_chk[i]:
                self._sim_elem_chk[i].setChecked(active)
            if not active: continue
            elem = elems[i]; Z = elem.get('Z', 79)
            sym_cb = self._sim_elem_sym[i]
            for j in range(sym_cb.count()):
                try:
                    if int(sym_cb.itemText(j).split('(')[1].rstrip(')'))==Z:
                        sym_cb.setCurrentIndex(j); break
                except Exception: pass
            self._sim_elem_emin[i].setValue(elem.get('E_min_keV',10.0))
            self._sim_elem_emax[i].setValue(elem.get('E_max_keV',11.0))
            self._sim_elem_en[i].setValue(int(elem.get('n_energies',20)))
            self._on_sim_symbol_changed(i)
            # Layer combo
            li = elem.get('layer_idx', 0)
            loc_cb = self._sim_elem_loc[i]
            if li < loc_cb.count(): loc_cb.setCurrentIndex(li)

        qg = cfg.get('q_grid', {})
        if 'q_min'    in qg: self._q_min.setValue(qg['q_min'])
        if 'q_max'    in qg: self._q_max.setValue(qg['q_max'])
        if 'n_points' in qg: self._q_n.setValue(qg['n_points'])
        if 'spacing'  in qg:
            self._q_log.setCurrentIndex(0 if qg['spacing']=='log' else 1)

        n = cfg.get('noise', {})
        if n.get('model')=='poisson':  self._rb_poisson.setChecked(True)
        elif n.get('model')=='relative': self._rb_relative.setChecked(True)
        if 'N_peak'   in n: self._n_peak.setValue(int(n['N_peak']))
        if 'relative' in n: self._n_rel.setValue(n['relative'])

        out = cfg.get('output', {})
        if 'directory' in out: self._out_dir.setText(out['directory'])
        if 'name'      in out: self._out_name.setText(out['name'])
