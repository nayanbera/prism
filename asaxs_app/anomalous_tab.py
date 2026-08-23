"""Anomalous scattering factors tab — element, edge, f'/f'' from xraydb (NIST/Chantler)."""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QSpinBox, QComboBox, QPushButton, QDoubleSpinBox,
)
from PyQt6.QtCore import pyqtSignal
import pyqtgraph as pg

try:
    import xraydb
    _HAS_XRAYDB = True
except ImportError:
    _HAS_XRAYDB = False

from .core.decompose import build_A, condition_number

# xraydb uses string edge names — same as our UI labels
_SHELLS = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5']


def _fp_fpp(Z: int, energies_keV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (f', f'') for element Z at each energy (keV) using NIST Chantler tables."""
    if not _HAS_XRAYDB:
        raise RuntimeError("xraydb not installed — run: pip install xraydb")
    energies_eV = energies_keV * 1000.0
    fp  = np.array([xraydb.f1_chantler(Z, e) for e in energies_eV], dtype=float)
    fpp = np.array([abs(xraydb.f2_chantler(Z, e)) for e in energies_eV], dtype=float)
    return fp, fpp


def _edge_energy_keV(Z: int, shell: str) -> float:
    """Edge energy in keV from xraydb, or 0 if not found."""
    if not _HAS_XRAYDB:
        return 0.0
    try:
        edge = xraydb.xray_edge(Z, shell)
        return float(edge.energy) / 1000.0  # eV → keV
    except Exception:
        return 0.0


# Common resonant elements at synchrotron SAXS beamlines
_COMMON = {
    'Au (79) — L edges': 79,
    'Pt (78) — L edges': 78,
    'Se (34) — K edge':  34,
    'Br (35) — K edge':  35,
    'Sr (38) — K edge':  38,
    'Ge (32) — K edge':  32,
    'Fe (26) — K edge':  26,
    'Zn (30) — K edge':  30,
    'Cu (29) — K edge':  29,
    'Ni (28) — K edge':  28,
}


class AnomalousTab(QWidget):
    fp_fpp_ready = pyqtSignal(np.ndarray, np.ndarray)   # fp, fpp arrays

    def __init__(self, parent=None):
        super().__init__(parent)
        self._energies: np.ndarray = np.array([])
        self._fp:  np.ndarray = np.array([])
        self._fpp: np.ndarray = np.array([])
        self._preset_map: dict = {}   # label → (Z, shell)
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        lay = QVBoxLayout(self)

        # ── Element / edge controls ───────────────────────────────────────────
        ctrl = QGroupBox('Resonant element')
        form = QFormLayout(ctrl)

        self._combo_preset = QComboBox()
        self._combo_preset.addItem('(custom)')
        self._combo_preset.addItems(list(_COMMON.keys()))
        self._combo_preset.currentIndexChanged.connect(self._on_preset)
        form.addRow('Preset:', self._combo_preset)

        self._spin_Z = QSpinBox(); self._spin_Z.setRange(1, 100); self._spin_Z.setValue(79)
        form.addRow('Atomic number Z:', self._spin_Z)

        self._combo_shell = QComboBox()
        self._combo_shell.addItems(_SHELLS)
        self._combo_shell.setCurrentText('L3')
        form.addRow('Edge:', self._combo_shell)

        self._lbl_edge_E = QLabel()
        form.addRow('Edge energy:', self._lbl_edge_E)

        self._btn_compute = QPushButton('Compute f\'/f\'\' (NIST/Chantler via xraydb)')
        self._btn_compute.clicked.connect(self._compute)
        form.addRow(self._btn_compute)

        # condition number display
        self._lbl_kappa = QLabel('—')
        form.addRow('Condition number κ:', self._lbl_kappa)
        lay.addWidget(ctrl)

        # ── Plots ─────────────────────────────────────────────────────────────
        plots = QHBoxLayout()

        self._pw_fp = pg.PlotWidget(title="f'(E)  (e)")
        self._pw_fp.setLabel('bottom', 'E (keV)')
        self._pw_fp.setLabel('left', "f' (e)")
        self._curve_fp = self._pw_fp.plot(pen=pg.mkPen('b', width=1.5))
        self._pts_fp   = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush('b'), size=8)
        self._pw_fp.addItem(self._pts_fp)
        self._vline_fp = pg.InfiniteLine(angle=90, pen=pg.mkPen('r', style=pg.QtCore.Qt.PenStyle.DashLine))
        self._pw_fp.addItem(self._vline_fp)
        plots.addWidget(self._pw_fp)

        self._pw_fpp = pg.PlotWidget(title="f''(E)  (e)")
        self._pw_fpp.setLabel('bottom', 'E (keV)')
        self._pw_fpp.setLabel('left', "f'' (e)")
        self._curve_fpp = self._pw_fpp.plot(pen=pg.mkPen('r', width=1.5))
        self._pts_fpp   = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush('r'), size=8)
        self._pw_fpp.addItem(self._pts_fpp)
        self._vline_fpp = pg.InfiniteLine(angle=90, pen=pg.mkPen('r', style=pg.QtCore.Qt.PenStyle.DashLine))
        self._pw_fpp.addItem(self._vline_fpp)
        plots.addWidget(self._pw_fpp)

        lay.addLayout(plots)
        self._update_edge_label()
        self._spin_Z.valueChanged.connect(self._update_edge_label)
        self._combo_shell.currentIndexChanged.connect(self._update_edge_label)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_preset(self, idx):
        if idx == 0:
            return
        name = self._combo_preset.currentText()
        if name in self._preset_map:
            Z, shell = self._preset_map[name]
        elif name in _COMMON:
            Z = _COMMON[name]
            shell = 'L3' if Z > 50 else 'K'
        else:
            return
        self._spin_Z.setValue(Z)
        self._combo_shell.setCurrentText(shell)

    def _update_presets(self):
        """Rebuild preset list: edges within (or just outside) the data energy range."""
        if not _HAS_XRAYDB or len(self._energies) == 0:
            return
        e_min, e_max = float(self._energies.min()), float(self._energies.max())
        margin = 0.3  # keV — include edges slightly outside the scan range

        candidates = []
        for Z in range(10, 93):
            try:
                sym = xraydb.atomic_symbol(Z)
            except Exception:
                continue
            for shell in _SHELLS:
                e_edge = _edge_energy_keV(Z, shell)
                if e_edge <= 0:
                    continue
                if not (e_min - margin <= e_edge <= e_max + margin):
                    continue
                above = int(np.sum(self._energies > e_edge))
                below = int(np.sum(self._energies < e_edge))
                # Score: prefer edges well-bracketed by data on both sides
                score = min(above, below)
                label = f'{sym} ({Z}) — {shell}  [{e_edge:.4f} keV]'
                candidates.append((score, Z, shell, label))

        # Best bracketing first, then by Z
        candidates.sort(key=lambda x: (-x[0], x[1]))

        self._preset_map = {label: (Z, shell) for _, Z, shell, label in candidates}

        self._combo_preset.blockSignals(True)
        self._combo_preset.clear()
        self._combo_preset.addItem('(custom)')
        if candidates:
            for _, Z, shell, label in candidates:
                self._combo_preset.addItem(label)
        else:
            # Fallback: no edges found in range — keep static list
            self._combo_preset.addItems(list(_COMMON.keys()))
        self._combo_preset.blockSignals(False)

        # Auto-select the best candidate (highest score)
        if candidates:
            self._combo_preset.setCurrentIndex(1)
            # Trigger Z/shell update without emitting fp_fpp
            _, Z, shell, _ = candidates[0]
            self._spin_Z.blockSignals(True)
            self._combo_shell.blockSignals(True)
            self._spin_Z.setValue(Z)
            self._combo_shell.setCurrentText(shell)
            self._spin_Z.blockSignals(False)
            self._combo_shell.blockSignals(False)
            self._update_edge_label()

    def _update_edge_label(self):
        e = _edge_energy_keV(self._spin_Z.value(), self._combo_shell.currentText())
        self._lbl_edge_E.setText(f'{e:.4f} keV' if e else '—')
        if e:
            self._vline_fp.setValue(e)
            self._vline_fpp.setValue(e)

    def _compute(self):
        if len(self._energies) == 0:
            self._lbl_kappa.setText('No energies loaded — add data first')
            return
        Z = self._spin_Z.value()
        try:
            fp, fpp = _fp_fpp(Z, self._energies)
        except Exception as e:
            self._lbl_kappa.setText(str(e))
            return

        self._fp  = fp
        self._fpp = fpp

        # Dense curve for smooth display
        e_dense = np.linspace(self._energies.min() - 0.2,
                              self._energies.max() + 0.2, 300)
        fp_d, fpp_d = _fp_fpp(Z, e_dense)
        self._curve_fp.setData(e_dense, fp_d)
        self._pts_fp.setData(x=self._energies, y=fp)
        self._curve_fpp.setData(e_dense, fpp_d)
        self._pts_fpp.setData(x=self._energies, y=fpp)

        kappa = condition_number(fp, fpp)
        self._lbl_kappa.setText(f'{kappa:.1f}')
        self.fp_fpp_ready.emit(fp, fpp)

    # ── Public API ────────────────────────────────────────────────────────────
    def set_energies(self, energies: list[float]):
        self._energies = np.array([e for e in energies if e is not None])
        self._update_presets()

    def get_fp_fpp(self) -> tuple[np.ndarray, np.ndarray] | None:
        if len(self._fp) == len(self._energies) and len(self._fp) > 0:
            return self._fp, self._fpp
        return None
