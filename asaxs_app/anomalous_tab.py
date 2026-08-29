"""Anomalous scattering factors tab — f'/f'' plots (element selection lives in Data tab)."""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
import pyqtgraph as pg

try:
    import xraydb
    _HAS_XRAYDB = True
except ImportError:
    _HAS_XRAYDB = False

from .core.crosshair import add_crosshair

_SHELLS = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5']

_COMMON = {
    'Au (79) — L edges': 79, 'Pt (78) — L edges': 78,
    'Se (34) — K edge':  34, 'Br (35) — K edge':  35,
    'Sr (38) — K edge':  38, 'Ge (32) — K edge':  32,
    'Fe (26) — K edge':  26, 'Zn (30) — K edge':  30,
    'Cu (29) — K edge':  29, 'Ni (28) — K edge':  28,
}


def _fp_fpp(Z: int, energies_keV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not _HAS_XRAYDB:
        raise RuntimeError('xraydb not installed — run: pip install xraydb')
    e_eV = energies_keV * 1000.0
    fp  = np.array([xraydb.f1_chantler(Z, e) for e in e_eV], dtype=float)
    fpp = np.array([abs(xraydb.f2_chantler(Z, e)) for e in e_eV], dtype=float)
    return fp, fpp


def _edge_energy_keV(Z: int, shell: str) -> float:
    if not _HAS_XRAYDB:
        return 0.0
    try:
        return float(xraydb.xray_edge(Z, shell).energy) / 1000.0
    except Exception:
        return 0.0


# Per-element color scheme: (dense_curve_color, scatter_color)
_ELEM_COLORS = [
    ('b',       'darkblue'),    # Element 1: blue
    ('orange',  'darkorange'),  # Element 2: orange
    ('g',       'darkgreen'),   # Element 3: green
]


class AnomalousTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Per-element curve/scatter items — rebuilt in display()
        self._elem_curves_fp:  list = []
        self._elem_pts_fp:     list = []
        self._elem_curves_fpp: list = []
        self._elem_pts_fpp:    list = []
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        plots = QHBoxLayout()

        self._pw_fp = pg.PlotWidget(title="f'(E)  (e)")
        self._pw_fp.setLabel('bottom', 'E (keV)')
        self._pw_fp.setLabel('left', "f' (e)")
        self._pw_fp.addLegend(offset=(10, 10))
        self._vline_fp = pg.InfiniteLine(
            angle=90, pen=pg.mkPen('r', style=pg.QtCore.Qt.PenStyle.DashLine))
        self._pw_fp.addItem(self._vline_fp)
        plots.addWidget(self._pw_fp)

        self._pw_fpp = pg.PlotWidget(title="f''(E)  (e)")
        self._pw_fpp.setLabel('bottom', 'E (keV)')
        self._pw_fpp.setLabel('left', "f'' (e)")
        self._pw_fpp.addLegend(offset=(10, 10))
        self._vline_fpp = pg.InfiniteLine(
            angle=90, pen=pg.mkPen('r', style=pg.QtCore.Qt.PenStyle.DashLine))
        self._pw_fpp.addItem(self._vline_fpp)
        plots.addWidget(self._pw_fpp)

        lay.addLayout(plots)

        self._coord_lbl = QLabel()
        self._coord_lbl.setStyleSheet('font-family: monospace; color: #555555;')
        lay.addWidget(self._coord_lbl)

        self._ch = [add_crosshair(self._pw_fp,  x_fmt='.4f', y_fmt='.3f',
                                  label=self._coord_lbl),
                    add_crosshair(self._pw_fpp, x_fmt='.4f', y_fmt='.3f',
                                  label=self._coord_lbl)]

    def _clear_elem_items(self):
        for c in self._elem_curves_fp:
            self._pw_fp.removeItem(c)
        for p in self._elem_pts_fp:
            self._pw_fp.removeItem(p)
        for c in self._elem_curves_fpp:
            self._pw_fpp.removeItem(c)
        for p in self._elem_pts_fpp:
            self._pw_fpp.removeItem(p)
        self._elem_curves_fp.clear()
        self._elem_pts_fp.clear()
        self._elem_curves_fpp.clear()
        self._elem_pts_fpp.clear()

    # ── Public API ────────────────────────────────────────────────────────────
    def display(self, energies: np.ndarray, elements):
        """Update plots with pre-computed f'/f'' values.

        elements: list of {'Z': int, 'fp': array, 'fpp': array, 'label': str}
        For backward compat, a single-element call can pass a list with one dict.
        """
        # Legacy call signature: display(energies, Z, fp, fpp)
        if isinstance(elements, int):
            Z   = elements
            fp  = energies  # positional arg shuffle — handled below via *args
            # This branch is never reached from new callers; old callers use new
            # signature via main_window so this is a safety net only.
            return
        if len(energies) == 0 or not elements:
            return

        self._clear_elem_items()

        # Clear legends by removing and re-adding them
        try:
            self._pw_fp.getPlotItem().legend.clear()
        except Exception:
            pass
        try:
            self._pw_fpp.getPlotItem().legend.clear()
        except Exception:
            pass

        e_dense = np.linspace(float(energies.min()) - 0.2,
                              float(energies.max()) + 0.2, 300)

        first_edge_set = False
        for i, elem in enumerate(elements):
            Z   = elem.get('Z', 0)
            fp  = elem['fp']
            fpp = elem['fpp']
            lbl = elem.get('label', f'Elem {i+1}')
            if not lbl:
                lbl = f'Elem {i+1}'

            dense_col, pts_col = _ELEM_COLORS[min(i, len(_ELEM_COLORS)-1)]

            # Dense curve
            try:
                fp_d, fpp_d = _fp_fpp(Z, e_dense)
                c_fp  = self._pw_fp.plot(e_dense,  fp_d,
                                         pen=pg.mkPen(dense_col, width=1.5),
                                         name=lbl)
                c_fpp = self._pw_fpp.plot(e_dense, fpp_d,
                                          pen=pg.mkPen(dense_col, width=1.5),
                                          name=lbl)
            except Exception:
                c_fp  = self._pw_fp.plot(energies, fp,
                                         pen=pg.mkPen(dense_col, width=1.5),
                                         name=lbl)
                c_fpp = self._pw_fpp.plot(energies, fpp,
                                          pen=pg.mkPen(dense_col, width=1.5),
                                          name=lbl)
            self._elem_curves_fp.append(c_fp)
            self._elem_curves_fpp.append(c_fpp)

            # Scatter points at measured energies
            p_fp  = pg.ScatterPlotItem(pen=None,
                                       brush=pg.mkBrush(pts_col), size=8)
            p_fpp = pg.ScatterPlotItem(pen=None,
                                       brush=pg.mkBrush(pts_col), size=8)
            p_fp.setData(x=energies, y=fp)
            p_fpp.setData(x=energies, y=fpp)
            self._pw_fp.addItem(p_fp)
            self._pw_fpp.addItem(p_fpp)
            self._elem_pts_fp.append(p_fp)
            self._elem_pts_fpp.append(p_fpp)

            # Edge vline — use first element's edge
            if not first_edge_set and Z > 0:
                shell = 'L3' if Z > 50 else 'K'
                e_edge = _edge_energy_keV(Z, shell)
                if e_edge:
                    self._vline_fp.setValue(e_edge)
                    self._vline_fpp.setValue(e_edge)
                first_edge_set = True
