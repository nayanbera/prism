"""
IFT tab — GNOM-style IFT of I_MM, I_RM, I_RR → p_MM, p_RM, p_RR
with propagated uncertainty bands on p(r).
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QDoubleSpinBox, QCheckBox, QFormLayout, QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

from .core.crosshair import add_crosshair
from .core.ift import ift_all, _sinc_kernel

_COLORS = {'p_MM': '#2196F3', 'p_RM': '#4CAF50', 'p_RR': '#F44336'}


class _AlphaControl(QGroupBox):
    def __init__(self, label: str, parent=None):
        super().__init__(label, parent)
        lay = QHBoxLayout(self)
        self._chk = QCheckBox('Auto (GCV)')
        self._chk.setChecked(True)
        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(3); self._spin.setRange(1e-6, 1e12)
        self._spin.setValue(1.0); self._spin.setEnabled(False)
        lay.addWidget(self._chk)
        lay.addWidget(QLabel('α ='))
        lay.addWidget(self._spin)
        self._chk.toggled.connect(lambda v: self._spin.setEnabled(not v))

    def alpha(self) -> float | None:
        return None if self._chk.isChecked() else self._spin.value()


class IFTTab(QWidget):
    ift_done = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._q: np.ndarray | None    = None
        self._I_MM = self._I_RM = self._I_RR = None
        self._s_MM = self._s_RM = self._s_RR = None
        self._result: dict | None     = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        lay = QVBoxLayout(self)

        ctrl = QGroupBox('IFT parameters')
        cform = QFormLayout(ctrl)

        self._spin_dmax = QDoubleSpinBox()
        self._spin_dmax.setRange(10, 10000)
        self._spin_dmax.setValue(600)
        self._spin_dmax.setSuffix(' Å')
        cform.addRow('D_max:', self._spin_dmax)

        self._spin_nr = QDoubleSpinBox()
        self._spin_nr.setRange(50, 2000); self._spin_nr.setDecimals(0)
        self._spin_nr.setValue(200)
        cform.addRow('N_r points:', self._spin_nr)

        self._alpha_MM = _AlphaControl('I_MM regularisation')
        self._alpha_RM = _AlphaControl('I_RM regularisation')
        self._alpha_RR = _AlphaControl('I_RR regularisation')
        for w in (self._alpha_MM, self._alpha_RM, self._alpha_RR):
            cform.addRow(w)

        self._chk_bands = QCheckBox('Show σ bands')
        self._chk_bands.setChecked(True)
        self._chk_bands.toggled.connect(self._refresh_bands)
        cform.addRow(self._chk_bands)

        self._btn_run = QPushButton('Run IFT for all components')
        self._btn_run.clicked.connect(self._run)
        self._lbl_status = QLabel('Run decomposition first')
        cform.addRow(self._btn_run, self._lbl_status)

        lay.addWidget(ctrl)

        tabs = QTabWidget()

        # ── p(r) plot ─────────────────────────────────────────────────────────
        self._pw_pr = pg.PlotWidget(title='Partial pair-distance distributions p(r)')
        self._pw_pr.setLabel('bottom', 'r (Å)')
        self._pw_pr.setLabel('left', 'p(r) (cm⁻¹ Å⁻¹)')
        self._pw_pr.addLegend()
        self._pw_pr.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen('gray', width=0.7)))
        self._curves_pr: dict[str, pg.PlotDataItem] = {}
        self._bands_pr:  dict[str, pg.FillBetweenItem] = {}
        self._upper_pr:  dict[str, pg.PlotDataItem] = {}
        self._lower_pr:  dict[str, pg.PlotDataItem] = {}
        for name, col in _COLORS.items():
            self._curves_pr[name] = self._pw_pr.plot(
                [], [], name=name, pen=pg.mkPen(col, width=2))
            up = pg.PlotDataItem(pen=None)
            lo = pg.PlotDataItem(pen=None)
            self._pw_pr.addItem(up); self._pw_pr.addItem(lo)
            fb = pg.FillBetweenItem(up, lo,
                                    brush=pg.mkBrush(col + '40'))
            self._pw_pr.addItem(fb)
            self._upper_pr[name] = up
            self._lower_pr[name] = lo
            self._bands_pr[name] = fb
        tabs.addTab(self._pw_pr, 'p(r)')

        # ── Back-transform check ──────────────────────────────────────────────
        self._pw_fit = pg.PlotWidget(title='Back-transform: K·p(r) vs I(q)')
        self._pw_fit.setLabel('bottom', 'q (Å⁻¹)')
        self._pw_fit.setLabel('left', '|I(q)| (cm⁻¹)')
        self._pw_fit.setLogMode(x=False, y=True)
        self._pw_fit.addLegend()
        self._fit_curves: list = []
        tabs.addTab(self._pw_fit, 'Back-transform')

        self._coord_lbl = QLabel()
        self._coord_lbl.setStyleSheet('font-family: monospace; color: #555555;')
        self._ch = [add_crosshair(self._pw_pr,  label=self._coord_lbl),
                    add_crosshair(self._pw_fit, label=self._coord_lbl)]
        lay.addWidget(tabs)
        lay.addWidget(self._coord_lbl)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _run(self):
        if self._q is None:
            self._lbl_status.setText('No decomposition result yet')
            return

        Dmax = self._spin_dmax.value()
        Nr   = int(self._spin_nr.value())
        self._lbl_status.setText('Running IFT…')

        try:
            result = ift_all(
                self._q,
                self._I_MM, self._I_RM, self._I_RR,
                Dmax, Nr,
                alpha_MM=self._alpha_MM.alpha(),
                alpha_RM=self._alpha_RM.alpha(),
                alpha_RR=self._alpha_RR.alpha(),
                sigma_MM=self._s_MM,
                sigma_RM=self._s_RM,
                sigma_RR=self._s_RR,
            )
        except Exception as e:
            self._lbl_status.setText(f'IFT error: {e}')
            raise

        self._result = result
        self._update_pr_plot(result)
        self._update_backtransform(result, Dmax, Nr)
        self.ift_done.emit(result)
        self._lbl_status.setText(f'Done — D_max={Dmax:.0f} Å, N_r={Nr}')

    def _update_pr_plot(self, res: dict):
        r = res['r']
        for name, key_p, key_s in [
            ('p_MM', 'p_MM', 'sp_MM'),
            ('p_RM', 'p_RM', 'sp_RM'),
            ('p_RR', 'p_RR', 'sp_RR'),
        ]:
            p  = res[key_p]
            sp = res[key_s]
            self._curves_pr[name].setData(r, p)
            self._upper_pr[name].setData(r, p + sp)
            self._lower_pr[name].setData(r, p - sp)
        self._refresh_bands()

    def _refresh_bands(self):
        show = self._chk_bands.isChecked()
        for fb in self._bands_pr.values():
            fb.setVisible(show)

    def _update_backtransform(self, res: dict, Dmax: float, Nr: int):
        for c in self._fit_curves:
            self._pw_fit.removeItem(c)
        self._fit_curves.clear()

        r  = res['r']
        dr = r[1] - r[0]
        K  = _sinc_kernel(self._q, r, dr)

        for name, Iq, pr, col in [
            ('I_MM', self._I_MM, res['p_MM'], '#2196F3'),
            ('I_RM', self._I_RM, res['p_RM'], '#4CAF50'),
            ('I_RR', self._I_RR, res['p_RR'], '#F44336'),
        ]:
            dc = self._pw_fit.plot(
                self._q, np.maximum(np.abs(Iq), 1e-40),
                pen=pg.mkPen(col, width=2), name=f'{name} data')
            fc = self._pw_fit.plot(
                self._q, np.maximum(np.abs(K @ pr), 1e-40),
                pen=pg.mkPen(col, width=1.5,
                             style=Qt.PenStyle.DashLine),
                name=f'{name} fit')
            self._fit_curves += [dc, fc]

    # ── Public API ────────────────────────────────────────────────────────────
    def set_partials(self, q, I_MM, I_RM, I_RR,
                     s_MM=None, s_RM=None, s_RR=None):
        self._q    = q
        self._I_MM = I_MM; self._I_RM = I_RM; self._I_RR = I_RR
        self._s_MM = s_MM; self._s_RM = s_RM; self._s_RR = s_RR
        if len(q) > 0:
            dmax_guess = np.pi / q[0] * 2
            self._spin_dmax.setValue(round(dmax_guess / 50) * 50)
        self._lbl_status.setText('Ready — adjust D_max and click Run IFT')
