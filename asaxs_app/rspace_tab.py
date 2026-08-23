"""
r-space decomposition tab.

Workflow: IFT each I(q, E_i) → p(r, E_i), then decompose the p(r,E) stack
at each r via WLS.  Avoids the q-space form-factor-minimum dip in I_RR.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QDoubleSpinBox, QCheckBox, QFormLayout, QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal  # noqa: F401
import pyqtgraph as pg

from .core.ift import ift_then_decompose, _sinc_kernel

_COLORS = {'p_MM': '#2196F3', 'p_RM': '#4CAF50', 'p_RR': '#F44336'}


class RSpaceTab(QWidget):
    rspace_done = pyqtSignal(dict)   # dict: r, p_MM..., sp_MM...

    def __init__(self, parent=None):
        super().__init__(parent)
        self._q: np.ndarray | None        = None
        self._I_matrix: np.ndarray | None = None
        self._sig: np.ndarray | None      = None
        self._fp: np.ndarray | None       = None
        self._fpp: np.ndarray | None      = None
        self._energies: list[float]       = []
        self._result: dict | None         = None
        self._wf_curves: list             = []
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        lay = QVBoxLayout(self)

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl = QGroupBox('IFT parameters (shared across all energies)')
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

        alpha_row = QHBoxLayout()
        self._chk_auto_alpha = QCheckBox('Auto α (GCV, shared)')
        self._chk_auto_alpha.setChecked(True)
        self._spin_alpha = QDoubleSpinBox()
        self._spin_alpha.setDecimals(4); self._spin_alpha.setRange(1e-6, 1e12)
        self._spin_alpha.setValue(1.0); self._spin_alpha.setEnabled(False)
        self._chk_auto_alpha.toggled.connect(
            lambda v: self._spin_alpha.setEnabled(not v))
        alpha_row.addWidget(self._chk_auto_alpha)
        alpha_row.addWidget(QLabel('α ='))
        alpha_row.addWidget(self._spin_alpha)
        alpha_row.addStretch()
        cform.addRow('Regularisation:', alpha_row)

        qrange_row = QHBoxLayout()
        self._spin_qmin = QDoubleSpinBox(); self._spin_qmin.setDecimals(5)
        self._spin_qmin.setRange(0, 10); self._spin_qmin.setValue(0.0)
        self._spin_qmin.setFixedWidth(85)
        self._spin_qmax = QDoubleSpinBox(); self._spin_qmax.setDecimals(4)
        self._spin_qmax.setRange(0, 10); self._spin_qmax.setValue(1.0)
        self._spin_qmax.setFixedWidth(80)
        qrange_row.addWidget(self._spin_qmin)
        qrange_row.addWidget(QLabel('–'))
        qrange_row.addWidget(self._spin_qmax)
        qrange_row.addWidget(QLabel('Å⁻¹'))
        qrange_row.addStretch()
        cform.addRow('q range for IFT:', qrange_row)

        smooth_row = QHBoxLayout()
        self._spin_smooth = QDoubleSpinBox()
        self._spin_smooth.setDecimals(0)
        self._spin_smooth.setRange(0, 99)
        self._spin_smooth.setValue(0)
        self._spin_smooth.setToolTip(
            '0 = off; odd integer = Savitzky-Golay window on decomposed components')
        self._spin_smooth.setSuffix(' pts')
        smooth_row.addWidget(QLabel('SG smooth:'))
        smooth_row.addWidget(self._spin_smooth)
        smooth_row.addStretch()
        cform.addRow('Smoothing:', smooth_row)

        self._chk_bands = QCheckBox('Show σ bands')
        self._chk_bands.setChecked(True)
        self._chk_bands.toggled.connect(self._refresh_bands)
        cform.addRow(self._chk_bands)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton('Run IFT → Decompose')
        self._btn_run.clicked.connect(self._run)
        self._lbl_status = QLabel('Load data and compute f\'/f\'\' first')
        btn_row.addWidget(self._btn_run)
        btn_row.addWidget(self._lbl_status)
        btn_row.addStretch()
        cform.addRow(btn_row)

        lay.addWidget(ctrl)

        # ── Inner tabs ────────────────────────────────────────────────────────
        tabs = QTabWidget()

        # Tab 1: individual p(r,E) waterfall
        wfw = QWidget()
        wflay = QVBoxLayout(wfw)
        wf_ctrl = QHBoxLayout()
        self._chk_wf_offset = QCheckBox('Offset curves')
        self._chk_wf_offset.toggled.connect(self._update_pr_waterfall)
        wf_ctrl.addWidget(self._chk_wf_offset)
        wf_ctrl.addStretch()
        wflay.addLayout(wf_ctrl)
        self._pw_wf = pg.PlotWidget(title='Individual p(r, E) — colored by energy')
        self._pw_wf.setLabel('bottom', 'r (Å)')
        self._pw_wf.setLabel('left', 'p(r, E)')
        wflay.addWidget(self._pw_wf)
        tabs.addTab(wfw, 'p(r, E) waterfall')

        # Tab 2: partial p(r)
        self._pw_pr = pg.PlotWidget(title='Partial PDDFs from r-space decomposition')
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
            fb = pg.FillBetweenItem(up, lo, brush=pg.mkBrush(col + '40'))
            self._pw_pr.addItem(fb)
            self._upper_pr[name] = up
            self._lower_pr[name] = lo
            self._bands_pr[name] = fb
        tabs.addTab(self._pw_pr, 'Partial p(r)')

        # Tab 3: back-transform
        self._pw_fit = pg.PlotWidget(title='Back-transform: K·p(r) vs I(q)')
        self._pw_fit.setLabel('bottom', 'q (Å⁻¹)')
        self._pw_fit.setLabel('left', '|I(q)| (cm⁻¹)')
        self._pw_fit.setLogMode(x=False, y=True)
        self._pw_fit.addLegend()
        self._fit_curves: list = []
        tabs.addTab(self._pw_fit, 'Back-transform')

        # Tab 4: r-space Stuhrmann
        stw = QWidget()
        slay = QVBoxLayout(stw)
        st_ctrl = QHBoxLayout()
        from PyQt6.QtWidgets import QSpinBox
        st_ctrl.addWidget(QLabel('r values to show:'))
        self._spin_nr_stuhr = QSpinBox()
        self._spin_nr_stuhr.setRange(2, 20); self._spin_nr_stuhr.setValue(8)
        self._spin_nr_stuhr.valueChanged.connect(self._update_stuhrmann)
        st_ctrl.addWidget(self._spin_nr_stuhr)
        self._chk_stuhr_log = QCheckBox('Log p axis')
        self._chk_stuhr_log.toggled.connect(
            lambda v: self._pw_stuhr.setLogMode(x=False, y=v))
        st_ctrl.addWidget(self._chk_stuhr_log)
        st_ctrl.addStretch()
        slay.addLayout(st_ctrl)

        self._pw_stuhr = pg.PlotWidget(
            title="Stuhrmann (r-space): p(r, E) vs f'(E) at representative r values")
        self._pw_stuhr.setLabel('bottom', "f' (e)")
        self._pw_stuhr.setLabel('left', 'p(r, E) (cm⁻¹ Å⁻¹)')
        self._pw_stuhr.addLegend(offset=(10, 10))
        slay.addWidget(self._pw_stuhr)

        self._lbl_stuhr_r = QLabel()
        slay.addWidget(self._lbl_stuhr_r)

        self._stuhr_items: list = []
        tabs.addTab(stw, 'Stuhrmann')

        lay.addWidget(tabs)

    # ── Run ───────────────────────────────────────────────────────────────────
    def _run(self):
        if self._q is None or self._fp is None:
            self._lbl_status.setText('Missing data or f\'/f\'\'')
            return
        if len(self._fp) != self._I_matrix.shape[0]:
            self._lbl_status.setText(
                f'Mismatch: {len(self._fp)} energies vs '
                f'{self._I_matrix.shape[0]} datasets')
            return

        Dmax   = self._spin_dmax.value()
        Nr     = int(self._spin_nr.value())
        alpha  = None if self._chk_auto_alpha.isChecked() else self._spin_alpha.value()
        smooth = int(self._spin_smooth.value())

        # Apply q range mask
        qmask = ((self._q >= self._spin_qmin.value()) &
                 (self._q <= self._spin_qmax.value()))
        q_use   = self._q[qmask]
        I_use   = self._I_matrix[:, qmask]
        sig_use = self._sig[:, qmask]

        if len(q_use) < 5:
            self._lbl_status.setText('q range too narrow — no data points')
            return

        n = self._I_matrix.shape[0]
        self._lbl_status.setText(f'IFT of {n} datasets…')

        def _progress(i, total):
            self._lbl_status.setText(f'IFT {i}/{total}…')
            # Force Qt to process events so label updates are visible
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()

        try:
            result = ift_then_decompose(
                q_use, I_use, sig_use,
                self._fp, self._fpp,
                Dmax=Dmax, Nr=Nr, alpha=alpha,
                smooth_window=smooth,
                progress_cb=_progress,
            )
        except Exception as e:
            self._lbl_status.setText(f'Error: {e}')
            raise

        self._result = result
        self._update_pr_waterfall()
        self._update_partial_pr(result)
        self._update_backtransform(result)
        self._update_stuhrmann()
        self.rspace_done.emit(result)
        self._lbl_status.setText(
            f'Done — D_max={Dmax:.0f} Å, N_r={Nr}, '
            f'α={result.get("alpha_used", alpha):.4g}'
            if alpha else
            f'Done — D_max={Dmax:.0f} Å, N_r={Nr}, α auto')

    # ── Plot helpers ──────────────────────────────────────────────────────────
    def _energy_color(self, t: float) -> pg.QtGui.QColor:
        stops = [(0.0, (0, 0, 255)), (0.33, (0, 200, 255)),
                 (0.5, (0, 220, 0)), (0.67, (255, 220, 0)), (1.0, (255, 0, 0))]
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]; t1, c1 = stops[i + 1]
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0)
                return pg.QtGui.QColor(
                    int(c0[0] + f * (c1[0] - c0[0])),
                    int(c0[1] + f * (c1[1] - c0[1])),
                    int(c0[2] + f * (c1[2] - c0[2])))
        return pg.QtGui.QColor(255, 0, 0)

    def _update_pr_waterfall(self):
        if self._result is None:
            return
        for c in self._wf_curves:
            self._pw_wf.removeItem(c)
        self._wf_curves.clear()

        r        = self._result['r']
        P_matrix = self._result['P_matrix']
        N_E      = P_matrix.shape[0]
        energies = self._energies if self._energies else list(range(N_E))
        E_arr    = np.array([e if e is not None else i
                             for i, e in enumerate(energies)], dtype=float)
        E_min, E_max = E_arr.min(), E_arr.max()
        E_span = E_max - E_min if E_max > E_min else 1.0
        offset = self._chk_wf_offset.isChecked()

        p_max = P_matrix.max()
        for k in range(N_E):
            t   = (E_arr[k] - E_min) / E_span
            col = self._energy_color(t)
            pr  = P_matrix[k].copy()
            if offset:
                pr = pr + k * p_max * 0.3
            label = f'{E_arr[k]:.3f} keV' if k in (0, N_E - 1) else None
            c = self._pw_wf.plot(r, pr,
                pen=pg.mkPen(col, width=1.5), name=label)
            self._wf_curves.append(c)

    def _update_partial_pr(self, res: dict):
        r = res['r']
        legend = self._pw_pr.getPlotItem().legend
        legend.clear()

        for name, key_p, key_s in [
            ('p_MM', 'p_MM', 'sp_MM'),
            ('p_RM', 'p_RM', 'sp_RM'),
            ('p_RR', 'p_RR', 'sp_RR'),
        ]:
            p  = res[key_p]
            sp = res[key_s]
            scale = np.max(np.abs(p)) if np.any(p != 0) else 1.0
            if scale == 0:
                scale = 1.0
            p_norm  = p  / scale
            sp_norm = sp / scale

            self._curves_pr[name].setData(r, p_norm)
            self._upper_pr[name].setData(r, p_norm + sp_norm)
            self._lower_pr[name].setData(r, p_norm - sp_norm)

            # Update legend label with the scale factor
            label = f'{name}  (×{scale:.3g})'
            legend.addItem(self._curves_pr[name], label)

        self._pw_pr.setLabel('left', 'p(r) / max  (normalised)')
        self._refresh_bands()

    def _refresh_bands(self):
        show = self._chk_bands.isChecked()
        for fb in self._bands_pr.values():
            fb.setVisible(show)

    def _update_backtransform(self, res: dict):
        for c in self._fit_curves:
            self._pw_fit.removeItem(c)
        self._fit_curves.clear()

        r  = res['r']
        dr = r[1] - r[0]
        K  = _sinc_kernel(self._q, r, dr)

        # Show mean measured I(q) as reference
        I_mean = self._I_matrix.mean(axis=0)
        dc = self._pw_fit.plot(
            self._q, np.maximum(np.abs(I_mean), 1e-40),
            pen=pg.mkPen('w', width=1, style=Qt.PenStyle.DotLine),
            name='I(q) mean')
        self._fit_curves.append(dc)

        for name, key_p, col in [
            ('p_MM', 'p_MM', '#2196F3'),
            ('p_RM', 'p_RM', '#4CAF50'),
            ('p_RR', 'p_RR', '#F44336'),
        ]:
            Kp = K @ res[key_p]
            mask = np.abs(Kp) > 0
            if mask.any():
                fc = self._pw_fit.plot(
                    self._q[mask], np.abs(Kp[mask]),
                    pen=pg.mkPen(col, width=1.5,
                                 style=Qt.PenStyle.DashLine),
                    name=f'K·{name}')
                self._fit_curves.append(fc)

    def _update_stuhrmann(self):
        """p(r_j, E) vs f'(E) at representative r values, with quadratic fits."""
        if self._result is None or self._fp is None:
            return

        for item in self._stuhr_items:
            self._pw_stuhr.removeItem(item)
        self._stuhr_items.clear()
        self._pw_stuhr.getPlotItem().legend.clear()

        res  = self._result
        r    = res['r']
        Nr   = len(r)
        fp   = self._fp
        fpp  = self._fpp
        N_r  = self._spin_nr_stuhr.value()

        # Representative r indices — skip first and last 5% to avoid edge noise
        lo_i = max(0, Nr // 20)
        hi_i = min(Nr - 1, Nr - Nr // 20)
        r_idxs = np.linspace(lo_i, hi_i, N_r, dtype=int)

        # Color: small r = blue, large r = red
        cmap = [
            pg.mkColor(int(255 * k / (N_r - 1)), 0, int(255 * (1 - k / (N_r - 1))))
            for k in range(N_r)
        ]

        fp_dense   = np.linspace(fp.min() - 0.5, fp.max() + 0.5, 300)
        fpp_dense  = np.interp(fp_dense, fp, fpp)

        for k, j in enumerate(r_idxs):
            col   = cmap[k]
            r_j   = r[j]
            pr_j  = res['P_matrix'][:, j]   # p(r_j, E_i) for all energies
            sp_j  = res['SP_matrix'][:, j]

            # Quadratic fit using the decomposed components at this r
            p_MM_j = res['p_MM'][j]
            p_RM_j = res['p_RM'][j]
            p_RR_j = res['p_RR'][j]
            p_fit  = (p_MM_j
                      + 2.0 * fp_dense * p_RM_j
                      + (fp_dense**2 + fpp_dense**2) * p_RR_j)

            label = f'r={r_j:.0f} Å'

            pts = pg.ScatterPlotItem(
                x=fp, y=pr_j, size=8,
                pen=pg.mkPen(col), brush=pg.mkBrush(col),
                name=label)
            self._pw_stuhr.addItem(pts)
            self._stuhr_items.append(pts)

            ei = pg.ErrorBarItem(
                x=fp, y=pr_j, top=sp_j, bottom=sp_j,
                pen=pg.mkPen(col, width=0.8))
            self._pw_stuhr.addItem(ei)
            self._stuhr_items.append(ei)

            fit_c = pg.PlotDataItem(
                fp_dense, p_fit,
                pen=pg.mkPen(col, width=1.5,
                             style=pg.QtCore.Qt.PenStyle.DashLine))
            self._pw_stuhr.addItem(fit_c)
            self._stuhr_items.append(fit_c)

        # Summary readout at r_max / 2
        mid_j = r_idxs[N_r // 2]
        self._lbl_stuhr_r.setText(
            f'At r≈{r[mid_j]:.0f} Å:  '
            f'p_MM = {res["p_MM"][mid_j]:.4g}   '
            f'p_RM = {res["p_RM"][mid_j]:.4g}   '
            f'p_RR = {res["p_RR"][mid_j]:.4g}  cm⁻¹ Å⁻¹')

    # ── Public API ────────────────────────────────────────────────────────────
    def set_data(self, q: np.ndarray, I_matrix: np.ndarray,
                 sig_matrix: np.ndarray, energies: list[float] | None = None):
        self._q        = q
        self._I_matrix = I_matrix
        self._sig      = sig_matrix
        if energies:
            self._energies = energies
        if len(q) > 0:
            dmax_guess = np.pi / q[0] * 2
            self._spin_dmax.setValue(round(dmax_guess / 50) * 50)
            for sp in (self._spin_qmin, self._spin_qmax):
                sp.blockSignals(True)
            self._spin_qmin.setValue(float(q[0]))
            self._spin_qmax.setValue(float(q[-1]))
            for sp in (self._spin_qmin, self._spin_qmax):
                sp.blockSignals(False)

    def set_fp_fpp(self, fp: np.ndarray, fpp: np.ndarray):
        self._fp  = fp
        self._fpp = fpp
        if self._q is not None:
            self._lbl_status.setText('Ready — adjust D_max and click Run')
        if self._result is not None:
            self._update_stuhrmann()

    def get_result(self) -> dict | None:
        return self._result
