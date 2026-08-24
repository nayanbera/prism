"""
q-space decomposition tab.

Direct mode   — WLS at each q → I_MM, I_RM, I_RR + propagated σ
Difference mode — subtract reference energy E_ref → I_RM, I_RR only
                  (I_MM cancels; more robust, lower κ)
"""

import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QCheckBox, QSplitter, QComboBox, QTabWidget, QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

from .core.crosshair import add_crosshair
from .core.decompose import (
    decompose, decompose_difference, stuhrmann_analysis, condition_number,
)

_COLORS = {'I_MM': '#2196F3', 'I_RM': '#4CAF50', 'I_RR': '#F44336'}


class DecompositionTab(QWidget):
    decomposition_done = pyqtSignal(
        np.ndarray,   # q
        np.ndarray,   # I_MM
        np.ndarray,   # I_RM
        np.ndarray,   # I_RR
        np.ndarray,   # σ_MM
        np.ndarray,   # σ_RM
        np.ndarray,   # σ_RR
    )
    reference_loaded = pyqtSignal(
        np.ndarray,   # q
        np.ndarray,   # I_MM
        np.ndarray,   # I_RM
        np.ndarray,   # I_RR
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._q: np.ndarray | None        = None   # full grid from DataTab
        self._I_matrix: np.ndarray | None = None
        self._sig: np.ndarray | None      = None
        self._fp: np.ndarray | None       = None
        self._fpp: np.ndarray | None      = None
        self._energies: list[float]       = []
        self._result: dict | None         = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        lay = QVBoxLayout(self)

        # ── Top controls ──────────────────────────────────────────────────────
        top = QHBoxLayout()

        self._btn_run = QPushButton('Run decomposition')
        self._btn_run.clicked.connect(self._run)
        top.addWidget(self._btn_run)

        top.addWidget(QLabel('q range:'))
        from PyQt6.QtWidgets import QDoubleSpinBox as _DSB2
        self._spin_qmin = _DSB2(); self._spin_qmin.setDecimals(5)
        self._spin_qmin.setRange(0, 10); self._spin_qmin.setValue(0.0)
        self._spin_qmin.setFixedWidth(80)
        self._spin_qmax = _DSB2(); self._spin_qmax.setDecimals(4)
        self._spin_qmax.setRange(0, 10); self._spin_qmax.setValue(1.0)
        self._spin_qmax.setFixedWidth(75)
        top.addWidget(self._spin_qmin)
        top.addWidget(QLabel('–'))
        top.addWidget(self._spin_qmax)
        top.addWidget(QLabel('Å⁻¹'))

        self._chk_diff = QCheckBox('Difference mode (subtract reference E)')
        top.addWidget(self._chk_diff)

        top.addWidget(QLabel('Ref energy:'))
        self._combo_ref = QComboBox()
        self._combo_ref.setMinimumWidth(160)
        top.addWidget(self._combo_ref)

        self._chk_log = QCheckBox('Log I')
        self._chk_log.setChecked(True)
        self._chk_log.toggled.connect(self._update_log)
        top.addWidget(self._chk_log)

        self._chk_errbar = QCheckBox('Error bars')
        self._chk_errbar.setChecked(True)
        self._chk_errbar.toggled.connect(self._refresh_errorbars)
        top.addWidget(self._chk_errbar)

        top.addStretch()
        self._btn_ref = QPushButton('Load reference partials…')
        self._btn_ref.clicked.connect(self._load_reference)
        top.addWidget(self._btn_ref)
        self._lbl_status = QLabel('Load data and compute f\'/f\'\' first')
        top.addWidget(self._lbl_status)
        lay.addLayout(top)

        # ── Tab: I(q) plot / Stuhrmann ────────────────────────────────────────
        inner = QTabWidget()

        # Partial I(q) plot
        self._pw_iq = pg.PlotWidget(title='Partial structure factors I(q)')
        self._pw_iq.setLabel('bottom', 'q (Å⁻¹)')
        self._pw_iq.setLabel('left', 'I(q) (cm⁻¹)')
        self._pw_iq.addLegend()
        self._pw_iq.setLogMode(x=False, y=True)
        self._curves_iq: dict[str, pg.PlotDataItem] = {}
        self._err_items: dict[str, pg.ErrorBarItem] = {}
        for name, col in _COLORS.items():
            self._curves_iq[name] = self._pw_iq.plot(
                [], [], name=name, pen=pg.mkPen(col, width=2))
            ei = pg.ErrorBarItem(x=np.array([]), y=np.array([]),
                                 top=np.array([]), bottom=np.array([]),
                                 pen=pg.mkPen(col, width=0.8))
            self._pw_iq.addItem(ei)
            self._err_items[name] = ei
        self._raw_curves: list = []
        # Dashed reference overlay — populated by _load_reference()
        self._ref_iq_curves: dict[str, pg.PlotDataItem] = {}
        for name, col in _COLORS.items():
            self._ref_iq_curves[name] = self._pw_iq.plot(
                [], [], name=f'{name} ref',
                pen=pg.mkPen(col, width=1.5,
                             style=pg.QtCore.Qt.PenStyle.DashLine))
        inner.addTab(self._pw_iq, 'Partial I(q)')

        # ── Waterfall plot ────────────────────────────────────────────────────
        wfw = QWidget()
        wflay = QVBoxLayout(wfw)

        wf_ctrl = QHBoxLayout()
        self._chk_wf_log  = QCheckBox('Log I')
        self._chk_wf_log.setChecked(True)
        self._chk_wf_log.toggled.connect(self._update_waterfall)
        self._chk_wf_logq = QCheckBox('Log q')
        self._chk_wf_logq.toggled.connect(self._update_waterfall)
        wf_ctrl.addWidget(self._chk_wf_log)
        wf_ctrl.addWidget(self._chk_wf_logq)
        wf_ctrl.addWidget(QLabel('  Vertical offset factor:'))
        from PyQt6.QtWidgets import QDoubleSpinBox as _DSB
        self._spin_wf_offset = _DSB()
        self._spin_wf_offset.setRange(0.0, 100.0)
        self._spin_wf_offset.setValue(0.0)
        self._spin_wf_offset.setSingleStep(0.5)
        self._spin_wf_offset.setDecimals(1)
        self._spin_wf_offset.setToolTip(
            '0 = overlay; >0 = multiply each curve by offset^k (waterfall)')
        self._spin_wf_offset.valueChanged.connect(self._update_waterfall)
        wf_ctrl.addWidget(self._spin_wf_offset)
        wf_ctrl.addStretch()
        wflay.addLayout(wf_ctrl)

        wf_main = QHBoxLayout()

        self._pw_wf = pg.PlotWidget(title='I(q, E) — colored by energy')
        self._pw_wf.setLabel('bottom', 'q (Å⁻¹)')
        self._pw_wf.setLabel('left', 'I(q) (cm⁻¹)')
        self._pw_wf.setLogMode(x=False, y=True)
        wf_main.addWidget(self._pw_wf)

        # Narrow colorbar using a QLabel with a CSS linear-gradient
        from PyQt6.QtWidgets import QFrame
        cbar_frame = QFrame()
        cbar_frame.setFixedWidth(28)
        cbar_lay = QVBoxLayout(cbar_frame)
        cbar_lay.setContentsMargins(4, 4, 4, 4)
        self._lbl_cbar_hi  = QLabel()   # high energy label
        self._lbl_cbar_lo  = QLabel()   # low energy label
        self._lbl_gradient = QLabel()
        self._lbl_gradient.setFixedWidth(18)
        self._lbl_gradient.setSizePolicy(
            self._lbl_gradient.sizePolicy().horizontalPolicy(),
            __import__('PyQt6.QtWidgets', fromlist=['QSizePolicy']).QSizePolicy.Policy.Expanding)
        self._lbl_gradient.setStyleSheet(
            'background: qlineargradient(x1:0,y1:0,x2:0,y2:1,'
            'stop:0 #ff2222, stop:1 #2244ff);'
            'border-radius: 3px;')
        cbar_lay.addWidget(self._lbl_cbar_hi)
        cbar_lay.addWidget(self._lbl_gradient)
        cbar_lay.addWidget(self._lbl_cbar_lo)
        wf_main.addWidget(cbar_frame)

        wflay.addLayout(wf_main)

        self._wf_curves: list[pg.PlotDataItem] = []

        inner.addTab(wfw, 'Waterfall')

        # Stuhrmann plot
        stw = QWidget()
        slay = QVBoxLayout(stw)

        # Controls row
        st_ctrl = QHBoxLayout()
        from PyQt6.QtWidgets import QSpinBox
        st_ctrl.addWidget(QLabel('q values to show:'))
        self._spin_nq_stuhr = QSpinBox()
        self._spin_nq_stuhr.setRange(2, 20); self._spin_nq_stuhr.setValue(8)
        self._spin_nq_stuhr.valueChanged.connect(self._update_stuhrmann)
        st_ctrl.addWidget(self._spin_nq_stuhr)
        self._chk_stuhr_log = QCheckBox('Log I axis')
        self._chk_stuhr_log.toggled.connect(
            lambda v: self._pw_stuhr.setLogMode(x=False, y=v))
        st_ctrl.addWidget(self._chk_stuhr_log)
        st_ctrl.addStretch()
        slay.addLayout(st_ctrl)

        self._pw_stuhr = pg.PlotWidget(
            title="Stuhrmann: I(q, E) vs f'(E) at representative q values")
        self._pw_stuhr.setLabel('bottom', "f' (e)")
        self._pw_stuhr.setLabel('left', 'I(q, E) (cm⁻¹)')
        self._pw_stuhr.addLegend(offset=(10, 10))
        slay.addWidget(self._pw_stuhr)

        self._lbl_stuhr = QLabel()
        slay.addWidget(self._lbl_stuhr)

        # Dynamic items — rebuilt on each update
        self._stuhr_items: list = []

        inner.addTab(stw, 'Stuhrmann')

        self._coord_lbl = QLabel()
        self._coord_lbl.setStyleSheet('font-family: monospace; color: #555555;')
        self._ch = [add_crosshair(self._pw_iq,    label=self._coord_lbl),
                    add_crosshair(self._pw_wf,     label=self._coord_lbl),
                    add_crosshair(self._pw_stuhr,  label=self._coord_lbl)]
        lay.addWidget(inner)
        lay.addWidget(self._coord_lbl)

    def _load_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load reference partials (q  I_MM  I_RM  I_RR)', '',
            'Data files (*.dat *.txt);;All files (*)')
        if not path:
            return
        try:
            data = np.loadtxt(path, comments='#')
            q_ref, I_MM, I_RM, I_RR = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            for name, arr in [('I_MM', I_MM), ('I_RM', np.abs(I_RM)), ('I_RR', I_RR)]:
                self._ref_iq_curves[name].setData(q_ref, arr)
            self.reference_loaded.emit(q_ref, I_MM, I_RM, I_RR)
            self._lbl_status.setText(f'Reference: {Path(path).name}')
        except Exception as e:
            self._lbl_status.setText(f'Reference load error: {e}')

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _run(self):
        if self._q is None or self._fp is None:
            self._lbl_status.setText('Missing data or f\'/f\'\'')
            return
        if len(self._fp) != self._I_matrix.shape[0]:
            self._lbl_status.setText(
                f'Mismatch: {len(self._fp)} f\' values vs '
                f'{self._I_matrix.shape[0]} datasets')
            return

        diff_mode = self._chk_diff.isChecked()
        ref_idx   = self._combo_ref.currentIndex()

        # Apply q range mask
        qmask = ((self._q >= self._spin_qmin.value()) &
                 (self._q <= self._spin_qmax.value()))
        q   = self._q[qmask]
        I   = self._I_matrix[:, qmask]
        sig = self._sig[:, qmask]

        if len(q) < 3:
            self._lbl_status.setText('q range too narrow — no data points')
            return

        self._lbl_status.setText('Running…')
        try:
            if diff_mode:
                I_RM, I_RR, s_RM, s_RR = decompose_difference(
                    q, I, sig, self._fp, self._fpp, ref_idx)
                I_MM = np.zeros_like(I_RM)
                s_MM = np.zeros_like(I_RM)
            else:
                I_MM, I_RM, I_RR, s_MM, s_RM, s_RR = decompose(
                    q, I, sig, self._fp, self._fpp)
        except Exception as e:
            self._lbl_status.setText(f'Error: {e}')
            return

        self._result = dict(
            q=q,
            I_MM=I_MM, I_RM=I_RM, I_RR=I_RR,
            s_MM=s_MM, s_RM=s_RM, s_RR=s_RR,
            diff_mode=diff_mode,
        )

        self._update_iq_plot()
        self._update_stuhrmann()
        self.decomposition_done.emit(
            self._q, I_MM, I_RM, I_RR, s_MM, s_RM, s_RR)
        mode = 'difference' if diff_mode else 'direct'
        self._lbl_status.setText(
            f'Done [{mode}] — {self._I_matrix.shape[0]} energies, '
            f'{len(self._q)} q-points  |  '
            f'κ = {condition_number(self._fp, self._fpp):.1f}')

    def _update_iq_plot(self):
        r = self._result
        q = r['q']

        # Raw waterfall (gray)
        for c in self._raw_curves:
            self._pw_iq.removeItem(c)
        self._raw_curves.clear()
        for row in self._I_matrix:
            c = self._pw_iq.plot(q, row,
                pen=pg.mkPen((160, 160, 160, 60), width=0.8))
            self._raw_curves.append(c)

        for name, key_I, key_s in [
            ('I_MM', 'I_MM', 's_MM'),
            ('I_RM', 'I_RM', 's_RM'),
            ('I_RR', 'I_RR', 's_RR'),
        ]:
            Iq = r[key_I]
            si = r[key_s]
            self._curves_iq[name].setData(q, np.maximum(np.abs(Iq), 1e-40))
            self._refresh_errorbars()

    def _refresh_errorbars(self):
        if self._result is None:
            return
        r    = self._result
        q    = r['q']
        show = self._chk_errbar.isChecked()
        log  = self._chk_log.isChecked()
        for name, key_I, key_s in [
            ('I_MM', 'I_MM', 's_MM'),
            ('I_RM', 'I_RM', 's_RM'),
            ('I_RR', 'I_RR', 's_RR'),
        ]:
            Iq = np.maximum(np.abs(r[key_I]), 1e-40)
            si = r[key_s]
            if show and not log:
                self._err_items[name].setData(
                    x=q, y=Iq, top=si, bottom=si)
            else:
                self._err_items[name].setData(
                    x=np.array([]), y=np.array([]),
                    top=np.array([]), bottom=np.array([]))

    def _update_log(self, log: bool):
        self._pw_iq.setLogMode(x=False, y=log)
        self._refresh_errorbars()

    def _update_waterfall(self):
        if self._q is None or self._I_matrix is None:
            return

        # Clear old curves
        for c in self._wf_curves:
            self._pw_wf.removeItem(c)
        self._wf_curves.clear()

        log_y  = self._chk_wf_log.isChecked()
        log_q  = self._chk_wf_logq.isChecked()
        offset = self._spin_wf_offset.value()

        self._pw_wf.setLogMode(x=log_q, y=log_y)

        N_E      = self._I_matrix.shape[0]
        energies = self._energies if self._energies else list(range(N_E))
        E_arr    = np.array([e if e is not None else i
                             for i, e in enumerate(energies)], dtype=float)
        E_min, E_max = E_arr.min(), E_arr.max()
        E_span = E_max - E_min if E_max > E_min else 1.0

        def _energy_color(t: float) -> pg.QtGui.QColor:
            """Blue (t=0, low E) → cyan → green → yellow → red (t=1, high E)."""
            # 4-stop gradient: blue → cyan → green → yellow → red
            stops = [(0.0, (0, 0, 255)), (0.33, (0, 200, 255)),
                     (0.5, (0, 220, 0)), (0.67, (255, 220, 0)), (1.0, (255, 0, 0))]
            for i in range(len(stops) - 1):
                t0, c0 = stops[i]; t1, c1 = stops[i + 1]
                if t0 <= t <= t1:
                    f = (t - t0) / (t1 - t0)
                    r = int(c0[0] + f * (c1[0] - c0[0]))
                    g = int(c0[1] + f * (c1[1] - c0[1]))
                    b = int(c0[2] + f * (c1[2] - c0[2]))
                    return pg.QtGui.QColor(r, g, b)
            return pg.QtGui.QColor(255, 0, 0)

        for k in range(N_E):
            t   = (E_arr[k] - E_min) / E_span          # 0→1
            col = _energy_color(t)
            Iq  = self._I_matrix[k].copy()
            if offset > 0:
                Iq = Iq * (offset ** k)
            label = f'{E_arr[k]:.3f} keV' if k in (0, N_E - 1) else None
            mask = Iq > 0
            if not mask.any():
                continue
            c = self._pw_wf.plot(
                self._q[mask], Iq[mask],
                pen=pg.mkPen(col, width=1.5),
                name=label,
            )
            self._wf_curves.append(c)

        # Update gradient label energy annotations
        self._lbl_cbar_hi.setText(f'{E_max:.3f}')
        self._lbl_cbar_lo.setText(f'{E_min:.3f}')

    def _update_stuhrmann(self):
        if self._q is None or self._fp is None or self._I_matrix is None:
            return

        # Clear previous items
        for item in self._stuhr_items:
            self._pw_stuhr.removeItem(item)
        self._stuhr_items.clear()
        self._pw_stuhr.getPlotItem().legend.clear()

        N_q   = self._spin_nq_stuhr.value()
        NQ    = len(self._q)
        # Indices spread evenly across q range (skip first/last 2% to avoid edge noise)
        lo, hi = max(0, NQ // 50), min(NQ - 1, NQ - NQ // 50)
        q_idxs = np.linspace(lo, hi, N_q, dtype=int)

        # Color palette: low q = blue, high q = red
        cmap_colors = [
            pg.mkColor(int(255 * k / (N_q - 1)), 0, int(255 * (1 - k / (N_q - 1))))
            for k in range(N_q)
        ]

        fp  = self._fp
        fpp = self._fpp
        # Dense f' for smooth fit curves
        fp_dense = np.linspace(fp.min() - 0.5, fp.max() + 0.5, 300)

        from .core.decompose import build_A
        A = build_A(fp, fpp)

        for k, j in enumerate(q_idxs):
            col  = cmap_colors[k]
            q_j  = self._q[j]
            Iq_j = self._I_matrix[:, j]
            sig_j = self._sig[:, j]

            # WLS fit at this q: recover [I_MM_j, I_RM_j, I_RR_j]
            w    = 1.0 / np.maximum(sig_j, 1e-30)
            Aw   = A * w[:, None]
            bw   = Iq_j * w
            from numpy.linalg import lstsq
            sol, *_ = lstsq(Aw, bw, rcond=None)
            I_MM_j, I_RM_j, I_RR_j = sol

            # Fit curve over dense f' grid
            # Need fpp at fp_dense — interpolate from known fp/fpp
            fpp_dense = np.interp(fp_dense, fp, fpp)
            I_fit = I_MM_j + 2 * fp_dense * I_RM_j + (fp_dense**2 + fpp_dense**2) * I_RR_j

            label = f'q={q_j:.3f} Å⁻¹'

            # Scatter: I(q_j, E) vs f'(E)
            pts = pg.ScatterPlotItem(
                x=fp, y=Iq_j,
                size=8, pen=pg.mkPen(col), brush=pg.mkBrush(col),
                name=label)
            self._pw_stuhr.addItem(pts)
            self._stuhr_items.append(pts)

            # Error bars
            ei = pg.ErrorBarItem(
                x=fp, y=Iq_j, top=sig_j, bottom=sig_j,
                pen=pg.mkPen(col, width=0.8))
            self._pw_stuhr.addItem(ei)
            self._stuhr_items.append(ei)

            # Fit line (dashed)
            fit_curve = pg.PlotDataItem(
                fp_dense, I_fit,
                pen=pg.mkPen(col, width=1.5,
                             style=pg.QtCore.Qt.PenStyle.DashLine))
            self._pw_stuhr.addItem(fit_curve)
            self._stuhr_items.append(fit_curve)

        # Stuhrmann analysis at q[0] for the summary readout
        try:
            st = stuhrmann_analysis(
                self._q, self._I_matrix, self._sig, self._fp, self._fpp)
            self._lbl_stuhr.setText(
                f"At q≈0:  "
                f"I_MM(0) = {st['I_MM_0']:.4g} ± {st['sigma_MM_0']:.2g}   "
                f"I_RM(0) = {st['I_RM_0']:.4g} ± {st['sigma_RM_0']:.2g}   "
                f"I_RR(0) = {st['I_RR_0']:.4g} ± {st['sigma_RR_0']:.2g}  cm⁻¹"
                f"  |  κ = {st['kappa']:.1f}")
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────
    def set_data(self, q: np.ndarray, I_matrix: np.ndarray,
                 sig_matrix: np.ndarray, energies: list[float] | None = None):
        self._q        = q
        self._I_matrix = I_matrix
        self._sig      = sig_matrix
        if energies:
            self._energies = energies
            self._combo_ref.clear()
            for e in energies:
                self._combo_ref.addItem(f'{e:.4f} keV')
        # Auto-set q range spinboxes to full data extent
        for sp in (self._spin_qmin, self._spin_qmax):
            sp.blockSignals(True)
        self._spin_qmin.setValue(float(q[0]))
        self._spin_qmax.setValue(float(q[-1]))
        for sp in (self._spin_qmin, self._spin_qmax):
            sp.blockSignals(False)
        self._update_waterfall()

    def set_fp_fpp(self, fp: np.ndarray, fpp: np.ndarray):
        self._fp  = fp
        self._fpp = fpp
        self._update_stuhrmann()   # refresh as soon as f'/f'' land

    def get_result(self) -> dict | None:
        return self._result
