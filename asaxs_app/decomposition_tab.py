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
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

from .core.crosshair import add_crosshair
from .core.decompose import (
    decompose, decompose_difference, stuhrmann_analysis, condition_number,
    cauchy_schwarz_ratio, enforce_cauchy_schwarz,
    decompose_multi, build_A_multi, condition_number_multi,
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
        self._fp: np.ndarray | None       = None   # backward compat — elem 0
        self._fpp: np.ndarray | None      = None   # backward compat — elem 0
        self._elements: list[dict]        = []     # multi-element list
        self._energies: list[float]       = []
        self._result: dict | None         = None
        # Stuhrmann multi-element panel state
        self._stuhr_panels:         list  = []
        self._stuhr_vb_resid:       list  = []
        self._stuhr_pts_resid:      list  = []
        self._stuhr_lbls:           list  = []
        self._stuhr_items_by_panel: list  = []   # list[list] — items per panel
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

        self._chk_cs_enforce = QCheckBox('Enforce C-S')
        self._chk_cs_enforce.setToolTip(
            'Clamp I_RM(q) to ±√(I_MM·I_RR) wherever the Cauchy-Schwarz\n'
            'inequality I_RM² ≤ I_MM·I_RR is violated (direct mode only).\n'
            'Re-run decomposition to apply.')
        top.addWidget(self._chk_cs_enforce)

        self._btn_lp = QPushButton('I_RR bounds (LP)')
        self._btn_lp.setToolTip(
            'Compute the upper and lower bounds on I_RR at each q by solving\n'
            'a linear program: largest/smallest I_RR consistent with all\n'
            'measured intensities within 1σ.  Direct mode only.')
        self._btn_lp.clicked.connect(self._run_lp_bounds)
        top.addWidget(self._btn_lp)

        top.addStretch()
        self._btn_ref = QPushButton('Load reference partials…')
        self._btn_ref.clicked.connect(self._load_reference)
        top.addWidget(self._btn_ref)
        self._chk_show_ref = QCheckBox('Show ref')
        self._chk_show_ref.setChecked(True)
        self._chk_show_ref.toggled.connect(self._toggle_ref_visibility)
        top.addWidget(self._chk_show_ref)
        self._lbl_status = QLabel('Load data and compute f\'/f\'\' first')
        self._lbl_status.setSizePolicy(QSizePolicy.Policy.Ignored,
                                       QSizePolicy.Policy.Preferred)
        top.addWidget(self._lbl_status)
        lay.addLayout(top)

        # ── Tab: I(q) plot / Stuhrmann ────────────────────────────────────────
        inner = QTabWidget()

        # Partial I(q) plot
        self._pw_iq = pg.PlotWidget(title='Partial structure factors I(q)')
        self._pw_iq.setLabel('bottom', 'q (Å⁻¹)')
        self._pw_iq.setLabel('left', 'I(q) (cm⁻¹)')
        self._pw_iq.addLegend()
        self._pw_iq.setLogMode(x=True, y=True)
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

        # LP bounds band for I_RR (shown after clicking "I_RR bounds (LP)")
        _lp_pen = pg.mkPen('#ff4444', width=1,
                           style=pg.QtCore.Qt.PenStyle.DotLine)
        self._curve_lp_lo = self._pw_iq.plot([], [], pen=_lp_pen)
        self._curve_lp_hi = self._pw_iq.plot([], [], pen=_lp_pen,
                                              name='I_RR LP bounds')
        self._fill_lp = pg.FillBetweenItem(
            self._curve_lp_lo, self._curve_lp_hi,
            brush=pg.mkBrush(255, 68, 68, 45))
        self._pw_iq.addItem(self._fill_lp)

        # Cauchy-Schwarz diagnostic plot (below I(q))
        self._pw_cs = pg.PlotWidget(
            title='Cauchy-Schwarz ratio  CS(q) = I_RM² / (I_MM · I_RR)')
        self._pw_cs.setLabel('bottom', 'q (Å⁻¹)')
        self._pw_cs.setLabel('left', 'CS(q)')
        self._pw_cs.setLogMode(x=True, y=False)
        self._pw_cs.addItem(pg.InfiniteLine(
            pos=1.0, angle=0,
            pen=pg.mkPen('#ef5350', width=1.5,
                         style=pg.QtCore.Qt.PenStyle.DashLine)))
        self._curve_cs = self._pw_cs.plot(
            [], [], pen=pg.mkPen((180, 180, 180, 120), width=1.0))
        self._pts_cs_ok  = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush('#66bb6a'), size=5)
        self._pts_cs_vio = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush('#ef5350'), size=5)
        self._pw_cs.addItem(self._pts_cs_ok)
        self._pw_cs.addItem(self._pts_cs_vio)

        iq_split = QSplitter(Qt.Orientation.Vertical)
        iq_split.addWidget(self._pw_iq)
        iq_split.addWidget(self._pw_cs)
        iq_split.setSizes([340, 130])
        inner.addTab(iq_split, 'Partial I(q)')


        # Stuhrmann tab — embedded waterfall (left) + Stuhrmann (right) + q slider
        stw = QWidget()
        slay = QVBoxLayout(stw)

        st_ctrl = QHBoxLayout()
        self._chk_stuhr_log = QCheckBox('Log I (Stuhrmann)')
        self._chk_stuhr_log.toggled.connect(self._on_stuhr_log_toggled)
        self._chk_st_logi = QCheckBox('Log I (waterfall)')
        self._chk_st_logi.setChecked(True)
        self._chk_st_logi.toggled.connect(self._update_stuhr_waterfall)
        self._chk_st_logq = QCheckBox('Log q')
        self._chk_st_logq.setChecked(True)
        self._chk_st_logq.toggled.connect(self._on_st_logq_toggled)
        for w in (self._chk_stuhr_log, self._chk_st_logi, self._chk_st_logq):
            st_ctrl.addWidget(w)
        st_ctrl.addStretch()
        slay.addLayout(st_ctrl)

        st_split = QSplitter(Qt.Orientation.Horizontal)

        # Left: embedded waterfall with vertical q-selection line
        self._pw_st_wf = pg.PlotWidget(title='I(q, E)  —  select q with slider')
        self._pw_st_wf.setLabel('bottom', 'q (Å⁻¹)')
        self._pw_st_wf.setLabel('left', 'I(q)  (cm⁻¹)')
        self._pw_st_wf.setLogMode(x=True, y=True)
        self._st_wf_curves: list[pg.PlotDataItem] = []
        self._st_vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#ffffff', width=1.5,
                         style=pg.QtCore.Qt.PenStyle.DashLine))
        self._pw_st_wf.addItem(self._st_vline)
        st_split.addWidget(self._pw_st_wf)

        # Right: vertical splitter of per-element Stuhrmann panels
        self._stuhr_right_split = QSplitter(Qt.Orientation.Vertical)
        st_split.addWidget(self._stuhr_right_split)
        st_split.setSizes([500, 500])
        slay.addWidget(st_split, 1)

        # Q slider
        from PyQt6.QtWidgets import QSlider
        sl_row = QHBoxLayout()
        sl_row.addWidget(QLabel('q  ='))
        self._sl_q = QSlider(Qt.Orientation.Horizontal)
        self._sl_q.setMinimum(0)
        self._sl_q.setMaximum(0)
        self._sl_q.setValue(0)
        self._sl_q.valueChanged.connect(self._on_q_slider)
        sl_row.addWidget(self._sl_q, 1)
        self._lbl_q_val = QLabel('—')
        self._lbl_q_val.setFixedWidth(170)
        sl_row.addWidget(self._lbl_q_val)
        slay.addLayout(sl_row)

        self._stuhr_items: list = []
        self._stuhr_panel_crosshairs: list = []
        inner.addTab(stw, 'Stuhrmann')

        self._coord_lbl = QLabel()
        self._coord_lbl.setStyleSheet('font-family: monospace; color: #555555;')
        self._coord_lbl.setSizePolicy(QSizePolicy.Policy.Ignored,
                                      QSizePolicy.Policy.Preferred)
        self._ch = [add_crosshair(self._pw_iq,   label=self._coord_lbl),
                    add_crosshair(self._pw_cs,   label=self._coord_lbl),
                    add_crosshair(self._pw_st_wf, label=self._coord_lbl)]
        lay.addWidget(inner)
        lay.addWidget(self._coord_lbl)

        # Build initial single Stuhrmann panel
        self._rebuild_stuhr_panels(1)

    def _on_stuhr_log_toggled(self, v: bool):
        for pw in self._stuhr_panels:
            pw.setLogMode(x=False, y=v)

    def _rebuild_stuhr_panels(self, n: int):
        """Rebuild N Stuhrmann panels in the right vertical splitter."""
        while self._stuhr_right_split.count() > 0:
            w = self._stuhr_right_split.widget(0)
            w.setParent(None)
        self._stuhr_panels.clear()
        self._stuhr_vb_resid.clear()
        self._stuhr_pts_resid.clear()
        self._stuhr_lbls.clear()
        self._stuhr_panel_crosshairs.clear()
        self._stuhr_items_by_panel.clear()
        self._stuhr_items.clear()   # legacy flat list — keep for compat

        n = max(1, n)
        for i in range(n):
            title = (f"Stuhrmann elem {i+1}: I_marginal vs f'_{i+1}"
                     if n > 1
                     else "Stuhrmann: I vs f'  at selected q")
            pw = pg.PlotWidget(title=title)
            pw.setLabel('bottom', "f'  (e)")
            pw.setLabel('left', 'I  (cm⁻¹)')
            pw.addLegend(offset=(10, 10))

            # Secondary ViewBox for residuals
            vb = pg.ViewBox()
            pw.scene().addItem(vb)
            pw.showAxis('right')
            pw.getAxis('right').setLabel('Weighted residual  (σ)', color='#ffd54f')
            pw.getAxis('right').linkToView(vb)
            vb.setXLink(pw.getViewBox())

            def _make_sync(pw_ref, vb_ref):
                def _sync():
                    vb_ref.setGeometry(pw_ref.getViewBox().sceneBoundingRect())
                    vb_ref.linkedViewChanged(pw_ref.getViewBox(), vb_ref.XAxis)
                return _sync
            pw.getViewBox().sigResized.connect(_make_sync(pw, vb))

            vb.addItem(pg.InfiniteLine(
                pos=0, angle=0,
                pen=pg.mkPen('#555555', width=1.0,
                             style=pg.QtCore.Qt.PenStyle.DashLine)))
            pts = pg.ScatterPlotItem(
                pen=pg.mkPen('#ffd54f', width=0.5),
                brush=pg.mkBrush('#ffd54f80'), size=7, symbol='d')
            vb.addItem(pts)

            lbl = QLabel()

            # Wrap in QWidget with label
            w = QWidget()
            wl = QVBoxLayout(w)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.addWidget(pw)
            wl.addWidget(lbl)

            self._stuhr_right_split.addWidget(w)
            self._stuhr_panels.append(pw)
            self._stuhr_vb_resid.append(vb)
            self._stuhr_pts_resid.append(pts)
            self._stuhr_lbls.append(lbl)
            self._stuhr_items_by_panel.append([])
            ch = add_crosshair(pw, label=self._coord_lbl)
            self._stuhr_panel_crosshairs.append(ch)

        if n > 1:
            self._stuhr_right_split.setSizes([400] * n)

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
            self._toggle_ref_visibility(self._chk_show_ref.isChecked())
            self.reference_loaded.emit(q_ref, I_MM, I_RM, I_RR)
            self._lbl_status.setText(f'Reference: {Path(path).name}')
        except Exception as e:
            self._lbl_status.setText(f'Reference load error: {e}')

    def _toggle_ref_visibility(self, visible: bool):
        for curve in self._ref_iq_curves.values():
            curve.setVisible(visible)

    def _run_lp_bounds(self):
        """Compute upper/lower bounds on I_RR at every q via linear programming.

        For each q, solve:
            max/min  I_RR
            s.t.     |A·x - I_meas| ≤ 1·σ  (model within 1σ of all measurements)
                     I_MM ≥ 0,  I_RR ≥ 0
        where x = (I_MM, I_RM, I_RR) and A is the ASAXS design matrix.
        """
        from scipy.optimize import linprog

        if self._result is None or self._result.get('diff_mode', False):
            self._lbl_status.setText('LP bounds: direct mode only — run decomposition first')
            return

        r       = self._result
        q       = r['q']
        fp      = self._fp
        fpp     = self._fpp
        I_mat   = r['I_raw']    # (NE, NQ)
        sig_mat = r['sig_raw']  # (NE, NQ)
        NQ      = len(q)

        A       = np.column_stack([np.ones(len(fp)), 2*fp, fp**2 + fpp**2])
        I_MM_w  = r['I_MM']   # WLS estimates — used to impose CS floor on I_RR
        I_RM_w  = r['I_RM']

        lo_arr = np.full(NQ, np.nan)
        hi_arr = np.full(NQ, np.nan)

        self._lbl_status.setText('Running LP bounds…')
        for j in range(NQ):
            I_meas = I_mat[:, j]
            sigma  = np.maximum(sig_mat[:, j], 1e-30)

            # Adaptive k: use the WLS max normalised residual (+ 5% margin) so
            # the feasible region always contains the WLS solution.  This keeps
            # the band as tight as the data allow while guaranteeing feasibility.
            w      = 1.0 / sigma
            sol, *_ = np.linalg.lstsq(A * w[:, None], I_meas * w, rcond=None)
            k_j    = max(1.0, np.abs((A @ sol - I_meas) / sigma).max()) * 1.05

            # CS lower bound on I_RR: I_RR ≥ I_RM²/I_MM (from WLS estimates).
            # Without this, the LP minimises I_RR by also driving I_RM→0,
            # collapsing the CS floor along with it.
            cs_lo = I_RM_w[j]**2 / max(abs(I_MM_w[j]), 1e-30)

            bounds = [(0, None), (None, None), (max(0.0, cs_lo), None)]

            A_ub = np.vstack([ A, -A])
            b_ub = np.concatenate([I_meas + k_j*sigma, -(I_meas - k_j*sigma)])

            res_lo = linprog([0, 0,  1], A_ub=A_ub, b_ub=b_ub,
                             bounds=bounds, method='highs')
            res_hi = linprog([0, 0, -1], A_ub=A_ub, b_ub=b_ub,
                             bounds=bounds, method='highs')
            if res_lo.status == 0:
                lo_arr[j] = res_lo.fun
            if res_hi.status == 0:
                hi_arr[j] = -res_hi.fun

        valid   = np.isfinite(lo_arr) & np.isfinite(hi_arr) & (hi_arr > 0)
        lo_plot = np.maximum(lo_arr[valid], 1e-12)   # floor to avoid log(-inf)
        hi_plot = hi_arr[valid]

        self._curve_lp_lo.setData(q[valid], lo_plot)
        self._curve_lp_hi.setData(q[valid], hi_plot)

        n_ok = valid.sum()
        self._lbl_status.setText(
            f'LP bounds done ({n_ok}/{NQ} q-points feasible, adaptive-k band)')

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
                # Multi-element result (diff mode — use element 0 only)
                multi_result = None
            else:
                # Always use decompose_multi for consistency
                elems = [(e['fp'], e['fpp']) for e in self._elements] \
                        if self._elements else [(self._fp, self._fpp)]
                mr = decompose_multi(q, I, sig, elems)
                # Extract element-0 partials for backward compat
                names = mr['names']
                parts = mr['partials']
                errs  = mr['errors']
                I_MM = parts[names.index('I_MM')]
                I_RM = parts[names.index('I_R1M')]
                I_RR = parts[names.index('I_R1R1')]
                s_MM = errs[names.index('I_MM')]
                s_RM = errs[names.index('I_R1M')]
                s_RR = errs[names.index('I_R1R1')]
                multi_result = mr
        except Exception as e:
            self._lbl_status.setText(f'Error: {e}')
            return

        # Cauchy-Schwarz diagnostic and optional enforcement (direct mode only)
        n_viol = 0
        if not diff_mode:
            cs_ratio_raw = cauchy_schwarz_ratio(I_MM, I_RM, I_RR)
            n_viol = int((cs_ratio_raw > 1.0).sum())
            if self._chk_cs_enforce.isChecked() and n_viol > 0:
                I_RR = enforce_cauchy_schwarz(I_MM, I_RM, I_RR)
                s_RR = s_RR.copy()
                s_RR[cs_ratio_raw > 1.0] = 0.0
                # Update in multi_result too
                if multi_result is not None:
                    idx_RR = multi_result['names'].index('I_R1R1')
                    multi_result['partials'][idx_RR] = I_RR
            cs_ratio = cauchy_schwarz_ratio(I_MM, I_RM, I_RR)
        else:
            cs_ratio = np.zeros_like(I_RM)

        self._result = dict(
            q=q,
            I_MM=I_MM, I_RM=I_RM, I_RR=I_RR,
            s_MM=s_MM, s_RM=s_RM, s_RR=s_RR,
            cs_ratio=cs_ratio,
            diff_mode=diff_mode,
            I_raw=I,      # masked intensity
            sig_raw=sig,  # masked sigma — needed for LP bounds
            multi=multi_result,   # full multi-element result (None in diff mode)
        )

        self._update_iq_plot()
        self._update_stuhrmann()
        self.decomposition_done.emit(
            self._q, I_MM, I_RM, I_RR, s_MM, s_RM, s_RR)
        mode = 'difference' if diff_mode else 'direct'
        kappa_str = f'κ = {condition_number(self._fp, self._fpp):.1f}'
        if diff_mode:
            cs_str = ''
        elif n_viol == 0:
            cs_str = '  |  C-S: OK'
        elif self._chk_cs_enforce.isChecked():
            cs_str = f'  |  C-S: {n_viol}/{len(q)} → enforced'
        else:
            cs_str = f'  |  C-S: {n_viol}/{len(q)} violated'
        n_elems = len(self._elements) if self._elements else 1
        self._lbl_status.setText(
            f'Done [{mode}, {n_elems} elem] — {self._I_matrix.shape[0]} energies, '
            f'{len(q)} q-points  |  {kappa_str}{cs_str}')

    def _update_iq_plot(self):
        r = self._result
        q = r['q']

        # Raw waterfall (gray) — use masked intensity so range matches partials
        for c in self._raw_curves:
            self._pw_iq.removeItem(c)
        self._raw_curves.clear()
        for row in r['I_raw']:
            c = self._pw_iq.plot(q, row,
                pen=pg.mkPen((160, 160, 160, 60), width=0.8))
            self._raw_curves.append(c)

        # Always plot the 3 base partials (elem 0) with standard colors
        for name, key_I, key_s in [
            ('I_MM', 'I_MM', 's_MM'),
            ('I_RM', 'I_RM', 's_RM'),
            ('I_RR', 'I_RR', 's_RR'),
        ]:
            Iq = r[key_I]
            self._curves_iq[name].setData(q, np.maximum(np.abs(Iq), 1e-40))

        # Additional curves for N>1 elements (remove old ones first)
        for c in getattr(self, '_extra_iq_curves', []):
            self._pw_iq.removeItem(c)
        self._extra_iq_curves = []

        mr = r.get('multi')
        if mr is not None and len(mr['names']) > 3:
            extra_colors = ['#FF9800', '#9C27B0', '#009688', '#F44336', '#2196F3']
            ec_idx = 0
            for name in mr['names']:
                if name in ('I_MM', 'I_R1M', 'I_R1R1'):
                    continue  # already shown
                idx = mr['names'].index(name)
                Iq = mr['partials'][idx]
                col = extra_colors[ec_idx % len(extra_colors)]
                ec_idx += 1
                c = self._pw_iq.plot(q, np.maximum(np.abs(Iq), 1e-40),
                                     name=name, pen=pg.mkPen(col, width=2))
                self._extra_iq_curves.append(c)

        self._refresh_errorbars()
        self._update_cs_plot()

    def _update_cs_plot(self):
        """Refresh the Cauchy-Schwarz diagnostic panel."""
        if self._result is None:
            return
        q  = self._result['q']
        cs = self._result.get('cs_ratio')
        if cs is None or self._result.get('diff_mode', False):
            self._curve_cs.setData([], [])
            self._pts_cs_ok.setData([], [])
            self._pts_cs_vio.setData([], [])
            return
        ok  = cs <= 1.0
        vio = ~ok
        lq  = np.log10(q)           # ScatterPlotItem lives in ViewBox (log10) coords
        self._curve_cs.setData(q, cs)
        self._pts_cs_ok.setData( x=lq[ok],  y=cs[ok])
        self._pts_cs_vio.setData(x=lq[vio], y=cs[vio])

    def _refresh_errorbars(self):
        if self._result is None:
            return
        r    = self._result
        q    = r['q']
        show = self._chk_errbar.isChecked()
        log  = self._chk_log.isChecked()
        # CS floor for I_RR: lower bar never goes below I_RM²/|I_MM|
        if not r.get('diff_mode', False):
            cs_floor = r['I_RM']**2 / np.maximum(np.abs(r['I_MM']), 1e-30)
        else:
            cs_floor = None

        for name, key_I, key_s in [
            ('I_MM', 'I_MM', 's_MM'),
            ('I_RM', 'I_RM', 's_RM'),
            ('I_RR', 'I_RR', 's_RR'),
        ]:
            Iq  = np.maximum(np.abs(r[key_I]), 1e-40)
            si  = r[key_s]

            # For I_RR clip the bottom bar at the CS floor so it never extends
            # into the physically forbidden region below I_RM²/|I_MM|.
            if name == 'I_RR' and cs_floor is not None:
                bot = np.minimum(si, np.maximum(Iq - cs_floor, 0.0))
            else:
                bot = si

            if show:
                # Only plot where the bottom bar has positive extent.
                ok = bot > 0
                if log:
                    # ErrorBarItem is a raw ViewBox item — needs log10 coords.
                    lq  = np.log10(q[ok])
                    lIq = np.log10(Iq[ok])
                    top    = np.log10(Iq[ok] + si[ok]) - lIq
                    bottom = lIq - np.log10(Iq[ok] - bot[ok])
                    self._err_items[name].setData(
                        x=lq, y=lIq, top=top, bottom=bottom)
                else:
                    self._err_items[name].setData(
                        x=q[ok], y=Iq[ok], top=si[ok], bottom=bot[ok])
            else:
                self._err_items[name].setData(
                    x=np.array([]), y=np.array([]),
                    top=np.array([]), bottom=np.array([]))

    def _update_log(self, log: bool):
        self._pw_iq.setLogMode(x=True, y=log)
        self._refresh_errorbars()


    def _update_stuhrmann(self):
        """Called when data/fp/fpp arrive; rebuilds embedded waterfall + redraws Stuhrmann."""
        if self._q is None or self._fp is None or self._I_matrix is None:
            return
        self._update_stuhr_waterfall()
        NQ = len(self._q)
        self._sl_q.blockSignals(True)
        self._sl_q.setMaximum(NQ - 1)
        default_idx = NQ // 10
        self._sl_q.setValue(default_idx)
        self._sl_q.blockSignals(False)
        self._on_q_slider(default_idx)

    def _update_stuhr_waterfall(self):
        """Rebuild the embedded waterfall (inside the Stuhrmann tab)."""
        if self._q is None or self._I_matrix is None:
            return
        for c in self._st_wf_curves:
            self._pw_st_wf.removeItem(c)
        self._st_wf_curves.clear()

        log_y = self._chk_st_logi.isChecked()
        log_q = self._chk_st_logq.isChecked()
        self._pw_st_wf.setLogMode(x=log_q, y=log_y)

        N_E      = self._I_matrix.shape[0]
        energies = self._energies if self._energies else list(range(N_E))
        E_arr    = np.array([e if e is not None else i
                             for i, e in enumerate(energies)], dtype=float)
        E_min, E_max = E_arr.min(), E_arr.max()
        E_span = E_max - E_min if E_max > E_min else 1.0

        def _ecol(t):
            stops = [(0.0, (0, 0, 255)), (0.33, (0, 200, 255)),
                     (0.5, (0, 220, 0)), (0.67, (255, 220, 0)), (1.0, (255, 0, 0))]
            for i in range(len(stops) - 1):
                t0, c0 = stops[i]; t1, c1 = stops[i + 1]
                if t0 <= t <= t1:
                    f = (t - t0) / (t1 - t0)
                    return pg.QtGui.QColor(*(int(c0[k] + f * (c1[k] - c0[k])) for k in range(3)))
            return pg.QtGui.QColor(255, 0, 0)

        for k in range(N_E):
            t    = (E_arr[k] - E_min) / E_span
            col  = _ecol(t)
            Iq   = self._I_matrix[k]
            mask = Iq > 0
            if not mask.any():
                continue
            c = self._pw_st_wf.plot(
                self._q[mask], Iq[mask],
                pen=pg.mkPen(col, width=1.2))
            self._st_wf_curves.append(c)

        self._update_st_vline()

    def _on_st_logq_toggled(self, log_q: bool):
        self._pw_st_wf.setLogMode(x=log_q, y=self._chk_st_logi.isChecked())
        self._update_stuhr_waterfall()

    def _update_st_vline(self):
        """Position the vertical dashed line at the currently selected q."""
        if self._q is None or len(self._q) == 0:
            return
        idx   = max(0, min(self._sl_q.value(), len(self._q) - 1))
        q_val = self._q[idx]
        # InfiniteLine lives in ViewBox coordinates; log mode transforms data→log10
        if self._chk_st_logq.isChecked():
            self._st_vline.setPos(np.log10(q_val))
        else:
            self._st_vline.setPos(q_val)

    def _on_q_slider(self, idx: int):
        if self._q is None or len(self._q) == 0:
            return
        idx   = max(0, min(idx, len(self._q) - 1))
        q_val = self._q[idx]
        self._lbl_q_val.setText(f'{q_val:.4f} Å⁻¹  [{idx + 1}/{len(self._q)}]')
        self._update_st_vline()
        self._plot_stuhrmann_at(idx)

    def _plot_stuhrmann_at(self, idx: int):
        """Plot I vs f'_i(E) for each active element in its own panel."""
        if self._q is None or self._fp is None or self._I_matrix is None:
            return
        if idx < 0 or idx >= len(self._q):
            return

        from .core.decompose import build_A
        from numpy.linalg import lstsq

        q_j   = self._q[idx]
        Iq_j  = self._I_matrix[:, idx]
        sig_j = self._sig[:, idx]

        # Determine active elements to plot
        elems = self._elements if self._elements else \
                [{'fp': self._fp, 'fpp': self._fpp, 'Z': 0, 'label': 'R1'}]
        n_elems = len(elems)

        # Rebuild panels if count changed
        if len(self._stuhr_panels) != n_elems:
            self._rebuild_stuhr_panels(n_elems)

        # Global decomp result lookup
        r    = self._result
        ri   = None
        if r is not None:
            ri = int(np.argmin(np.abs(r['q'] - q_j)))

        col_data = pg.mkColor('#4fc3f7')
        col_fit  = pg.mkColor('#ff8a65')

        for panel_i, elem in enumerate(elems):
            pw  = self._stuhr_panels[panel_i]
            vb  = self._stuhr_vb_resid[panel_i]
            pts_resid = self._stuhr_pts_resid[panel_i]
            lbl = self._stuhr_lbls[panel_i]

            # Clear panel — use per-panel list so multi-panel works correctly
            for item in list(self._stuhr_items_by_panel[panel_i]):
                try:
                    pw.removeItem(item)
                except Exception:
                    pass
            self._stuhr_items_by_panel[panel_i].clear()
            try:
                pw.getPlotItem().legend.clear()
            except Exception:
                pass
            pts_resid.setData([], [])

            fp  = elem['fp']
            fpp = elem['fpp']

            if len(fp) == 0:
                continue

            # Compute marginal intensity for this element
            if n_elems == 1 or r is None:
                # N=1: use raw I(q,E) directly (backward compat)
                I_marginal = Iq_j
            else:
                # Subtract other elements' pure contributions
                names   = r['multi']['names']
                parts_q = r['multi']['partials'][:, ri]
                I_MM_g  = parts_q[names.index('I_MM')]
                I_marginal = Iq_j.copy() - I_MM_g
                for j, other in enumerate(elems):
                    if j == panel_i:
                        continue
                    fp_j  = other['fp']
                    fpp_j = other['fpp']
                    nm_RjM  = f'I_R{j+1}M'
                    nm_RjRj = f'I_R{j+1}R{j+1}'
                    I_RjM  = parts_q[names.index(nm_RjM)]  if nm_RjM  in names else 0.0
                    I_RjRj = parts_q[names.index(nm_RjRj)] if nm_RjRj in names else 0.0
                    I_marginal -= 2.0 * fp_j * I_RjM + (fp_j**2 + fpp_j**2) * I_RjRj

            # Local WLS solve using just this element's 3-term model
            A_loc = build_A(fp, fpp)
            w     = 1.0 / np.maximum(sig_j, 1e-30)
            sol, *_ = lstsq(A_loc * w[:, None], I_marginal * w, rcond=None)
            I_MM_j, I_RM_j, I_RR_j = sol

            fp_dense  = np.linspace(fp.min() - 0.5, fp.max() + 0.5, 300)
            _sort     = np.argsort(fp)
            fpp_dense = np.interp(fp_dense, fp[_sort], fpp[_sort])
            I_fit     = (I_MM_j + 2*fp_dense*I_RM_j
                         + (fp_dense**2 + fpp_dense**2)*I_RR_j)

            new_items = []

            scatter = pg.ScatterPlotItem(
                x=fp, y=I_marginal, size=9,
                pen=pg.mkPen(col_data), brush=pg.mkBrush(col_data),
                name='data')
            pw.addItem(scatter)
            new_items.append(scatter)

            ei = pg.ErrorBarItem(x=fp, y=I_marginal, top=sig_j, bottom=sig_j,
                                 pen=pg.mkPen(col_data, width=1.0))
            pw.addItem(ei)
            new_items.append(ei)

            fit_c = pg.PlotDataItem(fp_dense, I_fit,
                                    pen=pg.mkPen(col_fit, width=2.0),
                                    name='local fit')
            pw.addItem(fit_c)
            new_items.append(fit_c)

            # Weighted residuals
            I_model_pts = (I_MM_j + 2*fp*I_RM_j
                           + (fp**2 + fpp**2)*I_RR_j)
            resid_w = (I_marginal - I_model_pts) / np.maximum(sig_j, 1e-30)
            pts_resid.setData(x=fp, y=resid_w)

            # Global decomp fit overlay
            if r is not None and ri is not None:
                nm_RiM  = f'I_R{panel_i+1}M'
                nm_RiRi = f'I_R{panel_i+1}R{panel_i+1}'
                names_g = r['multi']['names'] if r.get('multi') else ['I_MM','I_R1M','I_R1R1']
                parts_ri = r['multi']['partials'][:, ri] if r.get('multi') \
                           else np.array([r['I_MM'][ri], r['I_RM'][ri], r['I_RR'][ri]])
                I_MM_g  = parts_ri[names_g.index('I_MM')]
                I_RiM_g = parts_ri[names_g.index(nm_RiM)]  if nm_RiM  in names_g else r['I_RM'][ri]
                I_RiRi_g = parts_ri[names_g.index(nm_RiRi)] if nm_RiRi in names_g else r['I_RR'][ri]
                I_comp = (I_MM_g + 2*fp_dense*I_RiM_g
                          + (fp_dense**2 + fpp_dense**2)*I_RiRi_g)
                comp_c = pg.PlotDataItem(
                    fp_dense, I_comp,
                    pen=pg.mkPen('#aed581', width=1.5,
                                 style=pg.QtCore.Qt.PenStyle.DashLine),
                    name='decomp. fit')
                pw.addItem(comp_c)
                new_items.append(comp_c)

            self._stuhr_items_by_panel[panel_i] = new_items
            lbl.setText(
                f'q = {q_j:.4f} Å⁻¹  elem {panel_i+1}  |  '
                f'I_MM = {I_MM_j:.4g}   '
                f'I_RM = {I_RM_j:.4g}   '
                f'I_RR = {I_RR_j:.4g}  cm⁻¹')

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
        self._update_stuhr_waterfall()

    def set_elements(self, elements: list):
        """Set multi-element list. Each item: {'Z','label','fp','fpp'}."""
        self._elements = elements
        if elements:
            self._fp  = elements[0]['fp']
            self._fpp = elements[0]['fpp']
        n = max(1, len(elements))
        self._rebuild_stuhr_panels(n)
        self._update_stuhrmann()

    def set_fp_fpp(self, fp: np.ndarray, fpp: np.ndarray):
        """Backward-compat shim — wraps into single-element list."""
        self.set_elements([{'fp': fp, 'fpp': fpp, 'Z': 0, 'label': 'R1'}])

    def get_result(self) -> dict | None:
        return self._result
