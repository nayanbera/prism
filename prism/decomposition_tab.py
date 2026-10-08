"""
q-space decomposition tab.

Direct mode   — WLS at each q → I_MM, I_RM, I_RR + propagated σ
Difference mode — subtract reference energy E_ref → I_RM, I_RR only
                  (I_MM cancels; more robust, lower κ)
"""

import re
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
    cauchy_schwarz_ratio, solve_partials,
    decompose_multi, decompose_multi_per_edge, build_A_multi, condition_number_multi,
    compute_edge_groups, compute_edge_groups_by_energy, estimate_normalization_lowq,
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

        top.addWidget(QLabel('Method:'))
        self._combo_method = QComboBox()
        self._combo_method.addItems(['WLS', 'OLS'])
        self._combo_method.setToolTip(
            'WLS — Weighted Least Squares: each energy weighted by 1/σ².\n'
            'OLS — Ordinary Least Squares: all energies treated equally.\n'
            'WLS is recommended when σ values are reliable.')
        self._combo_method.setFixedWidth(60)
        top.addWidget(self._combo_method)

        self._chk_log = QCheckBox('Log I')
        self._chk_log.setChecked(True)
        self._chk_log.toggled.connect(self._update_log)
        top.addWidget(self._chk_log)
        self._chk_logq = QCheckBox('Log q')
        self._chk_logq.setChecked(True)
        self._chk_logq.setToolTip('Log or linear q axis for the partial I(q) and C-S plots.')
        self._chk_logq.toggled.connect(self._update_log)
        top.addWidget(self._chk_logq)

        self._chk_errbar = QCheckBox('Error bars')
        self._chk_errbar.setChecked(True)
        self._chk_errbar.toggled.connect(self._refresh_errorbars)
        top.addWidget(self._chk_errbar)

        self._chk_nonneg = QCheckBox('I_MM, I_RR ≥ 0')
        self._chk_nonneg.setChecked(True)
        self._chk_nonneg.setToolTip(
            'Constrain the fit so I_MM(q) ≥ 0 and I_RR(q) ≥ 0 (they are |F|²\n'
            'terms and cannot be negative).  Applies to direct and difference\n'
            'mode (difference mode has no I_MM).  I_RM and cross-terms stay\n'
            'signed.  Error bars are those of the unconstrained fit.\n'
            'Re-run decomposition to apply.')
        top.addWidget(self._chk_nonneg)

        self._chk_cs_enforce = QCheckBox('Enforce C-S')
        self._chk_cs_enforce.setToolTip(
            'Constrain the fit so the Cauchy-Schwarz inequality\n'
            'I_RM² ≤ I_MM·I_RR holds at every q (direct mode only; implies\n'
            'I_MM, I_RR ≥ 0).  For multi-edge fits the inequality is applied\n'
            'after the fit by raising I_RiRi.  Re-run decomposition to apply.')
        top.addWidget(self._chk_cs_enforce)

        self._btn_lp = QPushButton('I_RR bounds (LP)')
        self._btn_lp.setToolTip(
            'Compute the upper and lower bounds on I_RR at each q by solving\n'
            'a linear program: largest/smallest I_RR consistent with all\n'
            'measured intensities within 1σ.  Direct mode only.')
        self._btn_lp.clicked.connect(self._run_lp_bounds)
        top.addWidget(self._btn_lp)

        self._chk_norm_opt = QCheckBox('Normalize cross-edge')
        self._chk_norm_opt.setToolTip(
            'Show cross-edge normalization controls (second row).\n'
            'Set β₂ / β₃ to correct for different absolute scales between edge\n'
            'measurements.  I_edge_i(q,E) is divided by βᵢ before the WLS solve.\n'
            'Direct mode, multi-element only.')
        top.addWidget(self._chk_norm_opt)

        self._chk_norm_opt.toggled.connect(self._on_norm_opt_toggled)

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

        # ── Cross-edge normalisation row (hidden until checkbox is ticked) ──────
        from PyQt6.QtWidgets import QDoubleSpinBox as _BSPIN, QFrame
        self._beta_row = QWidget()
        beta_lay = QHBoxLayout(self._beta_row)
        beta_lay.setContentsMargins(6, 2, 6, 2)

        beta_lay.addWidget(QLabel('Cross-edge β:'))

        beta_lay.addWidget(QLabel('β₂ ='))
        self._beta_spin_2 = _BSPIN()
        self._beta_spin_2.setDecimals(4)
        self._beta_spin_2.setRange(0.01, 100.0)
        self._beta_spin_2.setValue(1.0)
        self._beta_spin_2.setFixedWidth(80)
        self._beta_spin_2.setToolTip(
            'Normalization for element 2.  I₂(q,E) is divided by this value\n'
            'before the WLS solve.  1.0 = no change.')
        beta_lay.addWidget(self._beta_spin_2)

        self._beta_lbl_3 = QLabel('β₃ =')
        self._beta_spin_3 = _BSPIN()
        self._beta_spin_3.setDecimals(4)
        self._beta_spin_3.setRange(0.01, 100.0)
        self._beta_spin_3.setValue(1.0)
        self._beta_spin_3.setFixedWidth(80)
        self._beta_spin_3.setToolTip(
            'Normalization for element 3.  I₃(q,E) is divided by this value\n'
            'before the WLS solve.  1.0 = no change.')
        self._beta_lbl_3.setVisible(False)
        self._beta_spin_3.setVisible(False)
        beta_lay.addWidget(self._beta_lbl_3)
        beta_lay.addWidget(self._beta_spin_3)

        self._btn_est_beta = QPushButton('Estimate from low-q')
        self._btn_est_beta.setToolTip(
            'Fill β values from the low-q intensity ratio (assumes I(q_low) ≈ I_MM).\n'
            'UNRELIABLE when |2·f\'·I_RM| is comparable to I_MM at low q.\n'
            'Use as a starting point and verify via the Stuhrmann panels.')
        self._btn_est_beta.clicked.connect(self._estimate_beta_clicked)
        beta_lay.addWidget(self._btn_est_beta)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        beta_lay.addWidget(sep)

        self._beta_hint = QLabel(
            'Set β₂ from monitor count ratio or absolute flux calibration.  '
            '1.0 = no correction.')
        self._beta_hint.setStyleSheet('color: #888888; font-size: 11px;')
        beta_lay.addWidget(self._beta_hint)
        beta_lay.addStretch()

        self._beta_row.setVisible(False)
        lay.addWidget(self._beta_row)

        # ── Tab: I(q) plot / Stuhrmann ────────────────────────────────────────
        inner = QTabWidget()

        # Partial I(q) plot
        self._pw_iq = pg.PlotWidget(title='Partial structure factors I(q)')
        self._pw_iq.setLabel('bottom', 'q (Å⁻¹)')
        self._pw_iq.setLabel('left', 'I(q) (cm⁻¹)')
        _iq_legend = self._pw_iq.addLegend()
        self._pw_iq.setLogMode(x=True, y=True)
        # Use PlotCurveItem (not PlotDataItem) so we can apply log10 manually
        # and use connect='finite' reliably — PlotDataItem does not forward
        # connect='finite' to its inner PlotCurveItem in all PyQtGraph versions.
        self._curves_iq:     dict[str, pg.PlotCurveItem]   = {}
        self._curves_iq_neg: dict[str, pg.PlotCurveItem]   = {}
        self._neg_markers:   dict[str, pg.ScatterPlotItem] = {}
        self._err_items:     dict[str, pg.ErrorBarItem]    = {}
        for name, col in _COLORS.items():
            c = pg.PlotCurveItem(pen=pg.mkPen(col, width=2))
            self._pw_iq.addItem(c)
            _iq_legend.addItem(c, name)
            self._curves_iq[name] = c
            c_neg = pg.PlotCurveItem(
                pen=pg.mkPen(col, width=1.5,
                             style=pg.QtCore.Qt.PenStyle.DashLine))
            self._pw_iq.addItem(c_neg)
            self._curves_iq_neg[name] = c_neg
            # Inverted triangles at negative data points (conventional log-plot sign marker)
            m = pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(col),
                                   symbol='t1', size=7)
            self._pw_iq.addItem(m)
            self._neg_markers[name] = m
            ei = pg.ErrorBarItem(x=np.array([]), y=np.array([]),
                                 top=np.array([]), bottom=np.array([]),
                                 pen=pg.mkPen(col, width=0.8))
            self._pw_iq.addItem(ei)
            self._err_items[name] = ei
        self._raw_curves: list = []
        # Dashed reference overlay — populated dynamically by _load_reference()
        self._ref_iq_curves: dict[str, pg.PlotDataItem] = {}

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

    # Color scheme for reference partial curves
    _REF_COLORS = {
        'I_MM':   '#42A5F5',                         # blue 400
        'I_R1M':  '#66BB6A', 'I_R2M': '#26C6DA', 'I_R3M': '#FFCA28',   # green, cyan, amber
        'I_RM':   '#66BB6A',                         # legacy single-element alias
        'I_R1R1': '#EF5350', 'I_R2R2': '#AB47BC', 'I_R3R3': '#FF7043', # red, purple, deep-orange
        'I_RR':   '#EF5350',                         # legacy alias
        'I_R1R2': '#FF9800', 'I_R1R3': '#26A69A', 'I_R2R3': '#EC407A', # orange, teal, pink
    }

    def _load_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load reference partials', '',
            'Data files (*.dat *.txt);;All files (*)')
        if not path:
            return
        try:
            # Parse column names from header line "# q(1/A)  I_MM(cm-1)  ..."
            col_names = []
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line.startswith('#'):
                        break
                    # Look for the header row that has q(1/A) and partial names
                    if 'q(' in line or ('I_MM' in line and 'I_R' in line):
                        parts = line.lstrip('#').split()
                        col_names = [p.split('(')[0] for p in parts]
            data = np.loadtxt(path, comments='#')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            q_ref = data[:, 0]

            # Fall back to positional names if header parse failed
            if len(col_names) != data.shape[1]:
                from .core.decompose import partial_names_multi
                n_partials = data.shape[1] - 1
                # Guess n_elem: (n+1)(n+2)/2 = n_partials
                for n_try in range(1, 4):
                    if (n_try + 1) * (n_try + 2) // 2 == n_partials:
                        col_names = ['q'] + partial_names_multi(n_try)
                        break
                else:
                    col_names = ['q'] + [f'col_{i}' for i in range(1, data.shape[1])]

            # Remove old reference curves from plot
            for curve in self._ref_iq_curves.values():
                self._pw_iq.removeItem(curve)
            self._ref_iq_curves.clear()

            # Create one dashed curve per partial column
            for ci, name in enumerate(col_names[1:], start=1):
                arr = np.abs(data[:, ci]) if 'RM' in name else data[:, ci]
                col = self._REF_COLORS.get(name, '#888888')
                curve = self._pw_iq.plot(
                    q_ref, arr, name=f'{name} ref',
                    pen=pg.mkPen(col, width=1.5,
                                 style=pg.QtCore.Qt.PenStyle.DashLine))
                self._ref_iq_curves[name] = curve

            self._toggle_ref_visibility(self._chk_show_ref.isChecked())

            # Emit backward-compat signal with first-element partials
            names = col_names[1:]
            def _col(n): return data[:, col_names.index(n)] if n in col_names else np.zeros_like(q_ref)
            I_MM = _col('I_MM')
            I_RM = _col('I_R1M') if 'I_R1M' in col_names else _col('I_RM')
            I_RR = _col('I_R1R1') if 'I_R1R1' in col_names else _col('I_RR')
            self.reference_loaded.emit(q_ref, I_MM, I_RM, I_RR)
            self._lbl_status.setText(
                f'Reference: {Path(path).name}  ({len(names)} partials)')
        except Exception as e:
            self._lbl_status.setText(f'Reference load error: {e}')

    def _toggle_ref_visibility(self, visible: bool):
        for curve in self._ref_iq_curves.values():
            curve.setVisible(visible)

    def _run_lp_bounds(self):
        """Compute upper/lower bounds on I_RiRi at every q via linear programming.

        For each element i and each q, solve:
            max/min  I_RiRi
            s.t.     |A·x - I_meas| ≤ k·σ  (adaptive k)
                     I_MM ≥ 0,  all I_RkRk ≥ 0,  I_RiRi ≥ I_RiM²/I_MM  (CS floor)
        """
        from scipy.optimize import linprog
        from .core.decompose import build_A_multi, partial_names_multi

        if self._result is None or self._result.get('diff_mode', False):
            self._lbl_status.setText('LP bounds: direct mode only — run decomposition first')
            return

        r       = self._result
        q       = r['q']
        I_mat   = r['I_raw']
        sig_mat = r['sig_raw']
        NQ      = len(q)

        elems   = self._elements if self._elements else \
                  [{'fp': self._fp, 'fpp': self._fpp}]
        n_elems = len(elems)

        if n_elems == 1:
            fp  = elems[0].get('fp', self._fp)
            fpp = elems[0].get('fpp', self._fpp)
            A      = np.column_stack([np.ones(len(fp)), 2*fp, fp**2 + fpp**2])
            names  = ['I_MM', 'I_R1M', 'I_R1R1']
        else:
            elem_pairs = [(e['fp'], e['fpp']) for e in elems]
            A      = build_A_multi(elem_pairs, include_cross=False)
            names  = partial_names_multi(n_elems, include_cross=False)

        n_cols  = A.shape[1]
        mm_idx  = names.index('I_MM')
        rr_idxs = [names.index(f'I_R{i+1}R{i+1}') for i in range(n_elems)]
        rm_idxs = [names.index(f'I_R{i+1}M')       for i in range(n_elems)]

        # WLS estimates for CS floor
        mr      = r.get('multi')
        mr_parts = mr['partials'] if mr is not None else None
        mr_names = mr['names']    if mr is not None else names

        # Variable bounds template: I_MM ≥ 0, all I_RkRk ≥ 0, others free
        base_bounds = [(None, None)] * n_cols
        base_bounds[mm_idx] = (0, None)
        for rr_idx in rr_idxs:
            base_bounds[rr_idx] = (0, None)

        all_lo = [np.full(NQ, np.nan) for _ in range(n_elems)]
        all_hi = [np.full(NQ, np.nan) for _ in range(n_elems)]

        self._lbl_status.setText('Running LP bounds…')
        for j in range(NQ):
            I_meas = I_mat[:, j]
            sigma  = np.maximum(sig_mat[:, j], 1e-30)
            w      = 1.0 / sigma
            sol, *_ = np.linalg.lstsq(A * w[:, None], I_meas * w, rcond=None)
            k_j    = max(1.0, np.abs((A @ sol - I_meas) / sigma).max()) * 1.05
            A_ub   = np.vstack([A, -A])
            b_ub   = np.concatenate([I_meas + k_j*sigma, -(I_meas - k_j*sigma)])

            for elem_i, rr_idx in enumerate(rr_idxs):
                # CS floor for this element using WLS estimates
                if mr_parts is not None:
                    I_RiM_w = float(mr_parts[mr_names.index(f'I_R{elem_i+1}M'), j])
                    I_MM_w  = float(mr_parts[mr_names.index('I_MM'), j])
                else:
                    I_RiM_w = float(r['I_RM'][j])
                    I_MM_w  = float(r['I_MM'][j])
                cs_lo = I_RiM_w**2 / max(abs(I_MM_w), 1e-30)

                var_bounds = list(base_bounds)
                var_bounds[rr_idx] = (max(0.0, cs_lo), None)

                c_lo = [0.0] * n_cols; c_lo[rr_idx] =  1.0
                c_hi = [0.0] * n_cols; c_hi[rr_idx] = -1.0

                res_lo = linprog(c_lo, A_ub=A_ub, b_ub=b_ub, bounds=var_bounds, method='highs')
                res_hi = linprog(c_hi, A_ub=A_ub, b_ub=b_ub, bounds=var_bounds, method='highs')
                if res_lo.status == 0:
                    all_lo[elem_i][j] = res_lo.fun
                if res_hi.status == 0:
                    all_hi[elem_i][j] = -res_hi.fun

        # Remove old extra bands for elements 1+
        for item in getattr(self, '_extra_lp_items', []):
            try:
                self._pw_iq.removeItem(item)
            except Exception:
                pass
        self._extra_lp_items = []

        lp_colors = ['#ff4444', '#FF9800', '#9C27B0']

        # Element 0: existing curves
        valid0  = np.isfinite(all_lo[0]) & np.isfinite(all_hi[0]) & (all_hi[0] > 0)
        self._curve_lp_lo.setData(q[valid0], np.maximum(all_lo[0][valid0], 1e-12))
        self._curve_lp_hi.setData(q[valid0], all_hi[0][valid0])

        # Elements 1+: create new curves
        for elem_i in range(1, n_elems):
            col   = lp_colors[elem_i % len(lp_colors)]
            pen   = pg.mkPen(col, width=1, style=pg.QtCore.Qt.PenStyle.DotLine)
            valid_i = np.isfinite(all_lo[elem_i]) & np.isfinite(all_hi[elem_i]) & (all_hi[elem_i] > 0)
            lo_i  = np.maximum(all_lo[elem_i][valid_i], 1e-12)
            hi_i  = all_hi[elem_i][valid_i]
            c_lo  = self._pw_iq.plot(q[valid_i], lo_i, pen=pen)
            c_hi  = self._pw_iq.plot(q[valid_i], hi_i, pen=pen,
                                     name=f'I_R{elem_i+1}R{elem_i+1} LP bounds')
            r_col = pg.mkColor(col).getRgb()[:3]
            fill  = pg.FillBetweenItem(c_lo, c_hi, brush=pg.mkBrush(*r_col, 45))
            self._pw_iq.addItem(fill)
            self._extra_lp_items.extend([c_lo, c_hi, fill])

        n_ok = sum(int((np.isfinite(all_lo[i]) & np.isfinite(all_hi[i])).sum())
                   for i in range(n_elems))
        self._lbl_status.setText(
            f'LP bounds done ({n_elems} element(s), {n_ok} feasible q-pts, adaptive-k)')

    # ── Cross-edge normalization helpers ──────────────────────────────────────
    def _on_norm_opt_toggled(self, checked: bool):
        n_elems = len(self._elements) if self._elements else 1
        self._beta_row.setVisible(checked)
        self._beta_lbl_3.setVisible(n_elems >= 3)
        self._beta_spin_3.setVisible(n_elems >= 3)

    def _estimate_beta_clicked(self):
        if self._q is None or self._I_matrix is None or not self._elements:
            return
        n_elems = len(self._elements)
        if n_elems <= 1:
            return
        elems = [(e['fp'], e['fpp']) for e in self._elements]
        qmask = ((self._q >= self._spin_qmin.value()) &
                 (self._q <= self._spin_qmax.value()))
        q_m = self._q[qmask]
        I_m = self._I_matrix[:, qmask]
        # Assign energies by proximity to each element's edge energy.
        # compute_edge_groups (argmax fpp) is unreliable with tabulated f''
        # because the white-line enhancement is absent.
        energies_keV = np.array(self._energies) if self._energies else None
        if energies_keV is not None and len(energies_keV) == len(elems[0][0]):
            elem_meta = [{'Z': e.get('Z', 0), 'shell': e.get('shell', '')}
                         for e in self._elements]
            edge_groups = compute_edge_groups_by_energy(elem_meta, energies_keV)
        else:
            edge_groups = compute_edge_groups(elems)
        try:
            beta = estimate_normalization_lowq(q_m, I_m, edge_groups)
        except Exception as exc:
            self._beta_hint.setText(f'Estimate failed: {exc}')
            self._beta_hint.setStyleSheet('color: #ff6666; font-size: 11px;')
            return
        if n_elems >= 2:
            self._beta_spin_2.setValue(float(beta[1]))
        if n_elems >= 3:
            self._beta_spin_3.setValue(float(beta[2]))
        parts = ', '.join(f'β{i+1}={b:.4f}' for i, b in enumerate(beta))
        self._beta_hint.setText(f'Estimated: {parts}  — click Run decomposition to apply')
        self._beta_hint.setStyleSheet('color: #aed581; font-size: 11px;')

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
        n_elems = len(self._elements) if self._elements else 1
        if n_elems == 1:
            self._lbl_status.setText(
                'Running with 1 resonant element — '
                'compute f\'/f\'\' for all elements in the Data tab first for multi-element decomp')

        diff_mode = self._chk_diff.isChecked()
        ref_idx   = self._combo_ref.currentIndex()
        method    = self._combo_method.currentText()
        nonneg    = self._chk_nonneg.isChecked()
        enforce_cs = self._chk_cs_enforce.isChecked() and not diff_mode

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
                    q, I, sig, self._fp, self._fpp, ref_idx, method=method,
                    nonneg=nonneg)
                I_MM = np.zeros_like(I_RM)
                s_MM = np.zeros_like(I_RM)
                # Multi-element result (diff mode — use element 0 only)
                multi_result = None
                beta_opt     = None
                edge_groups  = None
            else:
                # Always use decompose_multi for consistency
                elems = [(e['fp'], e['fpp']) for e in self._elements] \
                        if self._elements else [(self._fp, self._fpp)]

                # Build edge groups for N>1 (per-edge decomp + beta normalization)
                edge_groups_all = None
                if n_elems > 1:
                    energies_keV = np.array(self._energies) if self._energies else None
                    if (energies_keV is not None
                            and len(energies_keV) == len(elems[0][0])):
                        elem_meta = [{'Z': e.get('Z', 0), 'shell': e.get('shell', '')}
                                     for e in self._elements]
                        edge_groups_all = compute_edge_groups_by_energy(
                            elem_meta, energies_keV)
                    else:
                        edge_groups_all = compute_edge_groups(elems)

                # Optional cross-edge normalization (manual β from spin boxes)
                beta_opt    = None
                edge_groups = None   # stored in result; only set when beta active
                if n_elems > 1 and self._chk_norm_opt.isChecked():
                    beta_vals = [1.0]
                    if n_elems >= 2:
                        beta_vals.append(self._beta_spin_2.value())
                    if n_elems >= 3:
                        beta_vals.append(self._beta_spin_3.value())
                    while len(beta_vals) < n_elems:
                        beta_vals.append(1.0)
                    beta_opt = np.array(beta_vals[:n_elems])
                    if np.allclose(beta_opt, 1.0, atol=1e-6):
                        beta_opt = None
                    else:
                        edge_groups = edge_groups_all  # activate display scaling

                # Global WLS with per-edge offsets + cross-terms for N>1.
                # Per-edge offsets handle cross-edge absolute scale mismatch;
                # include_cross=True eliminates the systematic bias in I_RiRi
                # caused by unmodelled cross-terms bleeding into diagonal partials.
                if n_elems > 1 and edge_groups_all is not None:
                    mr = decompose_multi_per_edge(q, I, sig, elems,
                                                  edge_groups_all, beta=beta_opt,
                                                  include_cross=(n_elems > 1),
                                                  method=method,
                                                  nonneg=nonneg,
                                                  enforce_cs=enforce_cs)
                else:
                    mr = decompose_multi(q, I, sig, elems,
                                         beta=beta_opt, edge_groups=edge_groups,
                                         method=method, nonneg=nonneg,
                                         enforce_cs=enforce_cs)
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

        # Cauchy-Schwarz diagnostic (direct mode only).  Enforcement, if requested,
        # already happened inside the constrained solve.
        n_viol = 0
        if not diff_mode:
            cs_ratio = cauchy_schwarz_ratio(I_MM, I_RM, I_RR)
            n_viol = int((cs_ratio > 1.0 + 1e-6).sum())
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
            beta=beta_opt,        # normalization factors (None if not optimized)
            edge_groups=edge_groups,
            edge_groups_all=edge_groups_all if n_elems > 1 else None,
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
            cs_str = f'  |  C-S: enforced ({n_viol} residual violations)'
        else:
            cs_str = f'  |  C-S: {n_viol}/{len(q)} violated'
        n_elems = len(self._elements) if self._elements else 1
        if beta_opt is not None:
            beta_str = '  |  β=[' + ', '.join(f'{b:.4f}' for b in beta_opt) + ']'
        else:
            beta_str = ''
        self._lbl_status.setText(
            f'Done [{mode}, {n_elems} elem] — {self._I_matrix.shape[0]} energies, '
            f'{len(q)} q-points  |  {kappa_str}{cs_str}{beta_str}')

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

        # PlotCurveItem does not auto-apply log transform — we do it manually
        # so connect='finite' reliably creates gaps at nan (sign crossings).
        log_y = self._chk_log.isChecked()
        xq = self._xq(q)

        # Always plot the 3 base partials (elem 0) with standard colors.
        # Log y: positive values solid; negative values drawn as |value|, dashed,
        # with inverted-triangle markers.  Linear y: signed values as they are.
        for name, key_I in [('I_MM', 'I_MM'), ('I_RM', 'I_RM'), ('I_RR', 'I_RR')]:
            Iq = r[key_I]
            if log_y:
                pos, neg = Iq > 0, Iq < 0
                self._curves_iq[name].setData(
                    xq, np.where(pos, np.log10(np.where(pos, Iq, 1.0)), np.nan),
                    connect='finite')
                self._curves_iq_neg[name].setData(
                    xq, np.where(neg, np.log10(np.where(neg, -Iq, 1.0)), np.nan),
                    connect='finite')
                self._neg_markers[name].setData(
                    x=xq[neg], y=np.log10(-Iq[neg]))
            else:
                self._curves_iq[name].setData(xq, Iq)
                self._curves_iq_neg[name].setData([], [])
                self._neg_markers[name].setData([], [])

        # Additional curves + error bars for N>1 elements (remove old ones first)
        for c in getattr(self, '_extra_iq_curves', []):
            self._pw_iq.removeItem(c)
        self._extra_iq_curves = []
        for _, _, ei in getattr(self, '_extra_err_meta', []):
            self._pw_iq.removeItem(ei)
        self._extra_err_meta = []   # list of (name, mr_idx, ErrorBarItem)

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
                if log_y:
                    pos, neg = Iq > 0, Iq < 0
                    y_pos = np.where(pos, np.log10(np.where(pos, Iq, 1.0)), np.nan)
                    y_neg = np.where(neg, np.log10(np.where(neg, -Iq, 1.0)), np.nan)
                else:
                    y_pos, y_neg = Iq, np.full_like(Iq, np.nan)
                c_pos = pg.PlotCurveItem(
                    xq, y_pos,
                    name=name, pen=pg.mkPen(col, width=2), connect='finite')
                self._pw_iq.addItem(c_pos)
                self._extra_iq_curves.append(c_pos)
                if log_y and neg.any():
                    c_neg = pg.PlotCurveItem(
                        xq, y_neg, connect='finite',
                        pen=pg.mkPen(col, width=1.5,
                                     style=pg.QtCore.Qt.PenStyle.DashLine))
                    self._pw_iq.addItem(c_neg)
                    self._extra_iq_curves.append(c_neg)
                ei = pg.ErrorBarItem(x=np.array([]), y=np.array([]),
                                     top=np.array([]), bottom=np.array([]),
                                     pen=pg.mkPen(col, width=0.8))
                self._pw_iq.addItem(ei)
                self._extra_err_meta.append((name, idx, ei))

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
        xq  = self._xq(q)           # ScatterPlotItem lives in ViewBox coords
        cs  = np.minimum(cs, 2.0)   # keep infinite / huge violations on-screen
        self._curve_cs.setData(q, cs)
        self._pts_cs_ok.setData( x=xq[ok],  y=cs[ok])
        self._pts_cs_vio.setData(x=xq[vio], y=cs[vio])

    def _refresh_errorbars(self):
        if self._result is None:
            return
        r    = self._result
        q    = r['q']
        show = self._chk_errbar.isChecked()
        log  = self._chk_log.isChecked()
        xq   = self._xq(q)
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
            raw_Iq = r[key_I]
            Iq  = np.maximum(np.abs(raw_Iq), 1e-40)
            si  = r[key_s]

            # For I_RR clip the bottom bar at the CS floor so it never extends
            # into the physically forbidden region below I_RM²/|I_MM|.
            if name == 'I_RR' and cs_floor is not None:
                bot = np.minimum(si, np.maximum(Iq - cs_floor, 0.0))
            else:
                bot = si

            if show:
                if log:
                    # ErrorBarItem is a raw ViewBox item — needs log10 coords.
                    # Bars are drawn around |value| (negative points are shown as
                    # dashed |value|); require |I| > bot so log10(|I| - bot) is finite.
                    ok_log = (bot > 0) & (Iq > bot)
                    lIq = np.log10(Iq[ok_log])
                    top    = np.log10(Iq[ok_log] + si[ok_log]) - lIq
                    bottom = lIq - np.log10(Iq[ok_log] - bot[ok_log])
                    self._err_items[name].setData(
                        x=xq[ok_log], y=lIq, top=top, bottom=bottom)
                else:
                    ok = si > 0
                    self._err_items[name].setData(
                        x=xq[ok], y=raw_Iq[ok], top=si[ok], bottom=si[ok])
            else:
                self._err_items[name].setData(
                    x=np.array([]), y=np.array([]),
                    top=np.array([]), bottom=np.array([]))

        # Extra components for N>1 elements
        mr = r.get('multi')
        empty = np.array([])
        I_MM_g = mr['partials'][mr['names'].index('I_MM')] if mr is not None else None
        for comp_name, mr_idx, ei in getattr(self, '_extra_err_meta', []):
            if not show or mr is None:
                ei.setData(x=empty, y=empty, top=empty, bottom=empty)
                continue
            raw_Iq = mr['partials'][mr_idx]
            si     = mr['errors'][mr_idx]
            Iq     = np.maximum(np.abs(raw_Iq), 1e-40)

            # CS floor for I_RiRi components: bottom bar ≥ I_RiM² / |I_MM|
            rr_match = re.fullmatch(r'I_R(\d+)R\1', comp_name)
            if rr_match and I_MM_g is not None:
                i_str    = rr_match.group(1)
                rm_name  = f'I_R{i_str}M'
                if rm_name in mr['names']:
                    I_RiM_g  = mr['partials'][mr['names'].index(rm_name)]
                    cs_floor = I_RiM_g**2 / np.maximum(np.abs(I_MM_g), 1e-30)
                    bot = np.minimum(si, np.maximum(Iq - cs_floor, 0.0))
                else:
                    bot = si
            else:
                bot = si

            ok = (bot > 0) if log else (si > 0)
            if not ok.any():
                ei.setData(x=empty, y=empty, top=empty, bottom=empty)
                continue
            if log:
                # Also require Iq > bot so log10(Iq - bot) is finite.
                # Points where the lower bound reaches ≤ 0 are simply not drawn.
                ok_log = ok & (Iq > bot)
                if not ok_log.any():
                    ei.setData(x=empty, y=empty, top=empty, bottom=empty)
                    continue
                lq_ok  = xq[ok_log]
                lIq_ok = np.log10(Iq[ok_log])
                top    = np.log10(Iq[ok_log] + si[ok_log]) - lIq_ok
                bottom = lIq_ok - np.log10(Iq[ok_log] - bot[ok_log])
                ei.setData(x=lq_ok, y=lIq_ok, top=top, bottom=bottom)
            else:
                ei.setData(x=xq[ok], y=raw_Iq[ok], top=si[ok], bottom=si[ok])

    def _xq(self, q: np.ndarray) -> np.ndarray:
        """x coordinates for raw ViewBox items: log10(q) in log-q mode, else q."""
        return np.log10(q) if self._chk_logq.isChecked() else np.asarray(q)

    def _update_log(self, _=None):
        self._pw_iq.setLogMode(x=self._chk_logq.isChecked(),
                               y=self._chk_log.isChecked())
        self._pw_cs.setLogMode(x=self._chk_logq.isChecked(), y=False)
        if self._result is not None:
            self._update_iq_plot()   # PlotCurveItems need manual re-render on log toggle
        else:
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
        Iq_j  = self._I_matrix[:, idx].copy()
        sig_j = self._sig[:, idx].copy()

        # If a cross-edge β correction was applied to the decomposition, apply
        # the same scaling here so the marginal intensities are consistent with
        # the global partials extracted from the β-corrected data.
        r_beta = self._result
        if r_beta is not None:
            beta_v = r_beta.get('beta')
            eg_v   = r_beta.get('edge_groups')
            if beta_v is not None and eg_v is not None:
                scale  = beta_v[eg_v]      # (N_E,)
                Iq_j   = Iq_j  / scale
                sig_j  = sig_j / scale

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

            # Near-edge mask: keep only energies belonging to this element's edge.
            # For N>1 use the edge-group assignment so each panel shows only its
            # own edge energies (the fp-threshold alone also selects off-edge
            # energies from the other element's scan where f'_i is ~constant).
            eg_all = r.get('edge_groups_all') if r is not None else None
            if n_elems > 1 and eg_all is not None:
                mask = (eg_all == panel_i)
                if mask.sum() < 3:
                    mask = np.ones(len(fp), dtype=bool)   # fallback: show all
            else:
                fp_max   = float(np.max(fp))
                fp_range = float(np.max(fp) - np.min(fp))
                threshold = max(0.5, 0.05 * fp_range)
                mask = fp < (fp_max - threshold)
                if mask.sum() < 3:
                    mask = np.ones(len(fp), dtype=bool)   # fallback: show all

            fp_m   = fp[mask]
            fpp_m  = fpp[mask]
            Iq_m   = Iq_j[mask]
            sig_m  = sig_j[mask]

            # Compute marginal intensity for this element (on masked subset)
            if n_elems == 1 or r is None:
                I_marginal = Iq_m
            else:
                names   = r['multi']['names']
                parts_q = r['multi']['partials'][:, ri]
                I_MM_g  = parts_q[names.index('I_MM')]
                I_marginal = Iq_m.copy() - I_MM_g
                for j, other in enumerate(elems):
                    if j == panel_i:
                        continue
                    fp_j  = other['fp'][mask]
                    fpp_j = other['fpp'][mask]
                    nm_RjM  = f'I_R{j+1}M'
                    nm_RjRj = f'I_R{j+1}R{j+1}'
                    I_RjM  = parts_q[names.index(nm_RjM)]  if nm_RjM  in names else 0.0
                    I_RjRj = parts_q[names.index(nm_RjRj)] if nm_RjRj in names else 0.0
                    I_marginal -= 2.0 * fp_j * I_RjM + (fp_j**2 + fpp_j**2) * I_RjRj
                    # cross-term I_RiRj intentionally kept in I_marginal so the
                    # decomp fit curve (which includes it) matches the data

            # Local WLS solve on near-edge energies only
            A_loc = build_A(fp_m, fpp_m)
            w     = 1.0 / np.maximum(sig_m, 1e-30)
            Aw = A_loc * w[:, None]
            sol, _e = solve_partials(
                A_loc, I_marginal, 1.0 / w, 'WLS',
                nonneg=self._chk_nonneg.isChecked(),
                enforce_cs=self._chk_cs_enforce.isChecked())
            I_MM_j, I_RM_j, I_RR_j = sol
            try:
                loc_err = np.sqrt(np.maximum(np.diag(np.linalg.inv(Aw.T @ Aw)), 0))
            except np.linalg.LinAlgError:
                loc_err = np.full(3, np.nan)

            fp_dense  = np.linspace(fp_m.min() - 0.5, fp_m.max() + 0.5, 300)
            _sort     = np.argsort(fp_m)
            fpp_dense = np.interp(fp_dense, fp_m[_sort], fpp_m[_sort])
            I_fit     = (I_MM_j + 2*fp_dense*I_RM_j
                         + (fp_dense**2 + fpp_dense**2)*I_RR_j)

            new_items = []

            scatter = pg.ScatterPlotItem(
                x=fp_m, y=I_marginal, size=9,
                pen=pg.mkPen(col_data), brush=pg.mkBrush(col_data),
                name='data')
            pw.addItem(scatter)
            new_items.append(scatter)

            ei = pg.ErrorBarItem(x=fp_m, y=I_marginal, top=sig_m, bottom=sig_m,
                                 pen=pg.mkPen(col_data, width=1.0))
            pw.addItem(ei)
            new_items.append(ei)

            fit_c = pg.PlotDataItem(fp_dense, I_fit,
                                    pen=pg.mkPen(col_fit, width=2.0),
                                    name='local fit')
            pw.addItem(fit_c)
            new_items.append(fit_c)

            # Extract global decomp fit parameters before residuals (needed for
            # N>1 where we show global-model residuals, not local-fit residuals).
            I_MM_g = I_RiM_g = I_RiRi_g = None
            if r is not None and ri is not None:
                nm_RiM  = f'I_R{panel_i+1}M'
                nm_RiRi = f'I_R{panel_i+1}R{panel_i+1}'
                names_g  = r['multi']['names'] if r.get('multi') else ['I_MM', 'I_R1M', 'I_R1R1']
                parts_ri = r['multi']['partials'][:, ri] if r.get('multi') \
                           else np.array([r['I_MM'][ri], r['I_RM'][ri], r['I_RR'][ri]])
                I_MM_g   = parts_ri[names_g.index('I_MM')]
                I_RiM_g  = parts_ri[names_g.index(nm_RiM)]  if nm_RiM  in names_g else r['I_RM'][ri]
                I_RiRi_g = parts_ri[names_g.index(nm_RiRi)] if nm_RiRi in names_g else r['I_RR'][ri]

            # For N>1: find WLS-optimal intercept for the global slopes against
            # the marginal data.  Using I_MM_j (from local 3-param fit) would
            # mix local and global parameters and cause a systematic offset when
            # the local and global slopes differ.
            const_g = None
            if n_elems > 1 and I_RiM_g is not None:
                curve_g  = 2*fp_m*I_RiM_g + (fp_m**2 + fpp_m**2)*I_RiRi_g
                w2       = (1.0 / np.maximum(sig_m, 1e-30))**2
                const_g  = float(np.sum(w2 * (I_marginal - curve_g)) / np.sum(w2))

            # Weighted residuals.  For N=1 use local-fit model; for N>1 use
            # the global decomp fit with its optimally anchored intercept.
            if n_elems > 1 and const_g is not None:
                I_model_pts = (const_g + 2*fp_m*I_RiM_g
                               + (fp_m**2 + fpp_m**2)*I_RiRi_g)
            else:
                I_model_pts = (I_MM_j + 2*fp_m*I_RM_j
                               + (fp_m**2 + fpp_m**2)*I_RR_j)
            resid_w = (I_marginal - I_model_pts) / np.maximum(sig_m, 1e-30)
            pts_resid.setData(x=fp_m, y=resid_w)

            # Global decomp fit overlay
            if I_RiM_g is not None:
                if n_elems == 1:
                    I_comp = (I_MM_g + 2*fp_dense*I_RiM_g
                              + (fp_dense**2 + fpp_dense**2)*I_RiRi_g)
                else:
                    I_comp = (const_g + 2*fp_dense*I_RiM_g
                              + (fp_dense**2 + fpp_dense**2)*I_RiRi_g)
                comp_c = pg.PlotDataItem(
                    fp_dense, I_comp,
                    pen=pg.mkPen('#aed581', width=1.5,
                                 style=pg.QtCore.Qt.PenStyle.DashLine),
                    name='decomp. fit')
                pw.addItem(comp_c)
                new_items.append(comp_c)

            self._stuhr_items_by_panel[panel_i] = new_items
            txt = (f'q = {q_j:.4f} Å⁻¹  elem {panel_i+1}  |  local fit '
                   f'({int(mask.sum())}/{len(mask)} energies):  '
                   f'I_MM = {I_MM_j:.4g}±{loc_err[0]:.2g}   '
                   f'I_RM = {I_RM_j:.4g}±{loc_err[1]:.2g}   '
                   f'I_RR = {I_RR_j:.4g}±{loc_err[2]:.2g}  cm⁻¹')
            if I_RiM_g is not None and n_elems == 1:
                txt += (f'\nglobal decomposition (all {len(mask)} energies):  '
                        f"I_MM = {I_MM_g:.4g}±{r['s_MM'][ri]:.2g}   "
                        f"I_RM = {I_RiM_g:.4g}±{r['s_RM'][ri]:.2g}   "
                        f"I_RR = {I_RiRi_g:.4g}±{r['s_RR'][ri]:.2g}")
            lbl.setText(txt)

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
        # Refresh β₃ visibility based on actual element count
        self._beta_lbl_3.setVisible(n >= 3)
        self._beta_spin_3.setVisible(n >= 3)
        self._update_stuhrmann()

    def set_fp_fpp(self, fp: np.ndarray, fpp: np.ndarray):
        """Backward-compat shim — wraps into single-element list."""
        self.set_elements([{'fp': fp, 'fpp': fpp, 'Z': 0, 'label': 'R1'}])

    def get_result(self) -> dict | None:
        return self._result
