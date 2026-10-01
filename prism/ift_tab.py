"""
IFT tab — GNOM-style IFT of I_MM, I_RM, I_RR → p_MM, p_RM, p_RR
with propagated uncertainty bands on p(r).
"""

import traceback
import threading

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QCheckBox, QGridLayout, QTabWidget, QSizePolicy,
    QLineEdit, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QIntValidator
import pyqtgraph as pg

from .core.crosshair import add_crosshair
from .core.ift import ift_all, joint_ift, _sinc_kernel

_COLORS = {'p_MM': '#2196F3', 'p_RM': '#4CAF50', 'p_RR': '#F44336'}


class _IFTWorker(QObject):
    """Runs IFT in a Python thread (not QThread) to avoid macOS SIGBUS."""
    result_ready = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, kwargs: dict, joint: bool, parent=None):
        super().__init__(parent)
        self._kwargs = kwargs
        self._joint  = joint
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def isRunning(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        try:
            fn = joint_ift if self._joint else ift_all
            self.result_ready.emit(fn(**self._kwargs))
        except Exception:
            self.error_signal.emit(traceback.format_exc())


class _AlphaControl(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        self._chk = QCheckBox('Auto')
        self._chk.setChecked(True)
        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(3); self._spin.setRange(1e-6, 1e12)
        self._spin.setValue(1.0); self._spin.setEnabled(False)
        self._spin.setMaximumWidth(100)
        lay.addWidget(self._chk)
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
        self._worker: _IFTWorker | None = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        ctrl = QGroupBox('IFT parameters')
        ctrl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        g = QGridLayout(ctrl)
        g.setSpacing(3)
        g.setContentsMargins(6, 6, 6, 6)

        def _dmax_edit(default='600'):
            e = QLineEdit(default)
            e.setValidator(QDoubleValidator(10.0, 100000.0, 2))
            e.setMinimumWidth(60)
            e.setPlaceholderText('Å')
            return e

        def _nr_edit(default='200'):
            e = QLineEdit(default)
            e.setValidator(QIntValidator(1, 9999))
            e.setMinimumWidth(50)
            return e

        def _hdr(text):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            f = lbl.font(); f.setBold(True); lbl.setFont(f)
            return lbl

        def _rlbl(text):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return lbl

        # Column layout: col 0 = row label (fixed), cols 1-3 = I_MM / I_RM / I_RR
        g.setColumnStretch(0, 0)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(2, 1)
        g.setColumnStretch(3, 1)
        g.setColumnMinimumWidth(0, 80)

        # Row 0: column headers
        g.addWidget(_hdr(''),      0, 0)
        g.addWidget(_hdr('I_MM'),  0, 1)
        g.addWidget(_hdr('I_RM'),  0, 2)
        g.addWidget(_hdr('I_RR'),  0, 3)

        # Row 1: D_max
        g.addWidget(_rlbl('D_max (Å):'), 1, 0)
        self._spin_dmax_MM = _dmax_edit()
        self._spin_dmax_RM = _dmax_edit()
        self._spin_dmax_RR = _dmax_edit()
        for col, sp in enumerate([self._spin_dmax_MM, self._spin_dmax_RM, self._spin_dmax_RR], 1):
            g.addWidget(sp, 1, col)

        # Row 2: N_r
        g.addWidget(_rlbl('N_r:'), 2, 0)
        self._spin_nr_MM = _nr_edit()
        self._spin_nr_RM = _nr_edit()
        self._spin_nr_RR = _nr_edit()
        for col, sp in enumerate([self._spin_nr_MM, self._spin_nr_RM, self._spin_nr_RR], 1):
            g.addWidget(sp, 2, col)

        # Row 3: q range
        g.addWidget(_rlbl('q range (Å⁻¹):'), 3, 0)
        def _qrange_cell():
            qmin = QLineEdit(); qmin.setPlaceholderText('auto')
            qmax = QLineEdit(); qmax.setPlaceholderText('auto')
            for e in (qmin, qmax):
                e.setValidator(QDoubleValidator(0.0, 100.0, 6))
                e.setMinimumWidth(45)
            inner = QHBoxLayout()
            inner.setContentsMargins(0, 0, 0, 0); inner.setSpacing(2)
            inner.addWidget(qmin); inner.addWidget(QLabel('–')); inner.addWidget(qmax)
            wrap = QWidget(); wrap.setLayout(inner)
            return wrap, qmin, qmax

        wrap, self._qmin_MM, self._qmax_MM = _qrange_cell()
        g.addWidget(wrap, 3, 1)
        wrap, self._qmin_RM, self._qmax_RM = _qrange_cell()
        g.addWidget(wrap, 3, 2)
        wrap, self._qmin_RR, self._qmax_RR = _qrange_cell()
        g.addWidget(wrap, 3, 3)

        # Row 4: χ² factor (over-regularisation for noisy components)
        g.addWidget(_rlbl('χ² ×:'), 4, 0)
        self._chi2f_MM = QLineEdit('1'); self._chi2f_MM.setToolTip(
            'Morozov target multiplier (>1 = more smoothing). Default 1.')
        self._chi2f_RM = QLineEdit('1'); self._chi2f_RM.setToolTip(
            'Morozov target multiplier (>1 = more smoothing). Default 1.')
        self._chi2f_RR = QLineEdit('1'); self._chi2f_RR.setToolTip(
            'Morozov target multiplier (>1 = more smoothing). Default 1.')
        for col, e in enumerate([self._chi2f_MM, self._chi2f_RM, self._chi2f_RR], 1):
            e.setValidator(QDoubleValidator(0.01, 1e6, 3))
            g.addWidget(e, 4, col)

        # Row 5: SG smooth window
        g.addWidget(_rlbl('SG smooth:'), 5, 0)
        self._smooth_MM = QLineEdit(); self._smooth_MM.setPlaceholderText('off')
        self._smooth_RM = QLineEdit(); self._smooth_RM.setPlaceholderText('off')
        self._smooth_RR = QLineEdit(); self._smooth_RR.setPlaceholderText('off')
        for col, e in enumerate([self._smooth_MM, self._smooth_RM, self._smooth_RR], 1):
            e.setValidator(QIntValidator(3, 9999))
            e.setToolTip('Savitzky-Golay window (odd integer ≥ 3, blank = off)')
            g.addWidget(e, 5, col)

        # Row 6: Background — fixed subtract value + optional fit
        g.addWidget(_rlbl('Bg:'), 6, 0)
        self._edit_bg_MM = QLineEdit(); self._edit_bg_MM.setPlaceholderText('0.0')
        self._edit_bg_RM = QLineEdit(); self._edit_bg_RM.setPlaceholderText('0.0')
        self._edit_bg_RR = QLineEdit(); self._edit_bg_RR.setPlaceholderText('0.0')
        self._chk_bg_MM  = QCheckBox('Fit')
        self._chk_bg_RM  = QCheckBox('Fit')
        self._chk_bg_RR  = QCheckBox('Fit')
        for col, (edit, chk) in enumerate(zip(
                [self._edit_bg_MM, self._edit_bg_RM, self._edit_bg_RR],
                [self._chk_bg_MM,  self._chk_bg_RM,  self._chk_bg_RR]), 1):
            edit.setValidator(QDoubleValidator(-1e9, 1e9, 6))
            edit.setMinimumWidth(50)
            edit.setToolTip('Fixed background to subtract before IFT')
            chk.setToolTip('Jointly fit a residual constant background')
            inner = QHBoxLayout()
            inner.setContentsMargins(0, 0, 0, 0)
            inner.setSpacing(4)
            inner.addWidget(edit)
            inner.addWidget(chk)
            wrap = QWidget(); wrap.setLayout(inner)
            g.addWidget(wrap, 6, col)

        # Row 7: α (independent — one per column)
        self._lbl_alpha = _rlbl('α:')
        g.addWidget(self._lbl_alpha, 7, 0)
        self._alpha_MM = _AlphaControl()
        self._alpha_RM = _AlphaControl()
        self._alpha_RR = _AlphaControl()
        for col, ac in enumerate([self._alpha_MM, self._alpha_RM, self._alpha_RR], 1):
            g.addWidget(ac, 7, col)

        # Row 8: α (joint — spans all data columns, hidden by default)
        self._lbl_alpha_joint = _rlbl('α (shared):')
        self._lbl_alpha_joint.setVisible(False)
        g.addWidget(self._lbl_alpha_joint, 8, 0)
        self._alpha_joint = _AlphaControl()
        self._alpha_joint.setVisible(False)
        g.addWidget(self._alpha_joint, 8, 1, 1, 3)

        # Row 9: all toggles + Run button + status on one line
        self._chk_joint = QCheckBox('Joint fit')
        self._chk_joint.setToolTip(
            'Solve p_MM, p_RM, p_RR simultaneously.\n'
            'One shared α chosen by the Morozov discrepancy principle\n'
            'on the combined 3·NQ residual.')
        self._chk_bands = QCheckBox('σ bands')
        self._chk_bands.setChecked(True)
        self._chk_norm = QCheckBox('Normalize')
        self._chk_norm.setChecked(True)
        self._btn_run = QPushButton('Run IFT')
        self._lbl_status = QLabel('Run decomposition first')
        self._lbl_status.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        bot = QHBoxLayout()
        bot.setSpacing(8)
        for w in (self._chk_joint, self._chk_bands, self._chk_norm):
            bot.addWidget(w)
        bot.addStretch()
        bot.addWidget(self._btn_run)
        bot.addWidget(self._lbl_status)
        g.addLayout(bot, 9, 0, 1, 4)

        # Wire signals
        self._chk_joint.toggled.connect(self._on_joint_toggled)
        self._chk_bands.toggled.connect(self._refresh_bands)
        self._chk_norm.toggled.connect(self._on_norm_toggled)
        self._btn_run.clicked.connect(self._run)
        for e in (self._smooth_MM, self._smooth_RM, self._smooth_RR):
            e.editingFinished.connect(self._on_norm_toggled)

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
        self._pw_fit.setLogMode(x=True, y=True)
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
    def _on_joint_toggled(self, joint: bool):
        for w in (self._lbl_alpha, self._alpha_MM, self._alpha_RM, self._alpha_RR):
            w.setVisible(not joint)
        self._lbl_alpha_joint.setVisible(joint)
        self._alpha_joint.setVisible(joint)

    def _run(self):
        if self._q is None:
            self._lbl_status.setText('No decomposition result yet')
            return
        if self._worker and self._worker.isRunning():
            return

        Dmax_MM = float(self._spin_dmax_MM.text() or 600)
        Dmax_RM = float(self._spin_dmax_RM.text() or 600)
        Dmax_RR = float(self._spin_dmax_RR.text() or 600)
        Nr_MM   = max(1, int(self._spin_nr_MM.text() or 200))
        Nr_RM   = max(1, int(self._spin_nr_RM.text() or 200))
        Nr_RR   = max(1, int(self._spin_nr_RR.text() or 200))
        joint   = self._chk_joint.isChecked()
        fit_bg_MM = self._chk_bg_MM.isChecked()
        fit_bg_RM = self._chk_bg_RM.isChecked()
        fit_bg_RR = self._chk_bg_RR.isChecked()

        # Subtract fixed background before IFT
        sub_MM = float(self._edit_bg_MM.text() or 0)
        sub_RM = float(self._edit_bg_RM.text() or 0)
        sub_RR = float(self._edit_bg_RR.text() or 0)

        # Apply per-component q-range masks
        def _qmask(qmin_e, qmax_e):
            qlo = float(qmin_e.text()) if qmin_e.text() else self._q[0]
            qhi = float(qmax_e.text()) if qmax_e.text() else self._q[-1]
            return (self._q >= qlo) & (self._q <= qhi)

        msk_MM = _qmask(self._qmin_MM, self._qmax_MM)
        msk_RM = _qmask(self._qmin_RM, self._qmax_RM)
        msk_RR = _qmask(self._qmin_RR, self._qmax_RR)

        q_MM  = self._q[msk_MM];  I_MM = (self._I_MM - sub_MM)[msk_MM]
        q_RM  = self._q[msk_RM];  I_RM = (self._I_RM - sub_RM)[msk_RM]
        q_RR  = self._q[msk_RR];  I_RR = (self._I_RR - sub_RR)[msk_RR]
        s_MM  = self._s_MM[msk_MM] if self._s_MM is not None else None
        s_RM  = self._s_RM[msk_RM] if self._s_RM is not None else None
        s_RR  = self._s_RR[msk_RR] if self._s_RR is not None else None
        # store masks for back-transform reconstruction
        self._last_masks  = (msk_MM, msk_RM, msk_RR)
        self._last_q_comp = (q_MM,  q_RM,  q_RR)

        # joint_ift requires a shared q; fall back to ift_all if masks differ
        masks_equal = (np.array_equal(msk_MM, msk_RM) and
                       np.array_equal(msk_MM, msk_RR))
        chi2f_MM = float(self._chi2f_MM.text() or 1)
        chi2f_RM = float(self._chi2f_RM.text() or 1)
        chi2f_RR = float(self._chi2f_RR.text() or 1)

        if joint and masks_equal:
            kwargs = dict(
                q=q_MM, I_MM=I_MM, I_RM=I_RM, I_RR=I_RR,
                Dmax_MM=Dmax_MM, Dmax_RM=Dmax_RM, Dmax_RR=Dmax_RR,
                Nr_MM=Nr_MM, Nr_RM=Nr_RM, Nr_RR=Nr_RR,
                alpha=self._alpha_joint.alpha(),
                sigma_MM=s_MM, sigma_RM=s_RM, sigma_RR=s_RR,
                fit_bg_MM=fit_bg_MM, fit_bg_RM=fit_bg_RM, fit_bg_RR=fit_bg_RR,
                chi2_factor=max(chi2f_MM, chi2f_RM, chi2f_RR),
            )
            use_joint = True
        else:
            if joint and not masks_equal:
                self._lbl_status.setText(
                    'q ranges differ — running independent IFT instead of joint…')
            kwargs = dict(
                q=self._q, I_MM=I_MM, I_RM=I_RM, I_RR=I_RR,
                Dmax_MM=Dmax_MM, Dmax_RM=Dmax_RM, Dmax_RR=Dmax_RR,
                Nr_MM=Nr_MM, Nr_RM=Nr_RM, Nr_RR=Nr_RR,
                alpha_MM=self._alpha_MM.alpha(),
                alpha_RM=self._alpha_RM.alpha(),
                alpha_RR=self._alpha_RR.alpha(),
                sigma_MM=s_MM, sigma_RM=s_RM, sigma_RR=s_RR,
                fit_bg_MM=fit_bg_MM, fit_bg_RM=fit_bg_RM, fit_bg_RR=fit_bg_RR,
                q_MM=q_MM, q_RM=q_RM, q_RR=q_RR,
                chi2_factor_MM=chi2f_MM, chi2_factor_RM=chi2f_RM, chi2_factor_RR=chi2f_RR,
            )
            use_joint = False

        self._btn_run.setEnabled(False)
        self._lbl_status.setText('Running joint IFT…' if use_joint else 'Running IFT…')
        self._pending = (Dmax_MM, Dmax_RM, Dmax_RR, Nr_MM, Nr_RM, Nr_RR, use_joint)

        self._worker = _IFTWorker(kwargs, use_joint, self)
        self._worker.result_ready.connect(self._on_result_raw)
        self._worker.error_signal.connect(self._on_error)
        self._worker.start()

    def _on_result_raw(self, result: dict):
        Dmax_MM, Dmax_RM, Dmax_RR, Nr_MM, Nr_RM, Nr_RR, joint = self._pending
        self._on_result(result, Dmax_MM, Dmax_RM, Dmax_RR, Nr_MM, Nr_RM, Nr_RR, joint)

    def _on_result(self, result: dict,
                   Dmax_MM: float, Dmax_RM: float, Dmax_RR: float,
                   Nr_MM: int, Nr_RM: int, Nr_RR: int, joint: bool):
        self._btn_run.setEnabled(True)
        try:
            self._result = result
            self._update_pr_plot(result)
            self._update_backtransform(result)
            self.ift_done.emit(result)
            dmax_str = (f'D_max MM/RM/RR={Dmax_MM:.0f}/{Dmax_RM:.0f}/{Dmax_RR:.0f} Å'
                        if len({Dmax_MM, Dmax_RM, Dmax_RR}) > 1
                        else f'D_max={Dmax_MM:.0f} Å')
            nr_str = (f'N_r MM/RM/RR={Nr_MM}/{Nr_RM}/{Nr_RR}'
                      if len({Nr_MM, Nr_RM, Nr_RR}) > 1
                      else f'N_r={Nr_MM}')
            if joint:
                a    = result.get('alpha', float('nan'))
                chi2 = result.get('chi2_reduced', float('nan'))
                self._lbl_status.setText(
                    f'Joint done — {dmax_str}, {nr_str}, α={a:.3g}, χ²_red={chi2:.3f}')
            else:
                self._lbl_status.setText(f'Done — {dmax_str}, {nr_str}')
        except Exception as e:
            import traceback as _tb
            self._lbl_status.setText(f'Plot error: {e}')
            print(_tb.format_exc())

    def _on_error(self, tb: str):
        self._btn_run.setEnabled(True)
        self._lbl_status.setText('IFT error — see console')
        print(tb)

    @staticmethod
    def _sg(arr: np.ndarray, window: int) -> np.ndarray:
        from scipy.signal import savgol_filter
        win = window if window % 2 == 1 else window + 1
        win = max(win, 5)
        poly = min(3, win - 2)
        return savgol_filter(arr, win, poly)

    def _update_pr_plot(self, res: dict):
        normalize = self._chk_norm.isChecked()
        smooth_wins = {
            'p_MM': self._smooth_MM.text(),
            'p_RM': self._smooth_RM.text(),
            'p_RR': self._smooth_RR.text(),
        }
        # p_MM and p_RR are auto-correlations → physically non-negative
        _nonneg = {'p_MM', 'p_RR'}
        for name, r_key, p_key, sp_key in [
            ('p_MM', 'r_MM', 'p_MM', 'sp_MM'),
            ('p_RM', 'r_RM', 'p_RM', 'sp_RM'),
            ('p_RR', 'r_RR', 'p_RR', 'sp_RR'),
        ]:
            r  = res[r_key]
            p  = res[p_key].copy()
            sp = res[sp_key].copy()
            win_txt = smooth_wins[p_key]
            if win_txt:
                win = int(win_txt)
                if win >= 3 and len(p) >= win:
                    p  = self._sg(p,  win)
                    sp = self._sg(sp, win)
                    sp = np.maximum(sp, 0.0)
                    if p_key in _nonneg:
                        p = np.maximum(p, 0.0)
            scale = float(np.max(np.abs(p))) if normalize else 1.0
            if scale == 0.0:
                scale = 1.0
            self._curves_pr[name].setData(r, p / scale)
            self._upper_pr[name].setData(r, (p + sp) / scale)
            self._lower_pr[name].setData(r, (p - sp) / scale)
        ylabel = 'p(r) / max|p(r)|' if normalize else 'p(r) (cm⁻¹ Å⁻¹)'
        self._pw_pr.setLabel('left', ylabel)
        self._refresh_bands()

    def _on_norm_toggled(self, _ = None):
        if self._result is not None:
            self._update_pr_plot(self._result)

    def _refresh_bands(self):
        show = self._chk_bands.isChecked()
        for fb in self._bands_pr.values():
            fb.setVisible(show)

    def _update_backtransform(self, res: dict):
        for c in self._fit_curves:
            self._pw_fit.removeItem(c)
        self._fit_curves.clear()

        sub_vals = {
            'bg_MM': float(self._edit_bg_MM.text() or 0),
            'bg_RM': float(self._edit_bg_RM.text() or 0),
            'bg_RR': float(self._edit_bg_RR.text() or 0),
        }
        # Per-component q-grids used for IFT (may be masked)
        q_comps = {'r_MM': None, 'r_RM': None, 'r_RR': None}
        if hasattr(self, '_last_q_comp'):
            q_comps = {'r_MM': self._last_q_comp[0],
                       'r_RM': self._last_q_comp[1],
                       'r_RR': self._last_q_comp[2]}

        for name, Iq, r_key, pr_key, bg_key, col in [
            ('I_MM', self._I_MM, 'r_MM', 'p_MM', 'bg_MM', '#2196F3'),
            ('I_RM', self._I_RM, 'r_RM', 'p_RM', 'bg_RM', '#4CAF50'),
            ('I_RR', self._I_RR, 'r_RR', 'p_RR', 'bg_RR', '#F44336'),
        ]:
            r_i    = res[r_key]
            dr_i   = r_i[1] - r_i[0]
            fit_bg = res.get(bg_key, 0.0)
            sub    = sub_vals[bg_key]

            # Data in original units (full q-range)
            dc = self._pw_fit.plot(
                self._q, np.maximum(np.abs(Iq), 1e-40),
                pen=pg.mkPen(col, width=2), name=f'{name} data')

            # Fit evaluated on the q-grid used by IFT, reconstructed in original units
            q_fit  = q_comps[r_key] if q_comps[r_key] is not None else self._q
            K_i    = _sinc_kernel(q_fit, r_i, dr_i)
            fit    = K_i @ res[pr_key] + fit_bg + sub
            fc = self._pw_fit.plot(
                q_fit, np.maximum(np.abs(fit), 1e-40),
                pen=pg.mkPen(col, width=1.5, style=Qt.PenStyle.DashLine),
                name=f'{name} fit')
            self._fit_curves += [dc, fc]

            # When a background was applied, also show background-subtracted
            # data vs the raw IFT fit so fit quality is directly visible
            total_bg = sub + fit_bg
            if abs(total_bg) > 0:
                Iq_sub  = Iq - sub
                fit_sub = K_i @ res[pr_key] + fit_bg
                ds = self._pw_fit.plot(
                    self._q, np.maximum(np.abs(Iq_sub), 1e-40),
                    pen=pg.mkPen(col, width=1, style=Qt.PenStyle.DotLine),
                    name=f'{name} data−bg')
                fs = self._pw_fit.plot(
                    q_fit, np.maximum(np.abs(fit_sub), 1e-40),
                    pen=pg.mkPen(col, width=1, style=Qt.PenStyle.DashDotLine),
                    name=f'{name} fit−bg')
                self._fit_curves += [ds, fs]

    # ── Public API ────────────────────────────────────────────────────────────
    def set_partials(self, q, I_MM, I_RM, I_RR,
                     s_MM=None, s_RM=None, s_RR=None):
        self._q    = q
        self._I_MM = I_MM; self._I_RM = I_RM; self._I_RR = I_RR
        self._s_MM = s_MM; self._s_RM = s_RM; self._s_RR = s_RR
        if len(q) > 0:
            dmax_guess = str(round(np.pi / q[0] * 2 / 50) * 50)
            for e in (self._spin_dmax_MM, self._spin_dmax_RM, self._spin_dmax_RR):
                e.setText(dmax_guess)
        self._lbl_status.setText('Ready — adjust D_max and click Run IFT')
