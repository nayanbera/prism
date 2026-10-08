"""Data loading tab — add I(q,E) files, assign energies, preview plots."""

from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QAbstractItemView,
    QLabel, QDoubleSpinBox, QMessageBox, QSplitter, QCheckBox, QSlider,
    QComboBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
import pyqtgraph as pg

from .core.io import load_file, interpolate_to_common_q
from .core.crosshair import add_crosshair



def _energy_color(t: float) -> tuple[int, int, int]:
    """Blue (t=0) → cyan → green → yellow → red (t=1)."""
    stops = [(0.0, (0, 0, 255)), (0.33, (0, 200, 255)),
             (0.5, (0, 220, 0)), (0.67, (255, 220, 0)), (1.0, (255, 0, 0))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]; t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(int(c0[k] + f * (c1[k] - c0[k])) for k in range(3))
    return (255, 0, 0)


class DataTab(QWidget):
    datasets_changed = pyqtSignal(list)   # active datasets sorted by E
    fp_fpp_ready     = pyqtSignal(list)   # list of {'Z','label','fp','fpp'} dicts

    # Column indices — col 0 is the ✓ checkbox (Use), rest are info
    _COL_USE    = 0
    _COL_ENERGY = 1
    _COL_QRANGE = 2
    _COL_NPTS   = 3
    _COL_SNR    = 4
    _COL_FILE   = 5
    COLS = ['Use', 'Energy (keV)', 'q range (Å⁻¹)', 'N pts', 'σ/I', 'File']

    def __init__(self, parent=None):
        super().__init__(parent)
        self._datasets: list[dict]  = []
        self._curves:   list        = []
        # Per-element state (3 slots)
        self._elem_Z:     list[int]        = [0, 0, 0]
        self._elem_shell: list[str]        = ['', '', '']
        self._elem_fp:    list             = [np.array([]), np.array([]), np.array([])]
        self._elem_fpp:   list             = [np.array([]), np.array([]), np.array([])]
        self._elem_chk:   list             = []   # filled in _build_ui
        self._elem_combo: list             = []
        self._elem_lbl_edge:   list        = []
        self._elem_lbl_status: list        = []
        self._preset_maps: list[dict]      = [{}, {}, {}]
        # Fluorescence background estimate (None until Estimate is pressed)
        self._fluor: dict | None           = None
        self._fluor_lines: list            = []   # (InfiniteLine, value)
        self._fluor_q_touched              = False
        # Stuhrmann slice panels
        self._slice_panels: list           = []   # pg.PlotWidget per active elem
        self._slice_items:  list           = []   # list of lists of items
        self._stuhr_right_split            = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        lay = QVBoxLayout(self)

        # Toolbar
        btn_row = QHBoxLayout()
        self._btn_add       = QPushButton('Add files…')
        self._btn_remove    = QPushButton('Remove selected')
        self._btn_clear     = QPushButton('Clear all')
        self._btn_sel_all   = QPushButton('Select all')
        self._btn_sel_all.setToolTip("Tick the 'Use' box of every dataset.")
        self._btn_desel_all = QPushButton('Deselect all')
        self._btn_desel_all.setToolTip("Untick the 'Use' box of every dataset.")
        self._btn_save_list = QPushButton('Save list…')
        self._btn_save_list.setToolTip(
            'Save the current file list (paths, energies, element selections)\n'
            'to a JSON file so the session can be restored in one click.')
        self._btn_load_list = QPushButton('Load list…')
        self._btn_load_list.setToolTip(
            'Load a previously saved file list — reloads all data files\n'
            'and restores element selections and q-grid settings.')
        self._lbl_status = QLabel()
        for w in (self._btn_add, self._btn_remove, self._btn_clear,
                  self._btn_sel_all, self._btn_desel_all,
                  self._btn_save_list, self._btn_load_list, self._lbl_status):
            btn_row.addWidget(w)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # ── Resonant element rows (up to 3 elements) ─────────────────────────
        elem_grid = QHBoxLayout()
        for i in range(3):
            grp = QHBoxLayout()
            grp.addWidget(QLabel(f'Element {i+1}:'))
            chk = QCheckBox()
            if i == 0:
                chk.setChecked(True)
                chk.setEnabled(False)
            else:
                chk.setChecked(False)
            self._elem_chk.append(chk)
            grp.addWidget(chk)

            combo = QComboBox()
            combo.addItem('(select element)')
            combo.setMinimumWidth(220)
            if i != 0:
                combo.setEnabled(False)
            self._elem_combo.append(combo)
            grp.addWidget(combo)

            lbl_edge = QLabel('—')
            self._elem_lbl_edge.append(lbl_edge)
            grp.addWidget(QLabel('Edge:'))
            grp.addWidget(lbl_edge)

            lbl_st = QLabel()
            lbl_st.setStyleSheet('font-size: 11px;')
            lbl_st.setSizePolicy(QSizePolicy.Policy.Ignored,
                                  QSizePolicy.Policy.Preferred)
            self._elem_lbl_status.append(lbl_st)
            grp.addWidget(lbl_st)

            if i < 2:
                sep = QLabel(' | ')
                grp.addWidget(sep)

            elem_grid.addLayout(grp)

            # Wire signals — capture i in default arg
            combo.currentIndexChanged.connect(
                lambda idx, ei=i: self._on_element_preset(idx, ei))
            chk.toggled.connect(
                lambda checked, ei=i: self._on_elem_chk_toggled(checked, ei))

        elem_grid.addStretch()
        lay.addLayout(elem_grid)

        # Splitter: table on left, plot on right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel: table + q-range controls ──────────────────────────────
        left = QWidget()
        left.setMaximumWidth(520)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, len(self.COLS))
        self._table.setHorizontalHeaderLabels(self.COLS)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(self._COL_USE,    QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_ENERGY, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_QRANGE, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_NPTS,   QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_SNR,    QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_FILE,   QHeaderView.ResizeMode.Interactive)
        hdr.setDefaultSectionSize(160)   # initial File column width
        self._table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        left_lay.addWidget(self._table)

        qrow = QHBoxLayout()
        qrow.addWidget(QLabel('Common q range:'))
        self._q_min = QDoubleSpinBox(); self._q_min.setDecimals(5)
        self._q_min.setRange(0, 10); self._q_min.setValue(0.005)
        self._q_max = QDoubleSpinBox(); self._q_max.setDecimals(4)
        self._q_max.setRange(0, 10); self._q_max.setValue(0.35)
        self._n_pts = QDoubleSpinBox(); self._n_pts.setDecimals(0)
        self._n_pts.setRange(10, 5000); self._n_pts.setValue(300)
        for lbl, sp in [('q_min', self._q_min), ('q_max', self._q_max),
                         ('N pts', self._n_pts)]:
            qrow.addWidget(QLabel(lbl)); qrow.addWidget(sp)
        qrow.addStretch()
        left_lay.addLayout(qrow)

        frow = QHBoxLayout()
        frow.addWidget(QLabel('Fluorescence bg q range:'))
        self._fl_qmin = QDoubleSpinBox(); self._fl_qmin.setDecimals(5)
        self._fl_qmin.setRange(0, 10); self._fl_qmin.setValue(0.1)
        self._fl_qmax = QDoubleSpinBox(); self._fl_qmax.setDecimals(5)
        self._fl_qmax.setRange(0, 10); self._fl_qmax.setValue(0.145)
        for lbl, sp in [('from', self._fl_qmin), ('to', self._fl_qmax)]:
            frow.addWidget(QLabel(lbl)); frow.addWidget(sp)
        frow.addStretch()
        left_lay.addLayout(frow)

        frow2 = QHBoxLayout()
        self._btn_fl_est = QPushButton('Estimate')
        self._btn_fl_est.setToolTip(
            'bb = mean I in the q range at the energy farthest from the edge(s).\n'
            'be = mean I in the same q range at every energy.\n'
            'Shown as dashed horizontal lines (bb in white).')
        self._btn_fl_sub = QPushButton('Subtract')
        self._btn_fl_sub.setToolTip(
            'Subtract (be − bb) from every dataset; the corrected data replace\n'
            'the plotted data and are used by all later tabs. Always computed\n'
            'from the original data, so pressing it again does not double-subtract.')
        self._lbl_fl = QLabel()
        self._lbl_fl.setWordWrap(True)
        frow2.addWidget(self._btn_fl_est)
        frow2.addWidget(self._btn_fl_sub)
        frow2.addWidget(self._lbl_fl, 1)
        left_lay.addLayout(frow2)

        splitter.addWidget(left)
        splitter.setStretchFactor(0, 0)   # left panel: don't stretch

        # ── Right panel: waterfall (left) + Stuhrmann slice (right) ──────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)

        plot_ctrl = QHBoxLayout()
        self._chk_logx = QCheckBox('Log q')
        self._chk_logx.setChecked(True)
        self._chk_logy = QCheckBox('Log I')
        self._chk_logy.setChecked(True)
        self._chk_logx.toggled.connect(self._update_log)
        self._chk_logy.toggled.connect(self._update_log)
        self._chk_slice_logy = QCheckBox('Log I (slice)')
        self._chk_slice_logy.setChecked(True)
        self._chk_slice_logy.toggled.connect(self._on_slice_logy_toggled)
        plot_ctrl.addWidget(self._chk_logx)
        plot_ctrl.addWidget(self._chk_logy)
        plot_ctrl.addWidget(QLabel('   '))
        plot_ctrl.addWidget(self._chk_slice_logy)
        plot_ctrl.addStretch()
        right_lay.addLayout(plot_ctrl)

        inner_split = QSplitter(Qt.Orientation.Horizontal)

        # Left: I(q,E) waterfall with vertical q-selection line
        self._pw = pg.PlotWidget(title='I(q, E) — all loaded datasets')
        self._pw.setLabel('bottom', 'q (Å⁻¹)')
        self._pw.setLabel('left', 'I(q) (cm⁻¹)')
        self._pw.setLogMode(x=True, y=True)
        self._st_vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#ffffff', width=1.5,
                         style=Qt.PenStyle.DashLine))
        self._pw.addItem(self._st_vline)
        inner_split.addWidget(self._pw)

        # Right: vertical splitter containing N element slice panels
        self._stuhr_right_split = QSplitter(Qt.Orientation.Vertical)
        inner_split.addWidget(self._stuhr_right_split)
        inner_split.setSizes([580, 420])

        right_lay.addWidget(inner_split, 1)

        # Q slider
        sl_row = QHBoxLayout()
        sl_row.addWidget(QLabel('q  ='))
        self._sl_q = QSlider(Qt.Orientation.Horizontal)
        self._sl_q.setMinimum(0)
        self._sl_q.setMaximum(0)
        self._sl_q.setValue(0)
        self._sl_q.setEnabled(False)
        self._sl_q.valueChanged.connect(self._on_q_slider)
        sl_row.addWidget(self._sl_q, 1)
        self._lbl_q_val = QLabel('—')
        self._lbl_q_val.setFixedWidth(170)
        sl_row.addWidget(self._lbl_q_val)
        right_lay.addLayout(sl_row)

        self._coord_lbl = QLabel()
        self._coord_lbl.setStyleSheet('font-family: monospace; color: #aaaaaa;')
        self._coord_lbl.setSizePolicy(QSizePolicy.Policy.Ignored,
                                      QSizePolicy.Policy.Preferred)
        right_lay.addWidget(self._coord_lbl)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)   # right panel: absorbs all extra space
        splitter.setSizes([420, 580])

        lay.addWidget(splitter, 1)

        self._ch = [add_crosshair(self._pw, label=self._coord_lbl)]
        self._slice_crosshairs: list = []   # extended in _rebuild_slice_panels

        # ── Signal connections ────────────────────────────────────────────────
        self._btn_add.clicked.connect(self._add_files)
        self._btn_remove.clicked.connect(self._remove_selected)
        self._btn_clear.clicked.connect(self._clear)
        self._btn_sel_all.clicked.connect(lambda: self._set_all_active(True))
        self._btn_desel_all.clicked.connect(lambda: self._set_all_active(False))
        self._btn_save_list.clicked.connect(self._save_filelist)
        self._btn_load_list.clicked.connect(self._load_filelist)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.currentItemChanged.connect(
            lambda cur, _: self._on_row_changed(cur.row() if cur else -1))
        self._btn_fl_est.clicked.connect(self._fluor_estimate)
        self._btn_fl_sub.clicked.connect(self._fluor_subtract)
        self._fl_qmin.valueChanged.connect(self._on_fluor_q_changed)
        self._fl_qmax.valueChanged.connect(self._on_fluor_q_changed)
        self._q_min.valueChanged.connect(self._update_slider_range)
        self._q_max.valueChanged.connect(self._update_slider_range)
        self._n_pts.valueChanged.connect(self._update_slider_range)

        # Create initial single slice panel
        self._rebuild_slice_panels(1)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Open SAXS data files', '',
            'Data files (*.dat *.txt *.chi *.h5 *.hdf5 *.nxs);;All files (*)')
        for p in paths:
            try:
                ds = load_file(p)
                ds['active'] = True          # "Use" checkbox state
                self._datasets.append(ds)
            except Exception as e:
                QMessageBox.warning(self, 'Load error', f'{Path(p).name}: {e}')
        self._sort_and_emit()

    def _remove_selected(self):
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()},
                      reverse=True)
        if not rows:
            QMessageBox.information(
                self, 'Nothing selected',
                'Click a row in the table to select it (Ctrl/Shift-click for '
                'several), then press Remove selected.')
            return
        for r in rows:
            self._table.removeRow(r)
            curve = self._curves.pop(r)
            self._pw.removeItem(curve)
            self._datasets.pop(r)
        self._sort_and_emit()

    def _set_all_active(self, checked: bool):
        """Tick/untick the Use box of every dataset (one downstream update)."""
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._table.blockSignals(True)
        for r, ds in enumerate(self._datasets):
            ds['active'] = checked
            self._table.item(r, self._COL_USE).setCheckState(state)
            if r < len(self._curves):
                self._curves[r].setVisible(checked)
                self._curves[r].setOpacity(1.0 if checked else 0.25)
        self._table.blockSignals(False)
        self._emit_active()

    def _clear(self):
        self._table.setRowCount(0)
        for c in self._curves:
            self._pw.removeItem(c)
        self._curves.clear()
        self._datasets.clear()
        self._sort_and_emit()

    def _on_item_changed(self, item: QTableWidgetItem):
        row = item.row()
        if row >= len(self._datasets):
            return
        col = item.column()

        if col == self._COL_USE:
            # Checkbox toggled
            checked = item.checkState() == Qt.CheckState.Checked
            self._datasets[row]['active'] = checked
            if row < len(self._curves):
                c = self._curves[row]
                c.setVisible(checked)
                c.setOpacity(1.0 if checked else 0.25)
            self._emit_active()

        elif col == self._COL_ENERGY:
            try:
                e = float(item.text())
                self._datasets[row]['energy'] = e
                item.setBackground(QColor('white'))
                self._recolor_curves()
                self._emit_active()
            except ValueError:
                item.setBackground(QColor('#ffcccc'))

    def _on_row_changed(self, row: int):
        """Highlight the selected dataset's curve."""
        for i, c in enumerate(self._curves):
            if i == row:
                c.setPen(pg.mkPen(c.opts['pen'].color(), width=3))
            else:
                c.setPen(pg.mkPen(c.opts['pen'].color(), width=1.5))

    def _update_log(self):
        self._pw.setLogMode(x=self._chk_logx.isChecked(),
                            y=self._chk_logy.isChecked())
        self._update_st_vline()
        self._place_fluor_lines()

    # ── Fluorescence background ───────────────────────────────────────────────
    def _on_fluor_q_changed(self):
        self._fluor_q_touched = True
        self._clear_fluor()

    def _clear_fluor(self):
        for line, _ in self._fluor_lines:
            self._pw.removeItem(line)
        self._fluor_lines.clear()
        self._fluor = None
        self._lbl_fl.setText('')

    def _place_fluor_lines(self):
        """(Re)position dashed lines for the current log-y setting."""
        logy = self._chk_logy.isChecked()
        for line, val in self._fluor_lines:
            ok = bool(val > 0 or not logy)
            line.setVisible(ok)
            if ok:
                line.setPos(np.log10(val) if logy else val)

    def _fluor_estimate(self) -> bool:
        """Compute bb (farthest-from-edge energy) and be for every energy."""
        from .anomalous_tab import _edge_energy_keV
        self._clear_fluor()

        edges = [_edge_energy_keV(self._elem_Z[i], self._elem_shell[i])
                 for i in range(3)
                 if self._elem_Z[i] and (i == 0 or self._elem_chk[i].isChecked())]
        edges = [e for e in edges if e > 0]
        if not edges:
            self._lbl_fl.setText('Select a resonant element first.')
            return False
        q_lo, q_hi = sorted((self._fl_qmin.value(), self._fl_qmax.value()))

        n = len(self._datasets)
        be = np.full(n, np.nan)
        for k, ds in enumerate(self._datasets):
            if ds['energy'] is None:
                continue
            I = ds.get('I_raw', ds['I'])
            m = (ds['q'] >= q_lo) & (ds['q'] <= q_hi)
            if m.any():
                be[k] = float(np.mean(I[m]))

        # Base: active dataset whose energy is farthest from the nearest edge
        cand = [k for k, d in enumerate(self._datasets)
                if d.get('active', True) and not np.isnan(be[k])]
        if not cand:
            self._lbl_fl.setText(
                f'No active dataset has data in q = {q_lo:g}–{q_hi:g} Å⁻¹.')
            return False
        dist = lambda k: min(abs(self._datasets[k]['energy'] - e) for e in edges)
        base = max(cand, key=dist)
        bb = be[base]

        self._fluor = dict(base=base, bb=bb, be=be, q_range=(q_lo, q_hi))
        pen_w = pg.mkPen('#ffffff', width=2, style=Qt.PenStyle.DashLine)
        for k in range(n):
            if np.isnan(be[k]):
                continue
            if k == base:
                pen, val = pen_w, bb
            else:
                c = self._curves[k].opts['pen'].color()
                pen, val = pg.mkPen(c, width=1, style=Qt.PenStyle.DashLine), be[k]
            line = pg.InfiniteLine(angle=0, movable=False, pen=pen)
            self._pw.addItem(line)
            self._fluor_lines.append((line, val))
        self._place_fluor_lines()

        missing = int(np.isnan(be).sum())
        self._lbl_fl.setText(
            f"bb = {bb:.4g} at {self._datasets[base]['energy']:.4f} keV; "
            f"be − bb: {np.nanmin(be - bb):.3g} … {np.nanmax(be - bb):.3g}"
            + (f'  ({missing} dataset(s) skipped: no data in q range)'
               if missing else ''))
        return True

    def _fluor_subtract(self):
        """Replace I with I_raw − (be − bb) for every dataset (idempotent)."""
        if not self._datasets:
            return
        if self._fluor is None and not self._fluor_estimate():
            return
        f = self._fluor
        shift = f['be'] - f['bb']
        n_neg = 0
        self._table.blockSignals(True)
        for k, ds in enumerate(self._datasets):
            if np.isnan(shift[k]):
                continue
            ds.setdefault('I_raw', ds['I'].copy())
            ds['I'] = ds['I_raw'] - shift[k]
            ds['fluor_shift'] = float(shift[k])
            n_neg += int(np.sum(ds['I'] <= 0))
            m = ds['I'] > 0
            self._curves[k].setData(ds['q'][m], ds['I'][m])
            snr = float(np.median(ds['sigma'] / np.maximum(ds['I'], 1e-30)))
            self._table.item(k, self._COL_SNR).setText(f'{snr:.3f}')
        self._table.blockSignals(False)
        self._lbl_fl.setText(
            self._lbl_fl.text() + '\nSubtracted.'
            + (f' {n_neg} point(s) ≤ 0 are hidden on the log plot.' if n_neg else ''))
        self._emit_active()

    # ── Stuhrmann slice ───────────────────────────────────────────────────────
    def _q_grid(self) -> np.ndarray | None:
        active = [d for d in self._datasets if d.get('active', True)]
        if not active:
            return None
        return np.linspace(self._q_min.value(), self._q_max.value(),
                           int(self._n_pts.value()))

    def _update_slider_range(self):
        q = self._q_grid()
        if q is None or len(q) == 0:
            self._sl_q.setEnabled(False)
            return
        self._sl_q.setEnabled(True)
        prev_max = self._sl_q.maximum()
        self._sl_q.blockSignals(True)
        self._sl_q.setMaximum(len(q) - 1)
        if prev_max == 0:
            self._sl_q.setValue(len(q) // 10)
        self._sl_q.blockSignals(False)
        self._on_q_slider(self._sl_q.value())

    def _update_st_vline(self):
        q = self._q_grid()
        if q is None or len(q) == 0:
            return
        idx   = max(0, min(self._sl_q.value(), len(q) - 1))
        q_val = q[idx]
        if self._chk_logx.isChecked():
            self._st_vline.setPos(np.log10(q_val))
        else:
            self._st_vline.setPos(q_val)

    def _on_q_slider(self, idx: int):
        q = self._q_grid()
        if q is None or len(q) == 0:
            return
        idx   = max(0, min(idx, len(q) - 1))
        q_val = q[idx]
        self._lbl_q_val.setText(f'{q_val:.4f} Å⁻¹  [{idx + 1}/{len(q)}]')
        self._update_st_vline()
        self._update_stuhr_slice(q_val)

    # ── Resonant element ──────────────────────────────────────────────────────
    def _on_elem_chk_toggled(self, checked: bool, elem_idx: int):
        """Enable/disable the combo for secondary elements."""
        self._elem_combo[elem_idx].setEnabled(checked)
        if not checked:
            self._elem_Z[elem_idx] = 0
            self._elem_shell[elem_idx] = ''
            self._elem_fp[elem_idx] = np.array([])
            self._elem_fpp[elem_idx] = np.array([])
            self._elem_lbl_edge[elem_idx].setText('—')
            self._elem_lbl_status[elem_idx].setText('')
            self._rebuild_slice_panels(self._n_active_elems())
            self._emit_fp_fpp()
        else:
            # Trigger a compute if already has a selection
            if self._elem_Z[elem_idx] != 0:
                self._compute_fp_fpp(elem_idx)

    def _n_active_elems(self) -> int:
        """Number of enabled element rows that have fp computed."""
        return max(1, sum(
            1 for i in range(3)
            if (i == 0 or self._elem_chk[i].isChecked())
            and len(self._elem_fp[i]) > 0
        ))

    def _update_element_presets(self, elem_idx: int = 0):
        from .anomalous_tab import _edge_energy_keV, _SHELLS
        status = self._elem_lbl_status[elem_idx]
        try:
            import xraydb
        except ImportError:
            status.setText('xraydb not installed — run: pip install xraydb')
            return
        active = [d for d in self._datasets
                  if d.get('active', True) and d['energy'] is not None]
        if not active:
            status.setText('no energies found — set energies in the file table')
            return
        energies = np.array([d['energy'] for d in active])
        e_min, e_max = float(energies.min()), float(energies.max())
        candidates = []
        for Z in range(10, 93):
            try:
                sym = xraydb.atomic_symbol(Z)
            except Exception:
                continue
            for shell in _SHELLS:
                e_edge = _edge_energy_keV(Z, shell)
                if e_edge <= 0 or not (e_min - 0.3 <= e_edge <= e_max + 0.3):
                    continue
                score = min(int(np.sum(energies > e_edge)),
                            int(np.sum(energies < e_edge)))
                label = f'{sym} ({Z}) — {shell}  [{e_edge:.4f} keV]'
                candidates.append((score, Z, shell, label))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        if not candidates:
            status.setText(f'no absorption edge within {e_min:.2f}–{e_max:.2f} keV '
                           '(energies must be in keV)')
        elif status.text().startswith(('xraydb', 'no ')):
            status.setText('')
        new_map = {lbl: (Z, sh) for _, Z, sh, lbl in candidates}

        combo    = self._elem_combo[elem_idx]
        prev_Z   = self._elem_Z[elem_idx]
        prev_lbl = next((lbl for lbl, (Z, _) in new_map.items()
                         if Z == prev_Z), None)

        combo.blockSignals(True)
        combo.clear()
        combo.addItem('(select element)')
        for _, Z, shell, lbl in candidates:
            combo.addItem(lbl)
        self._preset_maps[elem_idx] = new_map

        if prev_lbl:
            idx = combo.findText(prev_lbl)
            combo.setCurrentIndex(max(1, idx))
            combo.blockSignals(False)
            self._compute_fp_fpp(elem_idx)
        elif candidates and (elem_idx == 0):
            combo.setCurrentIndex(1)
            combo.blockSignals(False)
            _, Z, shell, _ = candidates[0]
            self._elem_Z[elem_idx] = Z
            self._elem_shell[elem_idx] = shell
            self._update_edge_label(elem_idx)
            self._compute_fp_fpp(elem_idx)
        else:
            combo.blockSignals(False)

    def _update_edge_label(self, elem_idx: int):
        from .anomalous_tab import _edge_energy_keV
        e = _edge_energy_keV(self._elem_Z[elem_idx], self._elem_shell[elem_idx])
        self._elem_lbl_edge[elem_idx].setText(f'{e:.4f} keV' if e else '—')

    def _on_element_preset(self, idx: int, elem_idx: int):
        if idx == 0:
            self._elem_fp[elem_idx] = np.array([])
            self._elem_fpp[elem_idx] = np.array([])
            self._elem_Z[elem_idx] = 0
            self._elem_shell[elem_idx] = ''
            self._elem_lbl_edge[elem_idx].setText('—')
            self._elem_lbl_status[elem_idx].setText('')
            self._rebuild_slice_panels(self._n_active_elems())
            q = self._q_grid()
            if q is not None:
                self._update_stuhr_slice(
                    q[max(0, min(self._sl_q.value(), len(q)-1))])
            return
        lbl = self._elem_combo[elem_idx].currentText()
        if lbl not in self._preset_maps[elem_idx]:
            return
        Z, shell = self._preset_maps[elem_idx][lbl]
        self._elem_Z[elem_idx] = Z
        self._elem_shell[elem_idx] = shell
        self._update_edge_label(elem_idx)
        self._compute_fp_fpp(elem_idx)

    def _compute_fp_fpp(self, elem_idx: int):
        from .anomalous_tab import _fp_fpp
        from .core.decompose import condition_number
        Z = self._elem_Z[elem_idx]
        if Z == 0:
            return
        active = [d for d in self._datasets
                  if d.get('active', True) and d['energy'] is not None]
        if not active:
            return
        energies = np.array([d['energy'] for d in active])
        try:
            fp, fpp = _fp_fpp(Z, energies)
        except Exception as e:
            self._elem_lbl_status[elem_idx].setText(f'Error: {e}')
            self._elem_lbl_status[elem_idx].setStyleSheet(
                'color: #e57373; font-size: 11px;')
            return
        self._elem_fp[elem_idx]  = fp
        self._elem_fpp[elem_idx] = fpp
        kappa = condition_number(fp, fpp)
        self._elem_lbl_status[elem_idx].setText(f"f' ready  |  κ = {kappa:.1f}")
        self._elem_lbl_status[elem_idx].setStyleSheet(
            'color: #81c784; font-size: 11px;')
        self._rebuild_slice_panels(self._n_active_elems())
        self._emit_fp_fpp()
        q = self._q_grid()
        if q is not None:
            self._update_stuhr_slice(
                q[max(0, min(self._sl_q.value(), len(q)-1))])

    def _compute_all_fp_fpp(self):
        for i in range(3):
            if i == 0 or self._elem_chk[i].isChecked():
                self._compute_fp_fpp(i)

    def _emit_fp_fpp(self):
        """Collect active elements and emit fp_fpp_ready."""
        elems = self.active_elements
        if elems:
            self.fp_fpp_ready.emit(elems)

    @property
    def current_Z(self) -> int:
        """Backward compat — returns primary element Z."""
        return self._elem_Z[0]

    @property
    def active_elements(self) -> list:
        """List of dicts {'Z','label','fp','fpp'} for active elements with computed fp."""
        result = []
        for i in range(3):
            if i != 0 and not self._elem_chk[i].isChecked():
                continue
            if len(self._elem_fp[i]) == 0:
                continue
            try:
                import xraydb
                sym = xraydb.atomic_symbol(self._elem_Z[i])
            except Exception:
                sym = f'Z{self._elem_Z[i]}'
            result.append({
                'Z':     self._elem_Z[i],
                'shell': self._elem_shell[i],
                'label': f'{sym} (elem {i+1})',
                'fp':    self._elem_fp[i],
                'fpp':   self._elem_fpp[i],
            })
        return result

    @property
    def active_energies(self) -> np.ndarray:
        return np.array([d['energy'] for d in self._datasets
                         if d.get('active', True) and d['energy'] is not None])

    # ── Slice panels ──────────────────────────────────────────────────────────
    def _on_slice_logy_toggled(self, v: bool):
        for pw in self._slice_panels:
            pw.setLogMode(x=False, y=v)

    def _rebuild_slice_panels(self, n: int):
        """Rebuild n PlotWidget panels in the right vertical splitter."""
        # Remove all existing panels
        for pw in self._slice_panels:
            self._stuhr_right_split.widget(
                self._stuhr_right_split.indexOf(pw)
            ) if self._stuhr_right_split.indexOf(pw) >= 0 else None
        # Simpler: remove children from splitter
        while self._stuhr_right_split.count() > 0:
            w = self._stuhr_right_split.widget(0)
            w.setParent(None)
        self._slice_panels.clear()
        self._slice_items.clear()
        self._slice_crosshairs.clear()

        n = max(1, n)
        for i in range(n):
            pw = pg.PlotWidget(
                title=f"Element {i+1} — I vs f'₁" if n > 1
                      else "Stuhrmann: I(q, E) vs f'(E)  at selected q")
            pw.setLabel('bottom', "f'  (e)" if len(self._elem_fp[i]) > 0
                        else 'Energy (keV)')
            pw.setLabel('left', 'I(q, E)  (cm⁻¹)')
            pw.setLogMode(x=False, y=self._chk_slice_logy.isChecked())
            self._stuhr_right_split.addWidget(pw)
            self._slice_panels.append(pw)
            self._slice_items.append([])
            ch = add_crosshair(pw, label=self._coord_lbl)
            self._slice_crosshairs.append(ch)

        if n > 1:
            sz = 300 // n
            self._stuhr_right_split.setSizes([sz] * n)

    def _update_stuhr_slice(self, q_val: float):
        """Plot I(q_val, E) vs f'_i(E) in each panel."""
        if not self._slice_panels:
            self._rebuild_slice_panels(1)

        active = [d for d in self._datasets
                  if d.get('active', True) and d['energy'] is not None]
        if not active:
            return

        # Determine which elements have data for panels
        panel_elems: list[tuple[int, np.ndarray]] = []  # (elem_idx, fp_arr)
        for i in range(3):
            if i != 0 and not self._elem_chk[i].isChecked():
                continue
            fp_i = self._elem_fp[i]
            if len(fp_i) == len(active):
                panel_elems.append((i, fp_i))
            elif i == 0:
                # Fallback to energy axis for primary if fp not ready
                panel_elems.append((i, None))

        n = max(1, len(panel_elems))
        if n != len(self._slice_panels):
            self._rebuild_slice_panels(n)

        total = len(active)
        for panel_idx, (elem_i, fp_arr) in enumerate(panel_elems):
            pw = self._slice_panels[panel_idx]
            # Clear old items
            for item in self._slice_items[panel_idx]:
                pw.removeItem(item)
            self._slice_items[panel_idx].clear()

            if fp_arr is not None:
                x_vals = fp_arr
                pw.setLabel('bottom', f"f'  (e) — elem {elem_i+1}")
                title_suffix = f"f'₁(E)" if n <= 1 else f"f'_{elem_i+1}(E)"
                pw.setTitle(f"Element {elem_i+1} — I vs {title_suffix}  at q")
                # Keep only energies with meaningful anomalous contrast for this element.
                # Energies far from the edge have f'_i near its maximum (background);
                # filter them out so panels don't show the other element's energy range.
                fp_max = float(np.max(fp_arr))
                fp_range = float(np.max(fp_arr) - np.min(fp_arr))
                threshold = max(0.5, 0.05 * fp_range)
                near_edge_mask = fp_arr < (fp_max - threshold)
            else:
                x_vals = np.array([d['energy'] for d in active])
                pw.setLabel('bottom', 'Energy (keV)')
                pw.setTitle('I(q, E)  vs  energy  at selected q')
                near_edge_mask = np.ones(len(x_vals), dtype=bool)

            for k, (ds, x) in enumerate(zip(active, x_vals)):
                if not near_edge_mask[k]:
                    continue
                I_at_q   = float(np.interp(q_val, ds['q'], ds['I']))
                sig_at_q = float(np.interp(q_val, ds['q'], ds['sigma']))
                if I_at_q <= 0:
                    continue
                t   = k / max(total - 1, 1)
                col = pg.mkColor(*_energy_color(t))
                pt = pg.ScatterPlotItem(x=[x], y=[I_at_q], size=10,
                                        pen=pg.mkPen(col), brush=pg.mkBrush(col))
                pw.addItem(pt)
                self._slice_items[panel_idx].append(pt)
                if sig_at_q > 0:
                    ei = pg.ErrorBarItem(
                        x=np.array([x]), y=np.array([I_at_q]),
                        top=np.array([sig_at_q]), bottom=np.array([sig_at_q]),
                        pen=pg.mkPen(col, width=1.0))
                    pw.addItem(ei)
                    self._slice_items[panel_idx].append(ei)
            pw.setLogMode(x=False, y=self._chk_slice_logy.isChecked())

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _row_color(self, idx: int, total: int) -> tuple[int, int, int]:
        t = idx / max(total - 1, 1)
        return _energy_color(t)

    def _append_row(self, ds: dict, idx: int, total: int):
        r = self._table.rowCount()
        self._table.insertRow(r)

        # ✓ checkbox
        chk = QTableWidgetItem()
        chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                     | Qt.ItemFlag.ItemIsSelectable)
        chk.setCheckState(
            Qt.CheckState.Checked if ds.get('active', True) else Qt.CheckState.Unchecked)
        self._table.setItem(r, self._COL_USE, chk)

        # Info columns
        e_str  = f"{ds['energy']:.4f}" if ds['energy'] is not None else '?'
        qrange = f"{ds['q'].min():.4f}–{ds['q'].max():.4f}"
        snr    = float(np.median(ds['sigma'] / np.maximum(ds['I'], 1e-30)))

        for col, val in enumerate([e_str, qrange, str(len(ds['q'])), f'{snr:.3f}',
                                    Path(ds['path']).name],
                                   start=self._COL_ENERGY):
            it = QTableWidgetItem(val)
            if col != self._COL_ENERGY:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, col, it)

        if ds['energy'] is None:
            self._table.item(r, self._COL_ENERGY).setBackground(QColor('#ffe066'))

        # Plot curve
        rgb = self._row_color(idx, total)
        col_obj = pg.mkColor(*rgb)
        active = ds.get('active', True)
        mask = ds['I'] > 0
        c = self._pw.plot(
            ds['q'][mask], ds['I'][mask],
            pen=pg.mkPen(col_obj, width=1.5),
            name=e_str if active else None,
        )
        c.setVisible(active)
        self._curves.append(c)

    def _recolor_curves(self):
        """Recolor all curves after an energy edit changes sort order."""
        total = len(self._curves)
        for i, (ds, c) in enumerate(zip(self._datasets, self._curves)):
            rgb = self._row_color(i, total)
            c.setPen(pg.mkPen(pg.mkColor(*rgb), width=1.5))

    def _save_filelist(self):
        import json
        if not self._datasets:
            QMessageBox.information(self, 'Nothing to save', 'No files are loaded.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save file list', '',
            'ASAXS file list (*.json);;All files (*)')
        if not path:
            return
        if not path.endswith('.json'):
            path += '.json'
        data = {
            'asaxs_filelist_version': 1,
            'files': [
                {
                    'path': str(ds['path']),
                    'energy': ds.get('energy'),
                    'active': bool(ds.get('active', True)),
                }
                for ds in self._datasets
            ],
            'elements': [
                {
                    'Z': self._elem_Z[i],
                    'shell': self._elem_shell[i],
                    'enabled': bool(i == 0 or self._elem_chk[i].isChecked()),
                }
                for i in range(3)
            ],
            'q_min': self._q_min.value(),
            'q_max': self._q_max.value(),
            'n_pts': int(self._n_pts.value()),
        }
        try:
            Path(path).write_text(json.dumps(data, indent=2))
            self._lbl_status.setText(
                f'Saved: {Path(path).name}  ({len(self._datasets)} files)')
        except Exception as e:
            QMessageBox.warning(self, 'Save error', str(e))

    def _load_filelist(self):
        import json
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load file list', '',
            'ASAXS file list (*.json);;All files (*)')
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
        except Exception as e:
            QMessageBox.warning(self, 'Load error',
                                f'Cannot read {Path(path).name}:\n{e}')
            return
        if data.get('asaxs_filelist_version', 0) < 1:
            QMessageBox.warning(self, 'Load error', 'Unrecognised file list format.')
            return

        if self._datasets:
            reply = QMessageBox.question(
                self, 'Clear existing data?',
                'Clear the current file list before loading?',
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                self._clear()

        # Restore element Z/shell BEFORE _sort_and_emit so _update_element_presets
        # can match them against the newly available energies.
        for i, elem_info in enumerate(data.get('elements', [])):
            if i >= 3:
                break
            Z       = int(elem_info.get('Z', 0))
            shell   = str(elem_info.get('shell', ''))
            enabled = bool(elem_info.get('enabled', i == 0))
            self._elem_Z[i]     = Z
            self._elem_shell[i] = shell
            if i > 0:
                self._elem_chk[i].blockSignals(True)
                self._elem_chk[i].setChecked(enabled and Z > 0)
                self._elem_combo[i].setEnabled(enabled and Z > 0)
                self._elem_chk[i].blockSignals(False)

        # Restore q-grid settings (blockSignals to avoid early recomputation)
        for sp, key in [(self._q_min, 'q_min'), (self._q_max, 'q_max'),
                        (self._n_pts, 'n_pts')]:
            if key in data:
                sp.blockSignals(True)
                sp.setValue(data[key])
                sp.blockSignals(False)

        # Load each file
        errors = []
        for entry in data.get('files', []):
            p = entry.get('path', '')
            try:
                ds = load_file(p)
                ds['energy'] = entry.get('energy')
                ds['active'] = bool(entry.get('active', True))
                self._datasets.append(ds)
            except Exception as e:
                errors.append(f'{Path(p).name}: {e}')

        # Rebuild table, recolor, update element presets
        self._sort_and_emit()

        if errors:
            QMessageBox.warning(
                self, f'{len(errors)} file(s) failed to load',
                '\n'.join(errors[:15]) +
                (f'\n… and {len(errors)-15} more' if len(errors) > 15 else ''))
        else:
            n = len(data.get('files', []))
            self._lbl_status.setText(
                f'Loaded: {Path(path).name}  ({n} files)')

    def _sort_and_emit(self):
        # Sort by energy
        self._datasets.sort(key=lambda d: d['energy'] or 0)

        # Rebuild table and curves from scratch
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for c in self._curves:
            self._pw.removeItem(c)
        self._curves.clear()
        self._clear_fluor()

        total = len(self._datasets)
        for i, ds in enumerate(self._datasets):
            self._append_row(ds, i, total)

        self._table.blockSignals(False)

        n = total
        n_active = sum(1 for d in self._datasets if d.get('active', True))
        self._lbl_status.setText(
            f'{n} datasets loaded, {n_active} active')

        # Auto-set q range from active datasets
        active = [d for d in self._datasets if d.get('active', True)]
        if active:
            q_lo = max(d['q'].min() for d in active)
            q_hi = min(d['q'].max() for d in active)
            for sp in (self._q_min, self._q_max, self._n_pts):
                sp.blockSignals(True)
            self._q_min.setValue(q_lo)
            self._q_max.setValue(q_hi)
            for sp in (self._q_min, self._q_max, self._n_pts):
                sp.blockSignals(False)
            if not self._fluor_q_touched:
                self._fl_qmin.blockSignals(True); self._fl_qmax.blockSignals(True)
                self._fl_qmin.setValue(q_lo + 0.8 * (q_hi - q_lo))
                self._fl_qmax.setValue(q_hi)
                self._fl_qmin.blockSignals(False); self._fl_qmax.blockSignals(False)

        self._emit_active()

    def _emit_active(self):
        active = [d for d in self._datasets if d.get('active', True)]
        self.datasets_changed.emit(active)
        self._update_slider_range()
        for i in range(3):
            self._update_element_presets(i)

    # ── Public API ────────────────────────────────────────────────────────────
    def get_common_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        active = [d for d in self._datasets if d.get('active', True)]
        if len(active) < 3:
            return None
        return interpolate_to_common_q(
            active,
            q_min=self._q_min.value(),
            q_max=self._q_max.value(),
            n_points=int(self._n_pts.value()),
        )

    def energies(self) -> list[float]:
        return [d['energy'] for d in self._datasets if d.get('active', True)]
