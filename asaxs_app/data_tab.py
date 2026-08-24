"""Data loading tab — add I(q,E) files, assign energies, preview plots."""

from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QAbstractItemView,
    QLabel, QDoubleSpinBox, QMessageBox, QSplitter, QCheckBox,
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
    datasets_changed = pyqtSignal(list)  # emits active (checked) datasets sorted by E

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
        self._datasets: list[dict] = []   # all loaded datasets
        self._curves:   list       = []   # pg.PlotDataItem per dataset (same order)
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        lay = QVBoxLayout(self)

        # Toolbar
        btn_row = QHBoxLayout()
        self._btn_add    = QPushButton('Add files…')
        self._btn_remove = QPushButton('Remove selected')
        self._btn_clear  = QPushButton('Clear all')
        self._lbl_status = QLabel()
        for w in (self._btn_add, self._btn_remove, self._btn_clear, self._lbl_status):
            btn_row.addWidget(w)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # Splitter: table on left, plot on right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel: table + q-range controls ──────────────────────────────
        left = QWidget()
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
        hdr.setSectionResizeMode(self._COL_FILE,   QHeaderView.ResizeMode.Stretch)
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

        splitter.addWidget(left)

        # ── Right panel: plot ─────────────────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)

        plot_ctrl = QHBoxLayout()
        self._chk_logx = QCheckBox('Log q')
        self._chk_logy = QCheckBox('Log I')
        self._chk_logy.setChecked(True)
        self._chk_logx.toggled.connect(self._update_log)
        self._chk_logy.toggled.connect(self._update_log)
        plot_ctrl.addWidget(self._chk_logx)
        plot_ctrl.addWidget(self._chk_logy)
        plot_ctrl.addStretch()
        right_lay.addLayout(plot_ctrl)

        self._pw = pg.PlotWidget(title='I(q, E) — all loaded datasets')
        self._pw.setLabel('bottom', 'q (Å⁻¹)')
        self._pw.setLabel('left', 'I(q) (cm⁻¹)')
        self._pw.setLogMode(x=False, y=True)
        right_lay.addWidget(self._pw)

        splitter.addWidget(right)
        splitter.setSizes([420, 580])

        lay.addWidget(splitter, 1)

        self._ch = [add_crosshair(self._pw)]

        # ── Signal connections ────────────────────────────────────────────────
        self._btn_add.clicked.connect(self._add_files)
        self._btn_remove.clicked.connect(self._remove_selected)
        self._btn_clear.clicked.connect(self._clear)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.currentItemChanged.connect(
            lambda cur, _: self._on_row_changed(cur.row() if cur else -1))

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
        for r in rows:
            self._table.removeRow(r)
            curve = self._curves.pop(r)
            self._pw.removeItem(curve)
            self._datasets.pop(r)
        self._sort_and_emit()

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

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _row_color(self, idx: int, total: int) -> tuple[int, int, int]:
        t = idx / max(total - 1, 1)
        return _energy_color(t)

    def _append_row(self, ds: dict, idx: int, total: int):
        r = self._table.rowCount()
        self._table.insertRow(r)

        # ✓ checkbox
        chk = QTableWidgetItem()
        chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
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

    def _sort_and_emit(self):
        # Sort by energy
        self._datasets.sort(key=lambda d: d['energy'] or 0)

        # Rebuild table and curves from scratch
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for c in self._curves:
            self._pw.removeItem(c)
        self._curves.clear()

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

        self._emit_active()

    def _emit_active(self):
        active = [d for d in self._datasets if d.get('active', True)]
        self.datasets_changed.emit(active)

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
