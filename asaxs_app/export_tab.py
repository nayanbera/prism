"""Export tab — save partial I(q) and p(r) to txt or NXcanSAS."""

import numpy as np
from pathlib import Path
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QFileDialog, QCheckBox, QFormLayout, QMessageBox,
)
from PyQt6.QtCore import pyqtSignal


class ExportTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._decomp: dict | None = None
        self._ift:    dict | None = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        lay = QVBoxLayout(self)

        # Sample name
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel('Sample name:'))
        self._edit_name = QLineEdit('sample')
        self._edit_name.setMaximumWidth(300)
        hdr.addWidget(self._edit_name)
        hdr.addStretch()
        lay.addLayout(hdr)

        # ── Partial I(q) export ───────────────────────────────────────────────
        grp_iq = QGroupBox('Partial I(q)  [q, I_MM, σ_MM, I_RM, σ_RM, I_RR, σ_RR]')
        glay_iq = QFormLayout(grp_iq)

        self._chk_iq_txt = QCheckBox('txt (space-separated)')
        self._chk_iq_txt.setChecked(True)
        glay_iq.addRow(self._chk_iq_txt)

        btn_iq = QPushButton('Save partial I(q)…')
        btn_iq.clicked.connect(self._save_iq)
        self._lbl_iq = QLabel('Run decomposition first')
        glay_iq.addRow(btn_iq, self._lbl_iq)
        lay.addWidget(grp_iq)

        # ── p(r) export ───────────────────────────────────────────────────────
        grp_pr = QGroupBox('Partial p(r)  [r, p_MM, σ_MM, p_RM, σ_RM, p_RR, σ_RR]')
        glay_pr = QFormLayout(grp_pr)

        self._chk_pr_txt = QCheckBox('txt (space-separated)')
        self._chk_pr_txt.setChecked(True)
        self._chk_pr_hdf = QCheckBox('NXcanSAS HDF5  (requires h5py)')
        glay_pr.addRow(self._chk_pr_txt)
        glay_pr.addRow(self._chk_pr_hdf)

        btn_pr = QPushButton('Save partial p(r)…')
        btn_pr.clicked.connect(self._save_pr)
        self._lbl_pr = QLabel('Run IFT first')
        glay_pr.addRow(btn_pr, self._lbl_pr)
        lay.addWidget(grp_pr)

        lay.addStretch()

    # ── Save helpers ──────────────────────────────────────────────────────────
    def _header(self, content: str) -> str:
        return (f'# ASAXS partial {content}\n'
                f'# Sample: {self._edit_name.text()}\n'
                f'# Created: {datetime.now().isoformat(timespec="seconds")}\n')

    def _save_iq(self):
        if self._decomp is None:
            QMessageBox.warning(self, 'No data', 'Run decomposition first.'); return
        d = self._decomp
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save partial I(q)', f'{self._edit_name.text()}_partials_Iq.txt',
            'Text files (*.txt *.dat);;All files (*)')
        if not path:
            return
        cols = np.column_stack([
            d['q'],
            d['I_MM'], d.get('s_MM', np.zeros_like(d['q'])),
            d['I_RM'], d.get('s_RM', np.zeros_like(d['q'])),
            d['I_RR'], d.get('s_RR', np.zeros_like(d['q'])),
        ])
        hdr = (self._header('I(q)') +
               '# Columns: q[Å⁻¹]  I_MM  σ_MM  I_RM  σ_RM  I_RR  σ_RR  [cm⁻¹]\n')
        np.savetxt(path, cols, header=hdr, comments='')
        self._lbl_iq.setText(f'Saved → {Path(path).name}')

    def _save_pr(self):
        if self._ift is None:
            QMessageBox.warning(self, 'No data', 'Run IFT first.'); return
        d = self._ift
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save partial p(r)', f'{self._edit_name.text()}_partials_pr.txt',
            'Text files (*.txt *.dat);;All files (*)')
        if not path:
            return

        if self._chk_pr_txt.isChecked():
            cols = np.column_stack([
                d['r'],
                d['p_MM'], d.get('sp_MM', np.zeros_like(d['r'])),
                d['p_RM'], d.get('sp_RM', np.zeros_like(d['r'])),
                d['p_RR'], d.get('sp_RR', np.zeros_like(d['r'])),
            ])
            hdr = (self._header('p(r)') +
                   '# Columns: r[Å]  p_MM  σ_MM  p_RM  σ_RM  p_RR  σ_RR  [cm⁻¹ Å⁻¹]\n')
            np.savetxt(path, cols, header=hdr, comments='')

        if self._chk_pr_hdf.isChecked():
            hdf_path = Path(path).with_suffix('.nxs')
            self._save_nxcansas(hdf_path, d)

        self._lbl_pr.setText(f'Saved → {Path(path).name}')

    def _save_nxcansas(self, path: Path, d: dict):
        try:
            import h5py
        except ImportError:
            QMessageBox.warning(self, 'h5py missing',
                                'Install h5py for HDF5 export: pip install h5py'); return
        with h5py.File(path, 'w') as f:
            f.attrs['NX_class'] = 'NXroot'
            entry = f.create_group('entry')
            entry.attrs['NX_class'] = 'NXentry'
            sasdata = entry.create_group('sasdata')
            sasdata.attrs['NX_class'] = 'SASdata'
            sasdata.attrs['canSAS_class'] = 'SASdata'
            for name in ('p_MM', 'p_RM', 'p_RR'):
                grp = sasdata.create_group(name)
                grp.create_dataset('r', data=d['r']); grp['r'].attrs['units'] = 'Angstrom'
                grp.create_dataset('p', data=d[name])
                sp_key = 's' + name[1:]   # sp_MM, sp_RM, sp_RR
                if sp_key in d:
                    grp.create_dataset('sigma_p', data=d[sp_key])
            entry.create_dataset('sample_name',
                                  data=self._edit_name.text())

    # ── Public API ────────────────────────────────────────────────────────────
    def set_decomposition(self, result: dict):
        self._decomp = result
        self._lbl_iq.setText('Ready to save')

    def set_ift(self, result: dict):
        self._ift = result
        self._lbl_pr.setText('Ready to save')
