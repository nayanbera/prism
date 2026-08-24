"""Main window — wires all tabs together."""

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QStatusBar
import numpy as np

from .data_tab import DataTab
from .anomalous_tab import AnomalousTab
from .decomposition_tab import DecompositionTab
from .rspace_tab import RSpaceTab
from .ift_tab import IFTTab
from .export_tab import ExportTab
from .sim_data_tab import SimDataTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ASAXS Decomposer')
        self.resize(1400, 850)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self._data_tab    = DataTab()
        self._anom_tab    = AnomalousTab()
        self._decomp_tab  = DecompositionTab()
        self._rspace_tab  = RSpaceTab()
        self._ift_tab     = IFTTab()
        self._export_tab  = ExportTab()
        self._sim_tab     = SimDataTab()

        tabs.addTab(self._data_tab,   '1 · Data')
        tabs.addTab(self._anom_tab,   '2 · Anomalous factors')
        tabs.addTab(self._decomp_tab, '3 · q-space decomp')
        tabs.addTab(self._rspace_tab, '4 · r-space decomp')
        tabs.addTab(self._ift_tab,    '5 · IFT (q-first)')
        tabs.addTab(self._export_tab, '6 · Export')
        tabs.addTab(self._sim_tab,    '7 · Sim data')

        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # ── Signal wiring ─────────────────────────────────────────────────────
        self._data_tab.datasets_changed.connect(self._on_datasets_changed)
        self._anom_tab.fp_fpp_ready.connect(self._on_fp_fpp)
        self._decomp_tab.decomposition_done.connect(self._on_decomposition_done)
        self._decomp_tab.reference_loaded.connect(self._rspace_tab.set_reference_partials)
        self._ift_tab.ift_done.connect(self._on_ift_done)
        self._rspace_tab.rspace_done.connect(self._on_rspace_done)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_datasets_changed(self, datasets: list):
        energies = [d['energy'] for d in datasets]
        self._anom_tab.set_energies(energies)

        if len(datasets) >= 3:
            result = self._data_tab.get_common_grid()
            if result is not None:
                q, I_mat, sig_mat = result
                self._decomp_tab.set_data(q, I_mat, sig_mat, energies=energies)
                self._rspace_tab.set_data(q, I_mat, sig_mat, energies=energies)
                self._status.showMessage(
                    f'{len(datasets)} datasets — q: {q[0]:.4f}–{q[-1]:.4f} Å⁻¹ '
                    f'({len(q)} pts)', 5000)

    def _on_fp_fpp(self, fp: np.ndarray, fpp: np.ndarray):
        self._decomp_tab.set_fp_fpp(fp, fpp)
        self._rspace_tab.set_fp_fpp(fp, fpp)
        self._status.showMessage("f'/f'' ready — run decomposition in tab 3 or 4", 4000)

    def _on_decomposition_done(self, q, I_MM, I_RM, I_RR, s_MM, s_RM, s_RR):
        self._ift_tab.set_partials(q, I_MM, I_RM, I_RR, s_MM, s_RM, s_RR)
        result = self._decomp_tab.get_result()
        if result:
            self._export_tab.set_decomposition(result)
        self._status.showMessage(
            'Decomposition done — go to tab 4 to run IFT', 5000)

    def _on_ift_done(self, result: dict):
        self._export_tab.set_ift(result)
        self._status.showMessage('IFT complete — go to tab 6 to export', 4000)

    def _on_rspace_done(self, result: dict):
        self._export_tab.set_ift(result)
        self._status.showMessage('r-space decomposition done — go to tab 6 to export', 5000)
