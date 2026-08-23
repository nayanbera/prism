"""
File I/O for ASAXS datasets.

Supported formats
-----------------
- 3-column txt/dat  (q, I, σI) — energy parsed from header or filename
- NXcanSAS / NeXus HDF5        — reads Q, I, Idev; energy from metadata
"""

import re
import numpy as np
from pathlib import Path


def _parse_energy_from_header(lines: list[str]) -> float | None:
    """Scan comment lines for patterns like '#Energy=10.9' or '# E = 11.5 keV'."""
    for line in lines:
        if not line.startswith('#'):
            break
        m = re.search(r'[Ee]nergy\s*=\s*([0-9.]+)', line)
        if m:
            return float(m.group(1))
    return None


def _parse_energy_from_filename(path: Path) -> float | None:
    """Try to extract energy from filename, e.g. 'sample_11500eV.dat' or '11.5keV'."""
    name = path.stem
    m = re.search(r'(\d+\.?\d*)\s*[kK][eE][vV]', name)
    if m:
        return float(m.group(1))
    m = re.search(r'(\d{4,5})\s*[eE][vV]', name)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def load_txt(path: str | Path, energy: float | None = None) -> dict:
    """
    Load a 3-column (q, I, σI) text file.

    Returns dict: {q, I, sigma, energy, path}
    Energy is taken from the `energy` argument, or parsed from header/filename.
    """
    path = Path(path)
    with open(path) as f:
        lines = f.readlines()

    if energy is None:
        energy = _parse_energy_from_header(lines) or _parse_energy_from_filename(path)

    data = np.loadtxt(path, comments='#')
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path.name}: expected ≥2 columns, got {data.shape}")

    q  = data[:, 0]
    I  = data[:, 1]
    si = data[:, 2] if data.shape[1] >= 3 else np.sqrt(np.abs(I))

    # Trim non-positive q or I
    mask = (q > 0) & (I > 0) & (si > 0)
    return dict(q=q[mask], I=I[mask], sigma=si[mask],
                energy=energy, path=str(path))


def load_nxcansas(path: str | Path, energy: float | None = None) -> dict:
    """Load Q, I, Idev from a NXcanSAS / generic NeXus HDF5 file."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 support: pip install h5py")

    path = Path(path)
    with h5py.File(path, 'r') as f:
        def _find(f, name):
            results = []
            f.visititems(lambda k, v: results.append(k)
                         if isinstance(v, h5py.Dataset) and k.endswith(name) else None)
            return results

        q_key  = next(iter(_find(f, 'Q')),  None) or next(iter(_find(f, 'q')),  None)
        I_key  = next(iter(_find(f, 'I')),  None) or next(iter(_find(f, 'intensity')), None)
        si_key = next(iter(_find(f, 'Idev')), None) or next(iter(_find(f, 'dI')), None)

        if q_key is None or I_key is None:
            raise ValueError(f"Cannot locate Q/I datasets in {path.name}")

        q  = f[q_key][()]
        I  = f[I_key][()]
        si = f[si_key][()] if si_key else np.sqrt(np.abs(I))

        if energy is None:
            for key in ('entry/instrument/monochromator/energy',
                        'entry/sample/beam/incident_energy',
                        'energy', 'Energy'):
                if key in f:
                    energy = float(f[key][()])
                    break

    mask = (q > 0) & (I > 0) & (si > 0)
    return dict(q=q[mask], I=I[mask], sigma=si[mask],
                energy=energy, path=str(path))


def load_file(path: str | Path, energy: float | None = None) -> dict:
    """Auto-detect format and load."""
    path = Path(path)
    if path.suffix.lower() in ('.h5', '.hdf5', '.nxs', '.nx'):
        return load_nxcansas(path, energy)
    return load_txt(path, energy)


def interpolate_to_common_q(
    datasets: list[dict],
    q_min: float | None = None,
    q_max: float | None = None,
    n_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate all datasets to a common q-grid.

    Returns
    -------
    q_common   : (NQ,)
    I_matrix   : (N_E, NQ)
    sig_matrix : (N_E, NQ)
    """
    data_q_min = max(d['q'].min() for d in datasets)
    data_q_max = min(d['q'].max() for d in datasets)
    q_min = max(q_min if q_min else data_q_min, data_q_min)
    q_max = min(q_max if q_max else data_q_max, data_q_max)
    n_points = n_points or min(len(d['q']) for d in datasets)
    q_common = np.linspace(q_min, q_max, n_points)

    I_mat  = np.zeros((len(datasets), n_points))
    si_mat = np.zeros_like(I_mat)
    for k, ds in enumerate(datasets):
        I_mat[k]  = np.interp(q_common, ds['q'], ds['I'])
        si_mat[k] = np.interp(q_common, ds['q'], ds['sigma'])

    return q_common, I_mat, si_mat
