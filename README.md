# PRISM

**Partial scattering Resolution via Intensity Spectral Modulation**

A PyQt6 desktop application for analysing anomalous small-angle X-ray scattering (ASAXS) data. PRISM decomposes a stack of I(q, E) curves — measured at multiple energies near an absorption edge — into partial structure factors, then Fourier-transforms them to partial pair-distance distribution functions (PDDFs).

---

## Features

| Tab | Function |
|-----|----------|
| **1 · Data** | Load multi-energy `.dat` files, interpolate to a common q grid, compute f′(E) / f″(E) via xraydb |
| **2 · Anomalous factors** | Inspect and edit the anomalous dispersion correction per element |
| **3 · q-space decomp** | Least-squares decomposition of I(q, E) → partial intensities I_AA, I_AB, I_BB |
| **4 · IFT** | Morozov-regularised indirect Fourier transform → p(r) PDDFs; chi² tuning, p(D_max)=0 boundary condition, optional background fit |
| **5 · Export** | Save all results (partials, PDDFs, metadata) to file |
| **6 · Sim/Fit** | Generate synthetic ASAXS datasets from a multi-shell sphere model; supports N=1–3 elements, lognormal polydispersity, Poissonian noise |

---

## Installation

```bash
git clone https://github.com/mrinalkb/prism.git
cd prism
pip install -e .
```

**Requirements:** Python ≥ 3.11, PyQt6, pyqtgraph, numpy, scipy, xraydb

Optional HDF5 export: `pip install -e ".[hdf]"`

---

## Usage

```bash
prism
```

Or from the repo root:

```bash
python -m prism
```

---

## Test data

`test_data/` contains synthetic Au-SiO₂ core-shell nanoparticle datasets at 20 energies near the Au L₃ edge (11.919 keV), in both monodisperse and polydisperse (lognormal σ = 20%) variants. Ground-truth partial structure factors are provided for validation.

---

## Physics background

ASAXS exploits the energy dependence of the atomic scattering factor f(E) = f₀ + f′(E) + if″(E) near an absorption edge. Measuring I(q) at N energies with different f′ values yields a linear system whose solution gives the N(N+1)/2 partial structure factors. PRISM solves this system by weighted least squares and transforms the partials to real-space PDDFs via a regularised indirect Fourier transform.

---

## License

MIT
