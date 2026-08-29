"""
q-space ASAXS decomposition.

Direct mode  — solves A·[I_MM, I_RM, I_RR]^T = I(q, E) at each q.
Difference mode — subtracts a reference energy to cancel I_MM, solving
                  A_diff·[I_RM, I_RR]^T = ΔI(q, E) at each q.

Both use unconstrained WLS (lstsq).  Non-negativity is NOT enforced
in q-space — doing so causes a spurious dip in I_RR at form-factor
minima where I_tot → 0 for all energies simultaneously.

Error propagation: Cov(x) = (A^T W A)^{-1}  for the unconstrained WLS.

Multi-element support (N=1..3):
  partial_names_multi, build_A_multi, condition_number_multi, decompose_multi
"""

import numpy as np
from numpy.linalg import lstsq, inv


def build_A(fp: np.ndarray, fpp: np.ndarray) -> np.ndarray:
    """(N_E × 3) design matrix: columns [1, 2f', f'²+f''²]."""
    return np.column_stack([np.ones(len(fp)), 2.0 * fp, fp**2 + fpp**2])


def condition_number(fp: np.ndarray, fpp: np.ndarray) -> float:
    return float(np.linalg.cond(build_A(fp, fpp)))


def cauchy_schwarz_ratio(
    I_MM: np.ndarray, I_RM: np.ndarray, I_RR: np.ndarray
) -> np.ndarray:
    """CS(q) = I_RM² / (|I_MM|·|I_RR|).  Values > 1 violate the physical constraint."""
    denom = np.abs(I_MM) * np.abs(I_RR)
    return np.where(denom > 0, I_RM**2 / denom, 0.0)


def enforce_cauchy_schwarz(
    I_MM: np.ndarray, I_RM: np.ndarray, I_RR: np.ndarray
) -> np.ndarray:
    """Boost I_RR to I_RM²/|I_MM| where the C-S inequality is violated.

    Rearranges I_RM² ≤ I_MM·I_RR → I_RR ≥ I_RM²/|I_MM|.
    Adjusting I_RR (the weakest signal) upward is more physical than clamping
    I_RM — the WLS systematically underestimates I_RR due to its low SNR.
    Returns the corrected I_RR array.
    """
    I_RR_out = I_RR.copy()
    violated = cauchy_schwarz_ratio(I_MM, I_RM, I_RR) > 1.0
    if violated.any():
        denom = np.abs(I_MM[violated])
        I_RR_out[violated] = np.where(
            denom > 0, I_RM[violated]**2 / denom, I_RR_out[violated])
    return I_RR_out


def _wls_with_errors(A: np.ndarray, b: np.ndarray, w: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """WLS solve + propagated 1σ errors on the solution."""
    Aw = A * w[:, None]
    bw = b * w
    sol, *_ = lstsq(Aw, bw, rcond=None)
    # Cov(sol) = (A^T W A)^{-1}  where W = diag(w²)
    AtWA = Aw.T @ Aw
    try:
        cov_diag = np.diag(inv(AtWA))
        errs = np.sqrt(np.maximum(cov_diag, 0.0))
    except np.linalg.LinAlgError:
        errs = np.full(len(sol), np.nan)
    return sol, errs


# ── Direct decomposition ──────────────────────────────────────────────────────

def decompose(
    q: np.ndarray,
    I_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    fp: np.ndarray,
    fpp: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Decompose energy-resolved SAXS into partial structure factors.

    Returns
    -------
    I_MM, I_RM, I_RR, σ_MM, σ_RM, σ_RR  — each (NQ,)
    """
    A  = build_A(fp, fpp)
    NQ = len(q)
    parts = np.zeros((3, NQ))
    errs  = np.zeros((3, NQ))
    for j in range(NQ):
        sig = np.maximum(sigma_matrix[:, j], 1e-30)
        sol, e = _wls_with_errors(A, I_matrix[:, j], 1.0 / sig)
        parts[:, j] = sol
        errs[:, j]  = e
    return parts[0], parts[1], parts[2], errs[0], errs[1], errs[2]


# ── Difference decomposition ──────────────────────────────────────────────────

def build_A_diff(
    fp: np.ndarray,
    fpp: np.ndarray,
    ref_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the 2-column difference design matrix.

    Subtracting reference energy E_ref cancels I_MM:
        ΔI(q, E_i) = I(q,E_i) - I(q,E_ref)
                   = 2[f'_i - f'_ref]·I_RM  +  [(f'²+f''²)_i - (f'²+f''²)_ref]·I_RR

    Returns
    -------
    A_diff : (N_E-1, 2)   rows in same order as rows_mask
    rows_mask : bool array — True for rows kept (all except ref_idx)
    """
    mask   = np.ones(len(fp), dtype=bool)
    mask[ref_idx] = False
    fp_d   = fp[mask]  - fp[ref_idx]
    f2_d   = (fp[mask]**2 + fpp[mask]**2) - (fp[ref_idx]**2 + fpp[ref_idx]**2)
    A_diff = np.column_stack([2.0 * fp_d, f2_d])
    return A_diff, mask


def decompose_difference(
    q: np.ndarray,
    I_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    fp: np.ndarray,
    fpp: np.ndarray,
    ref_idx: int = 0,
) -> tuple[np.ndarray, ...]:
    """
    Difference ASAXS decomposition — recovers I_RM and I_RR only.

    I_MM (energy-independent) cancels on subtraction; this mode is more
    robust because it removes systematic background and reduces κ.

    Returns
    -------
    I_RM, I_RR, σ_RM, σ_RR  — each (NQ,)
    """
    A_diff, mask = build_A_diff(fp, fpp, ref_idx)
    NQ = len(q)
    parts = np.zeros((2, NQ))
    errs  = np.zeros((2, NQ))

    I_ref   = I_matrix[ref_idx]
    sig_ref = sigma_matrix[ref_idx]

    for j in range(NQ):
        dI  = I_matrix[mask, j] - I_ref[j]
        # Combined σ for subtracted data: sqrt(σ_i² + σ_ref²)
        sig = np.sqrt(sigma_matrix[mask, j]**2 + sig_ref[j]**2)
        sig = np.maximum(sig, 1e-30)
        sol, e = _wls_with_errors(A_diff, dI, 1.0 / sig)
        parts[:, j] = sol
        errs[:, j]  = e

    return parts[0], parts[1], errs[0], errs[1]


# ── Stuhrmann analysis ────────────────────────────────────────────────────────

def stuhrmann_analysis(
    q: np.ndarray,
    I_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    fp: np.ndarray,
    fpp: np.ndarray,
) -> dict:
    """
    Stuhrmann plot: I(q≈0, E) = I_MM(0) + 2f'·I_RM(0) + (f'²+f''²)·I_RR(0).

    Fits a quadratic in f':  I(0,E) = a·f'² + b·f' + c
    so:  I_RR(0) = a,  I_RM(0) = b/2,  I_MM(0) = c - a·f''²_avg

    Returns dict with keys: fp2, I0, I0_sigma, a, b, c, fit_I0,
                             I_MM_0, I_RM_0, I_RR_0, kappa
    """
    I0    = I_matrix[:, 0]
    sig0  = sigma_matrix[:, 0]
    fp2   = fp**2
    f2tot = fp**2 + fpp**2

    # Quadratic fit in f' (equivalent to linear in [1, f', f'²+f''²])
    A = build_A(fp, fpp)
    w = 1.0 / np.maximum(sig0, 1e-30)
    sol, errs = _wls_with_errors(A, I0, w)
    I_MM_0, I_RM_0, I_RR_0 = sol

    kappa = condition_number(fp, fpp)

    # Smooth fit curve over f' range.
    # Sort by fp before interpolating: fp is non-monotonic when energies cross
    # the absorption edge, and np.interp requires increasing xp.
    _sort   = np.argsort(fp)
    fp_fit  = np.linspace(fp.min(), fp.max(), 200)
    fpp_fit = np.interp(fp_fit, fp[_sort], fpp[_sort])
    I0_fit  = I_MM_0 + 2 * fp_fit * I_RM_0 + (fp_fit**2 + fpp_fit**2) * I_RR_0

    return dict(
        fp=fp, fp2=fp2, I0=I0, I0_sigma=sig0,
        fp_fit=fp_fit, I0_fit=I0_fit,
        I_MM_0=I_MM_0, I_RM_0=I_RM_0, I_RR_0=I_RR_0,
        sigma_MM_0=errs[0], sigma_RM_0=errs[1], sigma_RR_0=errs[2],
        kappa=kappa,
    )


# ── Multi-element decomposition ───────────────────────────────────────────────

def partial_names_multi(n_elem: int) -> list:
    """Ordered list of partial structure factor names for N elements.
    Order: I_MM, I_R1M,..,I_RNM, I_R1R1,..,I_RNRN, I_R1R2,..,I_R(N-1)RN
    """
    names = ['I_MM']
    for i in range(n_elem):
        names.append(f'I_R{i+1}M')
    for i in range(n_elem):
        names.append(f'I_R{i+1}R{i+1}')
    for i in range(n_elem):
        for j in range(i+1, n_elem):
            names.append(f'I_R{i+1}R{j+1}')
    return names


def build_A_multi(elements: list) -> np.ndarray:
    """Build (N_E, n_cols) design matrix for N resonant elements.
    elements: list of (fp, fpp) array pairs.
    Column order matches partial_names_multi().
    """
    N_E = len(elements[0][0])
    cols = [np.ones(N_E)]
    for fp, fpp in elements:
        cols.append(2.0 * fp)
    for fp, fpp in elements:
        cols.append(fp**2 + fpp**2)
    for i in range(len(elements)):
        for j in range(i+1, len(elements)):
            fp_i, fpp_i = elements[i]
            fp_j, fpp_j = elements[j]
            cols.append(2.0 * (fp_i*fp_j + fpp_i*fpp_j))
    return np.column_stack(cols)


def condition_number_multi(elements: list) -> float:
    return float(np.linalg.cond(build_A_multi(elements)))


def decompose_multi(
    q: np.ndarray,
    I_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    elements: list,
) -> dict:
    """Multi-element direct decomposition.
    elements: list of (fp, fpp) pairs.
    Returns dict with keys:
      'names'    : list of partial names (length n_cols)
      'partials' : ndarray (n_cols, NQ) — best-fit values
      'errors'   : ndarray (n_cols, NQ) — 1σ propagated errors
    """
    A     = build_A_multi(elements)
    names = partial_names_multi(len(elements))
    NQ    = len(q)
    n_cols = A.shape[1]
    parts = np.zeros((n_cols, NQ))
    errs  = np.zeros((n_cols, NQ))
    for j in range(NQ):
        sig = np.maximum(sigma_matrix[:, j], 1e-30)
        sol, e = _wls_with_errors(A, I_matrix[:, j], 1.0 / sig)
        parts[:, j] = sol
        errs[:, j]  = e
    return {'names': names, 'partials': parts, 'errors': errs}
