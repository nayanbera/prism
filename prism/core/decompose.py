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


def _ols_with_errors(A: np.ndarray, b: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """OLS solve + residual-based 1σ errors.

    Cov(x) = σ² (AᵀA)⁻¹  where σ² = ‖Ax - b‖² / (N - k)
    and k = number of parameters.  This is the standard unbiased OLS
    variance estimate — no measurement σ is used.
    """
    sol, residuals, rank, _ = lstsq(A, b, rcond=None)
    N, k = A.shape
    dof = N - k
    if dof > 0 and len(residuals) > 0:
        sigma2 = float(residuals[0]) / dof
    else:
        # lstsq returns empty residuals when rank < k; fall back to direct calc
        sigma2 = float(np.sum((A @ sol - b) ** 2)) / max(dof, 1)
    AtA = A.T @ A
    try:
        cov_diag = np.diag(inv(AtA)) * sigma2
        errs = np.sqrt(np.maximum(cov_diag, 0.0))
    except np.linalg.LinAlgError:
        errs = np.full(len(sol), np.nan)
    return sol, errs


def _solve(A: np.ndarray, b: np.ndarray, w: np.ndarray, method: str
           ) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to WLS or OLS solver."""
    if method == 'OLS':
        return _ols_with_errors(A, b)
    return _wls_with_errors(A, b, w)


# ── Direct decomposition ──────────────────────────────────────────────────────

def decompose(
    q: np.ndarray,
    I_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    fp: np.ndarray,
    fpp: np.ndarray,
    method: str = 'WLS',
) -> tuple[np.ndarray, ...]:
    """
    Decompose energy-resolved SAXS into partial structure factors.

    method : 'WLS' (default) or 'OLS'

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
        sol, e = _solve(A, I_matrix[:, j], 1.0 / sig, method)
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
    method: str = 'WLS',
) -> tuple[np.ndarray, ...]:
    """
    Difference ASAXS decomposition — recovers I_RM and I_RR only.

    I_MM (energy-independent) cancels on subtraction; this mode is more
    robust because it removes systematic background and reduces κ.

    method : 'WLS' (default) or 'OLS'

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
        sol, e = _solve(A_diff, dI, 1.0 / sig, method)
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

def partial_names_multi(n_elem: int, include_cross: bool = False) -> list:
    """Ordered list of partial structure factor names for N elements.
    Order: I_MM, I_R1M,..,I_RNM, I_R1R1,..,I_RNRN[, I_R1R2,..,I_R(N-1)RN]

    include_cross: include cross-terms I_RiRj (i≠j).  Default False because
    for the typical ASAXS experiment where each element's edge is scanned
    separately, f'_j is approximately constant at element-i's edge energies,
    making the I_RiRj column proportional to the I_RiM column → near-singular
    design matrix and unreliable estimates.  Set True only when energies are
    chosen so that both f'_i and f'_j vary simultaneously.
    """
    names = ['I_MM']
    for i in range(n_elem):
        names.append(f'I_R{i+1}M')
    for i in range(n_elem):
        names.append(f'I_R{i+1}R{i+1}')
    if include_cross:
        for i in range(n_elem):
            for j in range(i+1, n_elem):
                names.append(f'I_R{i+1}R{j+1}')
    return names


def build_A_multi(elements: list, include_cross: bool = False) -> np.ndarray:
    """Build (N_E, n_cols) design matrix for N resonant elements.
    elements: list of (fp, fpp) array pairs.
    Column order matches partial_names_multi(include_cross=include_cross).
    """
    N_E = len(elements[0][0])
    cols = [np.ones(N_E)]
    for fp, fpp in elements:
        cols.append(2.0 * fp)
    for fp, fpp in elements:
        cols.append(fp**2 + fpp**2)
    if include_cross:
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                fp_i, fpp_i = elements[i]
                fp_j, fpp_j = elements[j]
                cols.append(2.0 * (fp_i*fp_j + fpp_i*fpp_j))
    return np.column_stack(cols)


def condition_number_multi(elements: list, include_cross: bool = False) -> float:
    return float(np.linalg.cond(build_A_multi(elements, include_cross=include_cross)))


def decompose_multi_per_edge(
    q: np.ndarray,
    I_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    elements: list,
    edge_groups: np.ndarray,
    beta: np.ndarray | None = None,
    include_cross: bool = True,
    method: str = 'WLS',
) -> dict:
    """Global WLS with per-edge offset columns for N-element ASAXS.

    Builds an extended design matrix:

        A_ext = [1_{g0}, 1_{g1}, ...,               ← per-edge indicator offsets
                 2f'_1, 2f'_2, ...,                  ← I_RiM physical columns
                 f'_1²+f''_1², f'_2²+f''_2², ...,   ← I_RiRi physical columns
                 2(f'_1·f'_2+f''_1·f''_2), ...]      ← I_RiRj cross-terms (if include_cross)

    The indicator columns absorb cross-edge absolute-scale mismatch without
    restricting the physical parameters to a single edge's data.  All physical
    parameters are fitted jointly across every edge.

    include_cross=True (default): include cross-terms I_RiRj (i<j).  These are
    NOT collinear in this matrix because the cross-term column varies with f'_i
    at edge-i energies and with f'_j at edge-j energies — distinct patterns that
    the per-edge indicator columns cannot replicate.  Omitting cross-terms when
    they are physically present biases I_RiRi and I_RjRj (the unmodelled signal
    is absorbed into the diagonal partials).

    I_MM is estimated as the average of the per-edge intercepts after
    subtracting each edge's mean contribution from the other elements.
    """
    N_elem = len(elements)
    NQ     = len(q)
    NE     = I_matrix.shape[0]
    names  = partial_names_multi(N_elem, include_cross=include_cross)
    n_cols = len(names)
    parts  = np.zeros((n_cols, NQ))
    errs   = np.zeros((n_cols, NQ))

    if beta is not None and len(beta) > 1:
        scale   = beta[edge_groups]
        I_use   = I_matrix    / scale[:, None]
        sig_use = sigma_matrix / scale[:, None]
    else:
        I_use   = I_matrix
        sig_use = sigma_matrix

    RM_idx  = {i: names.index(f'I_R{i+1}M')      for i in range(N_elem)}
    RR_idx  = {i: names.index(f'I_R{i+1}R{i+1}') for i in range(N_elem)}
    MM_idx  = names.index('I_MM')
    # Cross-term indices (only if include_cross)
    cross_pairs = [(i, j) for i in range(N_elem) for j in range(i+1, N_elem)] \
                  if include_cross else []
    RC_idx  = {(i, j): names.index(f'I_R{i+1}R{j+1}') for i, j in cross_pairs}

    unique_groups = np.unique(edge_groups)
    N_groups      = len(unique_groups)

    if N_groups <= 1:
        # Single group: fall back to standard 3-param fit (no cross-term for 1 element)
        fp0, fpp0 = elements[0]
        A = build_A(fp0, fpp0)
        for j in range(NQ):
            sig = np.maximum(sig_use[:, j], 1e-30)
            sol, e = _solve(A, I_use[:, j], 1.0 / sig, method)
            parts[MM_idx,    j] = sol[0]; errs[MM_idx,    j] = e[0]
            parts[RM_idx[0], j] = sol[1]; errs[RM_idx[0], j] = e[1]
            parts[RR_idx[0], j] = sol[2]; errs[RR_idx[0], j] = e[2]
        C_local = parts[[MM_idx], :]
        return {'names': names, 'partials': parts, 'errors': errs, 'C_local': C_local}

    # Extended design matrix:
    #   cols 0 .. N_groups-1              : indicator 1_{g_k}
    #   cols N_groups .. N_groups+N-1     : 2·f'_i
    #   cols N_groups+N .. N_groups+2N-1  : f'_i²+f''_i²
    #   cols N_groups+2N ..               : cross-terms 2(f'_i·f'_j+f''_i·f''_j)
    n_ext = N_groups + 2 * N_elem + len(cross_pairs)
    A_ext = np.zeros((NE, n_ext))
    for k, g in enumerate(unique_groups):
        A_ext[edge_groups == g, k] = 1.0
    for i, (fp_i, fpp_i) in enumerate(elements):
        A_ext[:, N_groups + i]          = 2.0 * fp_i
        A_ext[:, N_groups + N_elem + i] = fp_i**2 + fpp_i**2
    for c, (i, j) in enumerate(cross_pairs):
        fp_i, fpp_i = elements[i]
        fp_j, fpp_j = elements[j]
        A_ext[:, N_groups + 2*N_elem + c] = 2.0 * (fp_i*fp_j + fpp_i*fpp_j)

    C_local    = np.full((N_elem, NQ), np.nan)
    # Physical part of WLS solution for each q (used to recover I_MM)
    phys_sols  = np.zeros((n_ext - N_groups, NQ))

    for qi in range(NQ):
        sig = np.maximum(sig_use[:, qi], 1e-30)
        sol, e = _solve(A_ext, I_use[:, qi], 1.0 / sig, method)
        phys_sols[:, qi] = sol[N_groups:]
        for k, g in enumerate(unique_groups):
            gi = int(g)
            if 0 <= gi < N_elem:
                C_local[gi, qi] = sol[k]
        for i in range(N_elem):
            parts[RM_idx[i], qi]  = sol[N_groups + i]
            errs[RM_idx[i],  qi]  = e[N_groups + i]
            parts[RR_idx[i], qi]  = sol[N_groups + N_elem + i]
            errs[RR_idx[i],  qi]  = e[N_groups + N_elem + i]
        for c, (i, j) in enumerate(cross_pairs):
            parts[RC_idx[(i, j)], qi] = sol[N_groups + 2*N_elem + c]
            errs[RC_idx[(i, j)],  qi] = e[N_groups + 2*N_elem + c]

    # I_MM: weighted mean of (data − physical column contributions) over ALL energies.
    # Using the per-edge indicator mean would absorb cross-term means into C_local[i],
    # requiring corrections that introduce approximation error (~0.6% bias at low q).
    # The direct weighted-mean approach is equivalent to M4 (single constant) for I_MM
    # while keeping per-edge offsets for the physical parameters.
    A_phys = A_ext[:, N_groups:]  # physical columns only
    for qi in range(NQ):
        sig     = np.maximum(sig_use[:, qi], 1e-30)
        w2      = (1.0 / sig) ** 2
        residual = I_use[:, qi] - A_phys @ phys_sols[:, qi]
        total_w2 = float(np.sum(w2))
        parts[MM_idx, qi] = float(np.sum(w2 * residual) / total_w2)
        errs[MM_idx, qi]  = float(1.0 / np.sqrt(total_w2))

    return {'names': names, 'partials': parts, 'errors': errs, 'C_local': C_local}


def decompose_multi(
    q: np.ndarray,
    I_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    elements: list,
    beta: np.ndarray | None = None,
    edge_groups: np.ndarray | None = None,
    include_cross: bool = False,
    method: str = 'WLS',
) -> dict:
    """Multi-element direct decomposition.
    elements: list of (fp, fpp) pairs.
    beta: optional (N_elem,) normalization vector — element 0 is reference (1.0),
          each element-i dataset is divided by beta[i] before solving.
    edge_groups: optional (N_E,) int array assigning each energy to an element index.
    include_cross: include cross-terms I_RiRj in the model (see partial_names_multi).
    method: 'WLS' (default) or 'OLS'
    Returns dict with keys:
      'names'    : list of partial names (length n_cols)
      'partials' : ndarray (n_cols, NQ) — best-fit values
      'errors'   : ndarray (n_cols, NQ) — 1σ propagated errors
    """
    A     = build_A_multi(elements, include_cross=include_cross)
    names = partial_names_multi(len(elements), include_cross=include_cross)
    NQ    = len(q)
    n_cols = A.shape[1]
    parts = np.zeros((n_cols, NQ))
    errs  = np.zeros((n_cols, NQ))

    I_use   = I_matrix
    sig_use = sigma_matrix
    if beta is not None and edge_groups is not None and len(beta) > 1:
        scale   = beta[edge_groups]           # (N_E,)
        I_use   = I_matrix    / scale[:, None]
        sig_use = sigma_matrix / scale[:, None]

    for j in range(NQ):
        sig = np.maximum(sig_use[:, j], 1e-30)
        sol, e = _solve(A, I_use[:, j], 1.0 / sig, method)
        parts[:, j] = sol
        errs[:, j]  = e
    return {'names': names, 'partials': parts, 'errors': errs}


# ── Cross-edge normalization ───────────────────────────────────────────────────

def compute_edge_groups(elements: list) -> np.ndarray:
    """Assign each energy to the element with the highest f'' at that energy.

    At an absorption edge f'' peaks sharply, so the element with the largest
    f'' at energy E_i is the one whose edge dataset that measurement belongs to.

    elements: list of (fp, fpp) array pairs (same format as build_A_multi).
    Returns int array of shape (N_E,) with values in {0, ..., N_elem-1}.

    NOTE: this can fail when tabulated (Cromer-Mann) f'' values are used,
    because the white-line enhancement is absent and one element's f'' may
    dominate all energies.  Prefer compute_edge_groups_by_energy() when
    element Z/shell and measurement energies are available.
    """
    if len(elements) <= 1:
        return np.zeros(len(elements[0][1]), dtype=int)
    fpp_stack = np.column_stack([fpp for _, fpp in elements])  # (N_E, N_elem)
    return np.argmax(fpp_stack, axis=1).astype(int)


def compute_edge_groups_by_energy(
    elem_meta: list,
    energies_keV: np.ndarray,
) -> np.ndarray:
    """Assign each measurement energy to the nearest element edge.

    This is more reliable than compute_edge_groups() when using tabulated
    (Cromer-Mann) f'' values that lack the true white-line enhancement.

    elem_meta : list of dicts with keys 'Z' (int) and 'shell' (str, e.g. 'K', 'L3').
    energies_keV : (N_E,) array of measurement energies in keV.
    Returns int array of shape (N_E,) with values in {0, ..., N_elem-1}.
    """
    try:
        import xraydb
    except ImportError:
        # Fall back to equal assignment
        N_E = len(energies_keV)
        return np.zeros(N_E, dtype=int)

    edge_eV = []
    for m in elem_meta:
        Z = m.get('Z', 0)
        shell = m.get('shell', '')
        try:
            e = float(xraydb.xray_edge(Z, shell).energy)   # eV
        except Exception:
            e = float('inf')
        edge_eV.append(e)

    energies_eV = np.asarray(energies_keV) * 1000.0
    # distance of each energy to each element's edge (absolute difference in eV)
    dists = np.column_stack([np.abs(energies_eV - e) for e in edge_eV])
    return np.argmin(dists, axis=1).astype(int)


def estimate_normalization_lowq(
    q: np.ndarray,
    I_matrix: np.ndarray,
    edge_groups: np.ndarray,
    q_frac: float = 0.05,
) -> np.ndarray:
    """Estimate relative edge-group normalizations from the low-q intensity ratio.

    At low q, I(q,E) ≈ I_MM(q) for all energies because the anomalous
    contributions 2f'·I_RM and (f'²+f''²)·I_RR are small compared to I_MM.
    The ratio of median intensities across edge groups therefore estimates the
    relative normalization: β_i = <I(q_low, E∈group_i)> / <I(q_low, E∈group_0)>.

    Assumptions:
      - Data are in absolute units with absorption/transmission corrections applied.
      - Anomalous contrast is small (I_RM ≲ 5% of I_MM at low q).
        For large anomalous contrast, the estimate will be biased.
      - The lowest `q_frac` fraction of the q-range is in the Guinier/near-Guinier
        regime where I(q) is nearly energy-independent.

    q_frac : fraction of q range to use (default 5%, i.e. bottom 5% of q values).
    Returns beta array of shape (N_elem,) with beta[0] = 1.0 (group 0 is reference).
    beta[i] > 1 means group i has higher absolute intensity than the reference.
    Divide group i's data by beta[i] to bring it to the reference scale.
    """
    N_elem = int(edge_groups.max()) + 1 if len(edge_groups) > 0 else 1
    if N_elem <= 1:
        return np.ones(1)

    # Check that every group has at least one energy point
    present = set(int(g) for g in edge_groups)
    if len(present) < N_elem:
        missing = [i for i in range(N_elem) if i not in present]
        raise ValueError(
            f'Edge groups {missing} have no energy points assigned')

    n_lq = max(3, int(len(q) * q_frac))
    sort_idx = np.argsort(q)
    I_lq = I_matrix[:, sort_idx[:n_lq]]   # (N_E, n_lq)

    group_medians = np.ones(N_elem)
    for g in range(N_elem):
        mask = edge_groups == g
        if mask.any():
            val = float(np.median(I_lq[mask, :]))
            group_medians[g] = val if np.isfinite(val) and val > 0 else 1.0

    beta = group_medians / group_medians[0]
    return np.where(np.isfinite(beta) & (beta > 0), beta, 1.0)
