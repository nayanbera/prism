"""
GNOM-style indirect Fourier transform with error propagation.

Tikhonov second-difference regularisation + optional NNLS (p(r) ≥ 0).
Regularisation parameter α is chosen by GCV if not supplied.

Error propagation
-----------------
For the regularised solution  p = (K^T W K + αL^T L)^{-1} K^T W I,
the propagated covariance is:

    Cov(p) = M (K^T W diag(σ²) W K) M^T
           = M (K_w^T K_w) M          [since W = diag(1/σ²), so W·diag(σ²)·W = W]

where  M = (K_w^T K_w + αL^T L)^{-1}  and  K_w = K · diag(1/σ).

For the diagonal: σ²(p_j) = [M (K_w^T K_w) M]_{jj}

We compute this efficiently as:
    B   = M @ K_w.T      (Nr × NQ)
    C   = B @ K_w        (Nr × Nr)  = M K_w^T K_w
    σ²  = diag(C @ M)    (Nr,)      = diag(M K_w^T K_w M)

Using einsum to avoid materialising the full Nr×Nr result twice.
"""

import numpy as np
from numpy.linalg import lstsq, inv
from scipy.optimize import nnls


def _sinc_kernel(q: np.ndarray, r: np.ndarray, dr: float) -> np.ndarray:
    """4π sinc(qr) dr kernel matrix (NQ × NR)."""
    qr = np.outer(q, r)
    return 4.0 * np.pi * np.where(np.abs(qr) < 1e-8, 1.0, np.sin(qr) / qr) * dr


def _smoothness(n: int) -> np.ndarray:
    """Second-difference matrix (n-2 × n)."""
    L = np.zeros((n - 2, n))
    for k in range(n - 2):
        L[k, k] = 1; L[k, k + 1] = -2; L[k, k + 2] = 1
    return L


def auto_alpha(K_w: np.ndarray, L: np.ndarray, b_w: np.ndarray,
               n_iter: int = 50) -> float:
    """
    Morozov discrepancy principle: find α such that χ²(α) = N_data.

    More robust than GCV for severely ill-posed problems.  GCV with the
    standard SVD approximation ignores L and systematically under-regularises,
    producing spiky p(r).  Morozov only needs the chi² value at each α,
    so the general-L case is handled exactly via bisection.

    Target: ||K_w·p(α) - b_w||² = N_data  (mean reduced chi² = 1)
    """
    N = len(b_w)
    target = float(N)

    def _chi2(a: float) -> float:
        K_aug = np.vstack([K_w, np.sqrt(a) * L])
        b_aug = np.concatenate([b_w, np.zeros(L.shape[0])])
        p, _ = nnls(K_aug, b_aug)
        r = K_w @ p - b_w
        return float(r @ r)

    # Bracket: at tiny α the fit is perfect (chi²<N), at huge α p=0 (chi²=||b||²>N)
    a_lo, a_hi = 1e-8, 1e14
    c_lo = _chi2(a_lo)
    c_hi = _chi2(a_hi)

    if c_lo >= target:
        return a_lo   # already over-regularised at minimum — use smallest α
    if c_hi <= target:
        return a_hi   # can't reach target — use maximum α

    for _ in range(n_iter):
        a_mid = np.sqrt(a_lo * a_hi)
        if _chi2(a_mid) < target:
            a_lo = a_mid
        else:
            a_hi = a_mid

    return np.sqrt(a_lo * a_hi)


def _propagate_errors(K_aug: np.ndarray, K_w: np.ndarray) -> np.ndarray:
    """
    Diagonal of  M K_w^T K_w M  where  M = (K_aug^T K_aug)^{-1}.
    Returns σ²(p) vector.
    """
    try:
        AtA     = K_aug.T @ K_aug          # (Nr, Nr)
        M       = inv(AtA)                  # (Nr, Nr)
        B       = M @ K_w.T                # (Nr, NQ)
        C       = B @ K_w                  # (Nr, Nr) = M K_w^T K_w
        sigma2p = np.einsum('ij,ji->i', C, M)   # diag(C @ M)
        return np.maximum(sigma2p, 0.0)
    except np.linalg.LinAlgError:
        return np.zeros(K_aug.shape[1])


def gnom_ift(
    q: np.ndarray,
    Iq: np.ndarray,
    Dmax: float,
    Nr: int = 200,
    alpha: float | None = None,
    nonneg: bool = True,
    sigma: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute p(r) and σ_p(r) from I(q) via Tikhonov-regularised IFT.

    Returns
    -------
    r  : (Nr,)
    pr : (Nr,)
    sp : (Nr,)  propagated 1σ uncertainty on p(r)
    """
    r  = np.linspace(Dmax / Nr, Dmax, Nr)
    dr = r[1] - r[0]
    K  = _sinc_kernel(q, r, dr)       # (NQ, Nr)
    L  = _smoothness(Nr)               # (Nr-2, Nr)

    if sigma is not None:
        w   = 1.0 / np.maximum(sigma, 1e-30)
        K_w = K * w[:, None]
        b_w = Iq * w
    else:
        w   = np.ones(len(q))
        K_w = K.copy()
        b_w = Iq.copy()

    if alpha is None:
        alpha = auto_alpha(K_w, L, b_w)

    K_aug = np.vstack([K_w, np.sqrt(alpha) * L])
    b_aug = np.concatenate([b_w, np.zeros(Nr - 2)])

    if nonneg:
        pr, _ = nnls(K_aug, b_aug)
    else:
        pr, *_ = lstsq(K_aug, b_aug, rcond=None)

    sigma2p = _propagate_errors(K_aug, K_w)
    sp = np.sqrt(sigma2p)

    return r, pr, sp


def _constrained_decompose_per_r(
    A: np.ndarray,          # (N_E, 3)
    P_matrix: np.ndarray,   # (N_E, Nr)
    SP_matrix: np.ndarray,  # (N_E, Nr)
    smooth_window: int = 0,
) -> tuple[np.ndarray, ...]:
    """
    Decompose p(r, E) stack at each r with physical constraints:
      p_MM(r) ≥ 0,  p_RM(r) unconstrained,  p_RR(r) ≥ 0.

    Uses scipy.optimize.lsq_linear (BVLS) per r-point.
    Optionally Savitzky-Golay smooths the result.

    Returns p_MM, p_RM, p_RR, sp_MM, sp_RM, sp_RR — each (Nr,).
    """
    from scipy.optimize import lsq_linear
    from scipy.signal import savgol_filter

    N_E, Nr = P_matrix.shape
    p_MM  = np.zeros(Nr); sp_MM = np.zeros(Nr)
    p_RM  = np.zeros(Nr); sp_RM = np.zeros(Nr)
    p_RR  = np.zeros(Nr); sp_RR = np.zeros(Nr)

    lo = np.array([0.0, -np.inf, 0.0])
    hi = np.full(3, np.inf)

    # Precompute AtWA inverse for σ estimation (done once per r, tiny 3×3)
    for j in range(Nr):
        w  = 1.0 / np.maximum(SP_matrix[:, j], 1e-30)
        Aw = A * w[:, None]
        bw = P_matrix[:, j] * w

        res = lsq_linear(Aw, bw, bounds=(lo, hi), method='bvls', tol=1e-10)
        p_MM[j], p_RM[j], p_RR[j] = res.x

        # σ from unconstrained WLS covariance (valid approximation away from bounds)
        try:
            cov = np.linalg.inv(Aw.T @ Aw)
            sp_MM[j] = np.sqrt(max(cov[0, 0], 0.0))
            sp_RM[j] = np.sqrt(max(cov[1, 1], 0.0))
            sp_RR[j] = np.sqrt(max(cov[2, 2], 0.0))
        except np.linalg.LinAlgError:
            pass

    if smooth_window > 1:
        # Savitzky-Golay: window must be odd and > polyorder
        win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        win = max(win, 5)
        poly = min(3, win - 2)
        for arr in (p_MM, p_RM, p_RR, sp_MM, sp_RM, sp_RR):
            arr[:] = savgol_filter(arr, win, poly)
        # Re-enforce non-negativity after smoothing
        np.clip(p_MM, 0, None, out=p_MM)
        np.clip(p_RR, 0, None, out=p_RR)
        np.clip(sp_MM, 0, None, out=sp_MM)
        np.clip(sp_RM, 0, None, out=sp_RM)
        np.clip(sp_RR, 0, None, out=sp_RR)

    return p_MM, p_RM, p_RR, sp_MM, sp_RM, sp_RR


def ift_then_decompose(
    q: np.ndarray,
    I_matrix: np.ndarray,    # (N_E, NQ)
    sig_matrix: np.ndarray,  # (N_E, NQ)
    fp: np.ndarray,          # (N_E,)
    fpp: np.ndarray,         # (N_E,)
    Dmax: float,
    Nr: int = 200,
    alpha: float | None = None,
    smooth_window: int = 0,  # SG window for decomposed components (0 = off)
    progress_cb=None,        # optional callable(i, N_E) for progress updates
) -> dict:
    """
    IFT each I(q, E_i) individually → p(r, E_i), then decompose the
    p(r, E) stack at each r with physical constraints:
      p_MM(r) ≥ 0,  p_RM(r) free,  p_RR(r) ≥ 0.

    This avoids the q-space form-factor-minimum artifact: because p(r) has
    no nodes at specific r values, the WLS system at each r is well-conditioned
    everywhere.

    Returns
    -------
    dict with keys:
      r           (Nr,)
      P_matrix    (N_E, Nr)  individual p(r, E_i)
      SP_matrix   (N_E, Nr)  1σ uncertainties on each p(r, E_i)
      p_MM, p_RM, p_RR       (Nr,)
      sp_MM, sp_RM, sp_RR    (Nr,)
    """
    from .decompose import build_A
    N_E = I_matrix.shape[0]
    r   = np.linspace(Dmax / Nr, Dmax, Nr)

    P_matrix  = np.zeros((N_E, Nr))
    SP_matrix = np.zeros((N_E, Nr))

    # Share a single alpha across all energies — keeps the r grids consistent
    # and is much faster (one Morozov solve instead of N_E).
    shared_alpha = alpha
    if shared_alpha is None:
        dr  = r[1] - r[0]
        K   = _sinc_kernel(q, r, dr)
        L   = _smoothness(Nr)
        si0 = np.maximum(sig_matrix[0], 1e-30)
        w0  = 1.0 / si0
        shared_alpha = auto_alpha(K * w0[:, None], L, I_matrix[0] * w0)

    for i in range(N_E):
        _, pr, sp = gnom_ift(q, I_matrix[i], Dmax, Nr,
                             alpha=shared_alpha, nonneg=True,
                             sigma=sig_matrix[i])
        P_matrix[i]  = pr
        SP_matrix[i] = sp
        if progress_cb:
            progress_cb(i + 1, N_E)

    A = build_A(fp, fpp)   # (N_E, 3)
    p_MM, p_RM, p_RR, sp_MM, sp_RM, sp_RR = _constrained_decompose_per_r(
        A, P_matrix, SP_matrix, smooth_window=smooth_window)

    return dict(
        r=r,
        P_matrix=P_matrix, SP_matrix=SP_matrix,
        p_MM=p_MM, p_RM=p_RM, p_RR=p_RR,
        sp_MM=sp_MM, sp_RM=sp_RM, sp_RR=sp_RR,
    )


def ift_all(
    q: np.ndarray,
    I_MM: np.ndarray,
    I_RM: np.ndarray,
    I_RR: np.ndarray,
    Dmax: float,
    Nr: int = 200,
    alpha_MM: float | None = None,
    alpha_RM: float | None = None,
    alpha_RR: float | None = None,
    sigma_MM: np.ndarray | None = None,
    sigma_RM: np.ndarray | None = None,
    sigma_RR: np.ndarray | None = None,
) -> dict:
    """
    IFT all three partial intensities.

    I_MM, I_RR → nonneg=True   (p ≥ 0 is physical for auto-correlation terms)
    I_RM       → nonneg=False  (cross-term can be negative)

    Returns dict: r, p_MM, p_RM, p_RR, sp_MM, sp_RM, sp_RR,
                  alpha_MM, alpha_RM, alpha_RR
    """
    r, p_MM, sp_MM = gnom_ift(q, I_MM, Dmax, Nr, alpha_MM, nonneg=True,  sigma=sigma_MM)
    _, p_RM, sp_RM = gnom_ift(q, I_RM, Dmax, Nr, alpha_RM, nonneg=False, sigma=sigma_RM)
    _, p_RR, sp_RR = gnom_ift(q, I_RR, Dmax, Nr, alpha_RR, nonneg=True,  sigma=sigma_RR)
    return dict(r=r, p_MM=p_MM, p_RM=p_RM, p_RR=p_RR,
                sp_MM=sp_MM, sp_RM=sp_RM, sp_RR=sp_RR)
