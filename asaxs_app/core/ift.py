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


def _alpha_smooth_fallback(K_w: np.ndarray, L: np.ndarray) -> float:
    """
    Fallback alpha when Morozov's lower bracket fails (chi2 floor > N).

    Returns trace(K_w^T K_w) / trace(L^T L).  This is large when the data
    weights (K_w = K/σ) are large, which forces the NNLS solution toward the
    smoothest non-negative function (L@p ≈ 0).  chi2_factor then scales from
    this base to give the user a "more smoothing" knob.
    """
    KtK = K_w.T @ K_w
    LtL = L.T @ L
    denom = float(np.trace(LtL))
    return float(np.trace(KtK)) / denom if denom > 1e-30 else 1.0


def auto_alpha(K_w: np.ndarray, L: np.ndarray, b_w: np.ndarray,
               n_iter: int = 50) -> float:
    """
    Morozov discrepancy principle: find α such that χ²(α) = N_data.

    When the chi2 floor exceeds N (NNLS can't fit the data to within noise,
    typically because the positivity constraint clashes with oscillatory noise),
    fall back to the trace-ratio heuristic which forces a smooth solution.
    When the chi2 ceiling is below N (signal too weak), keep maximum alpha.
    """
    N = len(b_w)
    target = float(N)

    def _chi2(a: float) -> float:
        K_aug = np.vstack([K_w, np.sqrt(a) * L])
        b_aug = np.concatenate([b_w, np.zeros(L.shape[0])])
        p, _ = nnls(K_aug, b_aug)
        r = K_w @ p - b_w
        return float(r @ r)

    a_lo, a_hi = 1e-8, 1e14
    c_lo = _chi2(a_lo)
    c_hi = _chi2(a_hi)

    if c_lo >= target:
        # NNLS chi2 floor > N — fall back to smooth-forcing trace-ratio alpha
        return _alpha_smooth_fallback(K_w, L)
    if c_hi <= target:
        return a_hi   # signal too weak, keep maximum regularisation

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

    Avoids explicit inversion: solve AtA @ X = K_w^T for X = M K_w^T, then
    sigma2[i] = ||X[i,:]||^2  (squared row norms of X).
    """
    try:
        AtA     = K_aug.T @ K_aug                       # (Nr, Nr)
        X       = np.linalg.solve(AtA, K_w.T)           # (Nr, NQ) = M K_w^T
        sigma2p = np.einsum('ij,ij->i', X, X)           # row sum of squares
        sigma2p = np.nan_to_num(sigma2p, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(sigma2p, 0.0)
    except Exception:
        return np.zeros(K_aug.shape[1])


def gnom_ift(
    q: np.ndarray,
    Iq: np.ndarray,
    Dmax: float,
    Nr: int = 200,
    alpha: float | None = None,
    nonneg: bool = True,
    sigma: np.ndarray | None = None,
    fit_bg: bool = False,
    chi2_factor: float = 1.0,
) -> tuple:
    """
    Compute p(r) and σ_p(r) from I(q) via Tikhonov-regularised IFT.

    When fit_bg=True, a constant background B is jointly fitted so that
    I(q) ≈ K·p(r) + B; B can be negative.

    Returns
    -------
    r  : (Nr,)
    pr : (Nr,)
    sp : (Nr,)  propagated 1σ uncertainty on p(r)
    bg : float  fitted constant background (0.0 if fit_bg=False)
    """
    from scipy.optimize import lsq_linear as _lsq_linear

    r  = np.linspace(Dmax / Nr, Dmax, Nr)
    dr = r[1] - r[0]
    K0 = _sinc_kernel(q, r, dr)       # (NQ, Nr)
    L  = _smoothness(Nr)               # (Nr-2, Nr)

    if sigma is not None:
        w    = 1.0 / np.maximum(sigma, 1e-30)
        K0_w = K0 * w[:, None]
        b_w  = Iq * w
    else:
        w    = np.ones(len(q))
        K0_w = K0.copy()
        b_w  = Iq.copy()

    # Alpha is estimated on the unaugmented system regardless of fit_bg.
    # chi2_factor multiplies the final alpha (more robust than scaling the
    # Morozov target, which can hit the chi2 ceiling at low SNR).
    if alpha is None:
        alpha = auto_alpha(K0_w, L, b_w)
        alpha *= max(chi2_factor, 1e-6)

    if fit_bg:
        # Drop last p(r) column to enforce p(Dmax)=0; bg column appended after.
        # Solution vector: [p(r_0)..p(r_{Nr-2}), bg]  — length Nr.
        K_w   = np.hstack([K0_w[:, :-1], w[:, None]])           # (NQ, Nr)
        L_bc  = L[:, :-1]                                        # (Nr-2, Nr-1)
        L_aug = np.hstack([L_bc, np.zeros((Nr - 2, 1))])         # (Nr-2, Nr)
        K_aug = np.vstack([K_w, np.sqrt(alpha) * L_aug])
        b_aug = np.concatenate([b_w, np.zeros(Nr - 2)])
        if nonneg:
            lo  = np.zeros(Nr); lo[-1] = -np.inf                 # bg can be negative
            sol = _lsq_linear(K_aug, b_aug,
                              bounds=(lo, np.full(Nr, np.inf)),
                              method='bvls').x
        else:
            sol, *_ = lstsq(K_aug, b_aug, rcond=None)
        pr = np.append(sol[:Nr - 1], 0.0)                        # hard zero at r = Dmax
        bg = float(sol[Nr - 1])
        sigma2p = _propagate_errors(K_aug, K_w)
        sp = np.append(np.sqrt(np.maximum(sigma2p[:Nr - 1], 0.0)), 0.0)
    else:
        # Drop the last column (r = Dmax) to enforce the hard boundary
        # condition p(Dmax) = 0, which prevents unphysical rising tails.
        K_w    = K0_w[:, :-1]                           # (NQ, Nr-1)
        L_bc   = L[:, :-1]                              # (Nr-2, Nr-1)
        K_aug  = np.vstack([K_w, np.sqrt(alpha) * L_bc])
        b_aug  = np.concatenate([b_w, np.zeros(Nr - 2)])
        if nonneg:
            pr_bc, _ = nnls(K_aug, b_aug)
        else:
            pr_bc, *_ = lstsq(K_aug, b_aug, rcond=None)
        pr = np.append(pr_bc, 0.0)                      # hard zero at r = Dmax
        bg = 0.0
        sigma2p = _propagate_errors(K_aug, K_w)
        sp = np.append(np.sqrt(sigma2p), 0.0)

    return r, pr, sp, bg


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
    Dmax_MM: float,
    Nr_MM: int = 200,
    Dmax_RM: float | None = None,
    Dmax_RR: float | None = None,
    Nr_RM: int | None = None,
    Nr_RR: int | None = None,
    alpha_MM: float | None = None,
    alpha_RM: float | None = None,
    alpha_RR: float | None = None,
    sigma_MM: np.ndarray | None = None,
    sigma_RM: np.ndarray | None = None,
    sigma_RR: np.ndarray | None = None,
    fit_bg_MM: bool = False,
    fit_bg_RM: bool = False,
    fit_bg_RR: bool = False,
    q_MM: np.ndarray | None = None,
    q_RM: np.ndarray | None = None,
    q_RR: np.ndarray | None = None,
    chi2_factor_MM: float = 1.0,
    chi2_factor_RM: float = 1.0,
    chi2_factor_RR: float = 1.0,
) -> dict:
    """
    IFT all three partial intensities independently with per-component D_max and N_r.

    Dmax_RM/RR and Nr_RM/RR default to the MM values if not supplied.
    I_MM, I_RR → nonneg=True   (p ≥ 0 is physical for auto-correlation terms)
    I_RM       → nonneg=False  (cross-term can be negative)
    fit_bg_*   → jointly fit a constant background B for that component

    Returns dict: r_MM, r_RM, r_RR, p_MM, p_RM, p_RR, sp_MM, sp_RM, sp_RR,
                  bg_MM, bg_RM, bg_RR
    """
    if Dmax_RM is None:
        Dmax_RM = Dmax_MM
    if Dmax_RR is None:
        Dmax_RR = Dmax_MM
    if Nr_RM is None:
        Nr_RM = Nr_MM
    if Nr_RR is None:
        Nr_RR = Nr_MM
    q_MM = q if q_MM is None else q_MM
    q_RM = q if q_RM is None else q_RM
    q_RR = q if q_RR is None else q_RR
    r_MM, p_MM, sp_MM, bg_MM = gnom_ift(q_MM, I_MM, Dmax_MM, Nr_MM, alpha_MM, nonneg=True,  sigma=sigma_MM, fit_bg=fit_bg_MM, chi2_factor=chi2_factor_MM)
    r_RM, p_RM, sp_RM, bg_RM = gnom_ift(q_RM, I_RM, Dmax_RM, Nr_RM, alpha_RM, nonneg=False, sigma=sigma_RM, fit_bg=fit_bg_RM, chi2_factor=chi2_factor_RM)
    r_RR, p_RR, sp_RR, bg_RR = gnom_ift(q_RR, I_RR, Dmax_RR, Nr_RR, alpha_RR, nonneg=True,  sigma=sigma_RR, fit_bg=fit_bg_RR, chi2_factor=chi2_factor_RR)
    return dict(r_MM=r_MM, r_RM=r_RM, r_RR=r_RR,
                p_MM=p_MM, p_RM=p_RM, p_RR=p_RR,
                sp_MM=sp_MM, sp_RM=sp_RM, sp_RR=sp_RR,
                bg_MM=bg_MM, bg_RM=bg_RM, bg_RR=bg_RR)


def joint_ift(
    q: np.ndarray,
    I_MM: np.ndarray,
    I_RM: np.ndarray,
    I_RR: np.ndarray,
    Dmax_MM: float,
    Nr_MM: int = 200,
    Dmax_RM: float | None = None,
    Dmax_RR: float | None = None,
    Nr_RM: int | None = None,
    Nr_RR: int | None = None,
    alpha: float | None = None,
    sigma_MM: np.ndarray | None = None,
    sigma_RM: np.ndarray | None = None,
    sigma_RR: np.ndarray | None = None,
    fit_bg_MM: bool = False,
    fit_bg_RM: bool = False,
    fit_bg_RR: bool = False,
    chi2_factor: float = 1.0,
) -> dict:
    """
    Joint IFT — shared α chosen by Morozov on the combined 3·NQ residual.

    Each component may have its own D_max and N_r (RM/RR default to MM values).
    The block-diagonal system decouples: since p_MM, p_RM, p_RR don't share
    variables, each is solved independently with the same α.

    Constraints: p_MM ≥ 0, p_RR ≥ 0 (NNLS/bvls), p_RM free (lstsq).
    fit_bg_*: if True, jointly fit a constant background B for that component.
    """
    from scipy.optimize import lsq_linear as _lsq_linear

    if Dmax_RM is None:
        Dmax_RM = Dmax_MM
    if Dmax_RR is None:
        Dmax_RR = Dmax_MM
    if Nr_RM is None:
        Nr_RM = Nr_MM
    if Nr_RR is None:
        Nr_RR = Nr_MM
    NQ = len(q)

    def _make_block(Dmax_i, Nr_i, Iq, sigma, fit_bg=False):
        r_i  = np.linspace(Dmax_i / Nr_i, Dmax_i, Nr_i)
        dr_i = r_i[1] - r_i[0]
        K0   = _sinc_kernel(q, r_i, dr_i)
        L_i  = _smoothness(Nr_i)
        w    = 1.0 / np.maximum(sigma, 1e-30) if sigma is not None else np.ones(NQ)
        K0_w = K0 * w[:, None]
        if fit_bg:
            K_i = np.hstack([K0,   np.ones((NQ, 1))])
            Kw  = np.hstack([K0_w, w[:, None]])
            L_i = np.hstack([L_i,  np.zeros((Nr_i - 2, 1))])
        else:
            K_i = K0
            Kw  = K0_w
        bw = Iq * w
        return r_i, K_i, L_i, Kw, bw

    r_MM, K_raw_MM, L_MM, K_MM, b_MM = _make_block(Dmax_MM, Nr_MM, I_MM, sigma_MM, fit_bg_MM)
    r_RM, K_raw_RM, L_RM, K_RM, b_RM = _make_block(Dmax_RM, Nr_RM, I_RM, sigma_RM, fit_bg_RM)
    r_RR, K_raw_RR, L_RR, K_RR, b_RR = _make_block(Dmax_RR, Nr_RR, I_RR, sigma_RR, fit_bg_RR)

    # Precompute per-block normal-equation terms for fast bisection
    blocks = [
        (K_MM, b_MM, K_MM.T @ K_MM, K_MM.T @ b_MM, L_MM, True,  Nr_MM, fit_bg_MM),
        (K_RM, b_RM, K_RM.T @ K_RM, K_RM.T @ b_RM, L_RM, False, Nr_RM, fit_bg_RM),
        (K_RR, b_RR, K_RR.T @ K_RR, K_RR.T @ b_RR, L_RR, True,  Nr_RR, fit_bg_RR),
    ]

    def _chi2_fast(a: float) -> float:
        total = 0.0
        for K_w, b_w, KtK, Ktb, L_i, _, Nr_i, _ in blocks:
            LtL_i = L_i.T @ L_i
            try:
                p = np.linalg.solve(KtK + a * LtL_i, Ktb)
            except np.linalg.LinAlgError:
                p, *_ = lstsq(KtK + a * LtL_i, Ktb, rcond=None)
            res = K_w @ p - b_w
            total += float(res @ res)
        return total

    def _solve_final(K_w, b_w, L_i, a, nonneg: bool, Nr_i: int, fit_bg: bool):
        NL_i  = L_i.shape[0]
        K_aug = np.vstack([K_w, np.sqrt(a) * L_i])
        b_aug = np.concatenate([b_w, np.zeros(NL_i)])
        N_sol = Nr_i + (1 if fit_bg else 0)
        if nonneg:
            if fit_bg:
                lo  = np.zeros(N_sol); lo[-1] = -np.inf
                sol = _lsq_linear(K_aug, b_aug,
                                  bounds=(lo, np.full(N_sol, np.inf)),
                                  method='bvls').x
            else:
                sol, _ = nnls(K_aug, b_aug)
        else:
            sol, *_ = lstsq(K_aug, b_aug, rcond=None)
        p  = sol[:Nr_i]
        bg = float(sol[Nr_i]) if fit_bg else 0.0
        resid = K_w @ sol - b_w
        return p, bg, float(resid @ resid)

    if alpha is None:
        target = float(3 * NQ)
        a_lo, a_hi = 1e-8, 1e14
        c_lo, c_hi = _chi2_fast(a_lo), _chi2_fast(a_hi)
        if c_lo >= target:
            alpha = a_lo
        elif c_hi <= target:
            alpha = a_hi
        else:
            for _ in range(30):
                a_mid = np.sqrt(a_lo * a_hi)
                if _chi2_fast(a_mid) < target:
                    a_lo = a_mid
                else:
                    a_hi = a_mid
            alpha = np.sqrt(a_lo * a_hi)
        alpha *= max(chi2_factor, 1e-6)

    p_MM, bg_MM, c_MM = _solve_final(K_MM, b_MM, L_MM, alpha, nonneg=True,  Nr_i=Nr_MM, fit_bg=fit_bg_MM)
    p_RM, bg_RM, c_RM = _solve_final(K_RM, b_RM, L_RM, alpha, nonneg=False, Nr_i=Nr_RM, fit_bg=fit_bg_RM)
    p_RR, bg_RR, c_RR = _solve_final(K_RR, b_RR, L_RR, alpha, nonneg=True,  Nr_i=Nr_RR, fit_bg=fit_bg_RR)

    chi2_red = (c_MM + c_RM + c_RR) / (3 * NQ)

    # Error propagation per block (on p(r) part only)
    sp_MM = np.sqrt(_propagate_errors(
        np.vstack([K_MM, np.sqrt(alpha) * L_MM]), K_raw_MM))[:Nr_MM]
    sp_RM = np.sqrt(_propagate_errors(
        np.vstack([K_RM, np.sqrt(alpha) * L_RM]), K_raw_RM))[:Nr_RM]
    sp_RR = np.sqrt(_propagate_errors(
        np.vstack([K_RR, np.sqrt(alpha) * L_RR]), K_raw_RR))[:Nr_RR]

    return dict(
        r_MM=r_MM, r_RM=r_RM, r_RR=r_RR,
        p_MM=p_MM, p_RM=p_RM, p_RR=p_RR,
        sp_MM=sp_MM, sp_RM=sp_RM, sp_RR=sp_RR,
        bg_MM=bg_MM, bg_RM=bg_RM, bg_RR=bg_RR,
        alpha=alpha, chi2_reduced=chi2_red,
    )
