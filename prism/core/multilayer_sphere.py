"""
Multilayer sphere forward model for ASAXS partial structure factors.

Shared between:
  - prism.sim_data_tab  (simulation / data generation)
  - XModFit2 MultiLayerSphere function (model fitting)

Physics
-------
For a concentric N-layer sphere with polydisperse layer thicknesses:

  I(q, E) = I_MM + 2 f'(E) I_RM + [f'(E)² + f''(E)²] I_RR

where:
  I_MM  = ⟨|F_total(q)|²⟩  — SAXS-term (total electron density contrast)
  I_RM  = ⟨Re(F_total F_r*)⟩ — Cross-term
  I_RR  = ⟨|F_r(q)|²⟩       — Resonant-term (resonant atoms only)
  f', f'' = real/imaginary anomalous corrections (electrons/atom, from xraydb)

Polydispersity is handled by Gauss-Hermite tensor-product quadrature
applied independently to each layer's thickness.

Normalization
-------------
All returned intensities are in absolute units [cm⁻¹]:
  I = (φ / V̄_p) × r_e² × ⟨|F|²⟩
where φ is the particle volume fraction, V̄_p the mean particle volume,
and r_e = 2.818e-13 cm is the Thomson radius.
"""

import numpy as np

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def _njit(**kw):
        return lambda f: f


# ── Basic geometry helpers ─────────────────────────────────────────────────────

def elec_density(rho, M, ne):
    """Electron number density [el/Å³] from mass density, molar mass, electrons/formula."""
    return (rho / M) * 6.022e23 * ne * 1e-24


def atom_volume(rho, M):
    """Volume per formula unit [Å³] from mass density and molar mass."""
    return M / (rho * 6.022e23) * 1e24


def sphere_V(R):
    return (4 / 3) * np.pi * R ** 3


def sphere_f0_2d(q, R_arr):
    """Sphere form factor f0(qR), broadcast over (NQ, M) arrays."""
    x = q[:, None] * R_arr[None, :]
    return np.where(np.abs(x) < 1e-8, 1.0,
                    3 * (np.sin(x) - x * np.cos(x)) / x ** 3)


# ── Gauss-Hermite quadrature ───────────────────────────────────────────────────

def gh_nodes(K):
    """K-point Gauss-Hermite nodes and weights normalised to sum = 1."""
    x, w = np.polynomial.hermite.hermgauss(K)
    return x, w / np.sqrt(np.pi)


def layer_grid(mean, sigma, dist_type, K):
    """(values, weights) for one layer's thickness polydispersity.

    Returns a single point (monodisperse) when sigma ≤ 0 or
    dist_type == 'Monodisperse'.
    """
    if sigma <= 0 or dist_type.lower() == 'monodisperse':
        return np.array([float(mean)]), np.array([1.0])
    x, w = gh_nodes(K)
    if dist_type.lower() == 'gaussian':
        vals = mean + np.sqrt(2.0) * sigma * x
        vals = np.maximum(vals, 1e-4)
    else:                           # log-normal
        s_ln = np.sqrt(np.log(1.0 + (sigma / mean) ** 2))
        m_ln = np.log(mean) - 0.5 * s_ln ** 2
        vals = np.exp(m_ln + np.sqrt(2.0) * s_ln * x)
    return vals, w


# ── NumPy quadrature loop ──────────────────────────────────────────────────────

def _quad_np(q, t_combos, W_total, rhos, rho_v, layer_idx_arr, V_at_arr):
    """Vectorised (NumPy) evaluation of ASAXS partial integrals."""
    NQ, M, N = len(q), t_combos.shape[0], t_combos.shape[1]
    n_e = len(layer_idx_arr)
    radii = np.cumsum(t_combos, axis=1)            # (M, N)

    Fn = np.zeros((NQ, M))
    rho_next = rho_v
    for i in range(N - 1, -1, -1):
        Ri  = radii[:, i]
        Fn += (rhos[i] - rho_next) * sphere_V(Ri)[None, :] * sphere_f0_2d(q, Ri)
        rho_next = rhos[i]

    W       = W_total[None, :]
    I_MM    = np.sum(W * Fn * Fn, axis=1)
    Vp      = float(np.sum(W_total * sphere_V(radii[:, -1])))
    I_RiM   = np.zeros((n_e, NQ))
    I_RiRi  = np.zeros((n_e, NQ))
    I_RiRj  = {}
    Fr_list = []

    for ei, j in enumerate(layer_idx_arr):
        Rj  = radii[:, j]
        f0j = sphere_f0_2d(q, Rj)
        if j > 0:
            Rm1  = radii[:, j - 1]
            shell = sphere_V(Rj)[None, :] * f0j - sphere_V(Rm1)[None, :] * sphere_f0_2d(q, Rm1)
        else:
            shell = sphere_V(Rj)[None, :] * f0j
        Fr = shell / V_at_arr[ei]
        Fr_list.append(Fr)
        I_RiM[ei]  = np.sum(W * Fn * Fr, axis=1)
        I_RiRi[ei] = np.sum(W * Fr * Fr, axis=1)

    for i in range(n_e):
        for jj in range(i + 1, n_e):
            I_RiRj[(i, jj)] = np.sum(W * Fr_list[i] * Fr_list[jj], axis=1)

    return I_MM, I_RiM, I_RiRi, I_RiRj, Vp


# ── Numba JIT (compiled on first call, with NumPy fallback) ───────────────────

@_njit(cache=True, fastmath=True)
def _quad_nb(q_arr, t_combos, W_total, rhos_arr, rho_v, layer_idx_arr, V_at_arr):
    PI43 = 4.188790204786391
    NQ   = q_arr.shape[0]
    M    = t_combos.shape[0]
    N    = t_combos.shape[1]
    n_e  = layer_idx_arr.shape[0]

    I_MM   = np.zeros(NQ)
    I_RiM  = np.zeros((n_e, NQ))
    I_RiRi = np.zeros((n_e, NQ))
    Vp     = 0.0
    radii  = np.empty(N)
    Fn     = np.empty(NQ)
    Fr_e   = np.empty(NQ)

    for m in range(M):
        w = W_total[m]
        radii[0] = t_combos[m, 0]
        for i in range(1, N):
            radii[i] = radii[i - 1] + t_combos[m, i]

        for qi in range(NQ): Fn[qi] = 0.0
        rho_next = rho_v
        for i in range(N - 1, -1, -1):
            R = radii[i]; V = PI43 * R * R * R; drho = rhos_arr[i] - rho_next
            for qi in range(NQ):
                qR = q_arr[qi] * R
                f0 = 1.0 if abs(qR) < 1e-8 else 3.0 * (np.sin(qR) - qR * np.cos(qR)) / (qR * qR * qR)
                Fn[qi] += drho * V * f0
            rho_next = rhos_arr[i]

        for qi in range(NQ): I_MM[qi] += w * Fn[qi] * Fn[qi]
        Vp += w * PI43 * radii[N - 1] * radii[N - 1] * radii[N - 1]

        for ei in range(n_e):
            j = layer_idx_arr[ei]; Rj = radii[j]; Vj = PI43 * Rj * Rj * Rj
            for qi in range(NQ):
                qRj = q_arr[qi] * Rj
                f0j = 1.0 if abs(qRj) < 1e-8 else 3.0 * (np.sin(qRj) - qRj * np.cos(qRj)) / (qRj * qRj * qRj)
                if j > 0:
                    Rm1 = radii[j - 1]; Vm1 = PI43 * Rm1 * Rm1 * Rm1; qRm1 = q_arr[qi] * Rm1
                    f0m1 = 1.0 if abs(qRm1) < 1e-8 else 3.0 * (np.sin(qRm1) - qRm1 * np.cos(qRm1)) / (qRm1 * qRm1 * qRm1)
                    Fr_e[qi] = (Vj * f0j - Vm1 * f0m1) / V_at_arr[ei]
                else:
                    Fr_e[qi] = Vj * f0j / V_at_arr[ei]
            for qi in range(NQ):
                I_RiM[ei, qi]  += w * Fn[qi] * Fr_e[qi]
                I_RiRi[ei, qi] += w * Fr_e[qi] * Fr_e[qi]

    return I_MM, I_RiM, I_RiRi, Vp


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_multilayer(q, layer_cfgs, solvent_cfg, phi, elem_cfgs, dist_type, K=12, log=None):
    """
    Compute ASAXS partial structure factors for an N-layer sphere.

    Parameters
    ----------
    q           : 1-D array, Å⁻¹
    layer_cfgs  : list of dicts (core first, NO solvent).
                  Required keys: density [g/cm³], molar_mass [g/mol],
                                 n_electrons [int], thickness [Å], sigma [Å].
    solvent_cfg : dict with density, molar_mass, n_electrons.
    phi         : particle volume fraction.
    elem_cfgs   : list of dicts, one per resonant element.
                  Required key: layer_idx (0-based index into layer_cfgs).
    dist_type   : 'Monodisperse' | 'Gaussian' | 'Log-Normal'
    K           : Gauss-Hermite quadrature points per active layer.
    log         : optional callable for progress messages.

    Returns
    -------
    I_MM    : (NQ,)        SAXS-term         [cm⁻¹]
    I_RiM   : (n_e, NQ)   Cross-terms       [cm⁻¹]
    I_RiRi  : (n_e, NQ)   Resonant-terms    [cm⁻¹]
    I_RiRj  : dict (i,j)→(NQ,) cross-resonant terms (≥2 elements)
    """
    rho_v = elec_density(solvent_cfg['density'], solvent_cfg['molar_mass'],
                         solvent_cfg['n_electrons'])
    rhos  = np.array([elec_density(L['density'], L['molar_mass'], L['n_electrons'])
                      for L in layer_cfgs], dtype=np.float64)
    n_e   = len(elem_cfgs)

    grids     = [layer_grid(L['thickness'], L.get('sigma', 0.0), dist_type, K)
                 for L in layer_cfgs]
    n_active  = sum(1 for v, _ in grids if len(v) > 1)
    M_total   = int(np.prod([len(v) for v, _ in grids]))
    if log:
        log(f'  Polydispersity: {n_active} active layer(s)  K={K}  '
            f'M={M_total} quad pts  backend={"Numba" if (_HAS_NUMBA and M_total >= 500) else "NumPy"}')

    mesh_v   = np.meshgrid(*[g[0] for g in grids], indexing='ij')
    mesh_w   = np.meshgrid(*[g[1] for g in grids], indexing='ij')
    W_total  = np.ones_like(mesh_w[0], dtype=np.float64)
    for mw in mesh_w: W_total *= mw
    W_total  = W_total.ravel()
    t_combos = np.stack([mv.ravel() for mv in mesh_v], axis=1).astype(np.float64)

    layer_idx_arr = np.array([int(e['layer_idx']) for e in elem_cfgs], dtype=np.int64)
    V_at_arr      = np.array([
        atom_volume(layer_cfgs[int(e['layer_idx'])]['density'],
                    layer_cfgs[int(e['layer_idx'])]['molar_mass'])
        for e in elem_cfgs], dtype=np.float64)

    if _HAS_NUMBA and M_total >= 500:
        I_MM_r, I_RiM_r, I_RiRi_r, Vp = _quad_nb(
            q.astype(np.float64), t_combos, W_total,
            rhos, float(rho_v), layer_idx_arr, V_at_arr)
        I_RiRj: dict = {}
        if n_e > 1:
            radii   = np.cumsum(t_combos, axis=1); W = W_total[None, :]
            Fr_list = []
            for ei, j in enumerate(layer_idx_arr):
                Rj  = radii[:, j]; f0j = sphere_f0_2d(q, Rj)
                shell = (sphere_V(Rj)[None, :] * f0j
                         - sphere_V(radii[:, j-1])[None, :] * sphere_f0_2d(q, radii[:, j-1])
                         if j > 0 else sphere_V(Rj)[None, :] * f0j)
                Fr_list.append(shell / V_at_arr[ei])
            for i in range(n_e):
                for jj in range(i + 1, n_e):
                    I_RiRj[(i, jj)] = np.sum(W * Fr_list[i] * Fr_list[jj], axis=1)
        I_RiM  = [I_RiM_r[i]  for i in range(n_e)]
        I_RiRi = [I_RiRi_r[i] for i in range(n_e)]
    else:
        I_MM_r, I_RiM_r, I_RiRi_r, I_RiRj, Vp = _quad_np(
            q, t_combos, W_total, rhos, rho_v, layer_idx_arr, V_at_arr)
        I_RiM  = [I_RiM_r[i]  for i in range(n_e)]
        I_RiRi = [I_RiRi_r[i] for i in range(n_e)]

    sc = phi / Vp * 1e24 * (2.818e-13) ** 2
    return (sc * I_MM_r,
            [sc * x for x in I_RiM],
            [sc * x for x in I_RiRi],
            {k: sc * v for k, v in I_RiRj.items()})


def rho_profile(layer_cfgs, solvent_cfg, n_pts=500, tail_frac=0.15):
    """Electron density step-function profile for display [el/Å³] vs r [Å]."""
    r_edges = np.concatenate([[0.0], np.cumsum([L['thickness'] for L in layer_cfgs])])
    rho_layers = [elec_density(L['density'], L['molar_mass'], L['n_electrons'])
                  for L in layer_cfgs]
    rho_s = elec_density(solvent_cfg['density'], solvent_cfg['molar_mass'],
                         solvent_cfg['n_electrons'])
    r = np.linspace(0, r_edges[-1] * (1 + tail_frac), n_pts)
    rho = np.full_like(r, rho_s)
    for i, rho_i in enumerate(rho_layers):
        rho[(r >= r_edges[i]) & (r < r_edges[i + 1])] = rho_i
    return r, rho
