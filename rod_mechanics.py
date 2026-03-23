
from gpu_utils import xp as np
from gpu_utils import batch_log_SO3
from parameters import (
    N, ds,
    Ks_arr, Kb_arr,
    u0_all, k0_all,
    rhoA, I_rod_arr, I_rod_inv,
    G_VEC,
    C_LIN, C_ANG,
)

ETA_S = 5e-4    # Kelvin-Voigt axial/shear viscosity [s]
ETA_B = 5e-4    # Kelvin-Voigt bending/torsion viscosity [s]

# ─────────────────────────────────────────────────────────────────────────────
# Scalar helpers (kept for body_mechanics / single-matrix use)
# ─────────────────────────────────────────────────────────────────────────────

def skew(v):
    return np.array([[ 0.0,  -v[2],  v[1]],
                     [ v[2],  0.0,  -v[0]],
                     [-v[1],  v[0],  0.0]])

def vect(S):
    return np.array([S[2,1], S[0,2], S[1,0]])

def log_SO3(R):
    """Scalar SO(3) log-map (used by clamp_reactions / body interface)."""
    cos_theta = float(np.clip(0.5*(float(np.trace(R)) - 1.0), -1.0, 1.0))
    import math
    theta = math.acos(cos_theta)
    if theta < 1.0e-8:
        return np.zeros(3)
    S = (R - R.T) / (2.0 * math.sin(theta))
    return np.array([float(S[2,1]), float(S[0,2]), float(S[1,0])]) * theta


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised strains  (N-1 segments, zero Python loops)
# ─────────────────────────────────────────────────────────────────────────────

def compute_strains(r, R, rod_id):
    
    u0 = u0_all[rod_id]   # (N-1, 3)
    k0 = k0_all[rod_id]   # (N-1, 3)

    # ── Linear strain: u[j] = R[j]ᵀ (r[j+1]-r[j]) / ds ──────────────────
    dr = (r[1:] - r[:-1]) / ds                          # (N-1, 3)
    # 'nki,nk->ni'  ≡  R[j].T @ dr[j]  for each j
    u = np.einsum('nki,nk->ni', R[:N-1], dr)            # (N-1, 3)

    # ── Curvature: kp[j] = log_SO3(R[j]ᵀ R[j+1]) / ds ───────────────────
    # R_rel[j] = R[j].T @ R[j+1]
    # 'nki,nkj->nij'  ≡  R[j].T @ R[j+1]
    R_rel = np.einsum('nki,nkj->nij', R[:N-1], R[1:])  # (N-1, 3, 3)
    kp    = batch_log_SO3(R_rel) / ds                   # (N-1, 3)

    # ── Constitutive ────────────────────────────────────────────────────
    n = np.einsum('nij,nj->ni', Ks_arr[:N-1], u  - u0) # (N-1, 3)
    m = np.einsum('nij,nj->ni', Kb_arr[:N-1], kp - k0) # (N-1, 3)

    return u, kp, n, m


# ─────────────────────────────────────────────────────────────────────────────
# Kelvin-Voigt viscous correction  (vectorised, Bug 13)
# ─────────────────────────────────────────────────────────────────────────────

def _kv_corrections(v, omega):
    """
    v, omega : (N, 3)
    Returns n_kv, m_kv : (N-1, 3)
    """
    dv_ds    = (v[1:]     - v[:-1])     / ds            # (N-1, 3)
    dom_ds   = (omega[1:] - omega[:-1]) / ds            # (N-1, 3)
    n_kv = ETA_S * np.einsum('nij,nj->ni', Ks_arr[:N-1], dv_ds)
    m_kv = ETA_B * np.einsum('nij,nj->ni', Kb_arr[:N-1], dom_ds)
    return n_kv, m_kv


# ─────────────────────────────────────────────────────────────────────────────
# Rod PDE right-hand sides  (fully vectorised, zero Python loops over N)
# ─────────────────────────────────────────────────────────────────────────────

def rod_rhs(r, R, v, omega, f_contact, f_ext_moments, rod_id,
            clamp_v, clamp_omega):
    """
    Returns dv_dt, domega_dt : (N, 3)
    Row 0 is zero — integrator enforces clamp constraint.
    """
    u, kp, n_seg_e, m_seg_e = compute_strains(r, R, rod_id)

    n_kv, m_kv = _kv_corrections(v, omega)
    n_seg = n_seg_e + n_kv     # (N-1, 3)
    m_seg = m_seg_e + m_kv     # (N-1, 3)

    # ── Build ∂_s n and ∂_s m for nodes 1..N-1 ───────────────────────────
    # Interior nodes 1..N-2: central diff  (n_seg[j] - n_seg[j-1]) / ds
    # Tip node N-1:          backward diff  (0 - n_seg[N-2]) / ds

    # n_seg has N-1 entries (seg 0..N-2)
    # For node j=1: n_seg[1]-n_seg[0]; …; node N-2: n_seg[N-2]-n_seg[N-3]
    n_diff = (n_seg[1:] - n_seg[:-1]) / ds              # (N-2, 3)  nodes 1..N-2
    n_tip  = -n_seg[-1:] / ds                           # (1, 3)    node N-1
    ds_n   = np.concatenate([n_diff, n_tip], axis=0)    # (N-1, 3)

    m_diff = (m_seg[1:] - m_seg[:-1]) / ds
    m_tip  = -m_seg[-1:] / ds
    ds_m   = np.concatenate([m_diff, m_tip], axis=0)    # (N-1, 3)

    # ── Interpolated u, n, kp, m at each node ─────────────────────────────
    # Interior nodes j=1..N-2: average of segments j-1 and j
    # Tip node j=N-1: use segment N-2 directly
    u_j  = np.concatenate([0.5*(u[:-1]     + u[1:]),     u[-1:] ],  axis=0)
    n_j  = np.concatenate([0.5*(n_seg[:-1] + n_seg[1:]), n_seg[-1:]], axis=0)
    kp_j = np.concatenate([0.5*(kp[:-1]    + kp[1:]),    kp[-1:]],   axis=0)
    m_j  = np.concatenate([0.5*(m_seg[:-1] + m_seg[1:]), m_seg[-1:]], axis=0)
    # All shape (N-1, 3)

    # ── Active nodes R[1:], v[1:], omega[1:] ─────────────────────────────
    R_a   = R[1:]              # (N-1, 3, 3)
    v_a   = v[1:]              # (N-1, 3)
    om_a  = omega[1:]          # (N-1, 3)
    rA    = rhoA[1:]           # (N-1,)

    # ── Body-frame gravity: R[j]ᵀ (rhoA[j] g)  ───────────────────────────
    # 'nki,k->ni' = R[j].T @ G_VEC  for each j
    f_grav = np.einsum('nki,k->ni', R_a, G_VEC) * rA[:, None]  # (N-1, 3)

    # ── External force body frame: R[j]ᵀ f_contact[j] ────────────────────
    # Divide by ds: f_contact is N/node; rod PDE expects N/m (force per
    # unit length). All other terms (f_grav, f_damp, ds_n) are N/m. ✓
    f_ext  = np.einsum('nki,nk->ni', R_a, f_contact[1:]) / ds  # (N-1, 3)

    # ── Rayleigh damping ─────────────────────────────────────────────────
    f_damp = -C_LIN * rA[:, None] * v_a                        # (N-1, 3)

    # ── External moment body frame: R[j]ᵀ l_ext[j] ───────────────────────
    l_ext  = np.einsum('nki,nk->ni', R_a, f_ext_moments[1:]) / ds  # /ds same reason   # (N-1, 3)

    # ── Linear momentum balance ───────────────────────────────────────────
    om_cross_rAv = np.cross(om_a, rA[:, None] * v_a)           # (N-1, 3)
    dv_active = (ds_n
                 + np.cross(kp_j, n_j)
                 - om_cross_rAv
                 + f_grav + f_ext + f_damp) / rA[:, None]       # (N-1, 3)

    # ── Angular momentum balance ──────────────────────────────────────────
    Iomj   = np.einsum('nij,nj->ni', I_rod_arr[1:], om_a)      # (N-1, 3)
    t_damp = -C_ANG * Iomj                                      # (N-1, 3)

    rhs = (ds_m
           + np.cross(kp_j, m_j)
           + np.cross(u_j,  n_j)
           - np.cross(om_a, Iomj)
           + t_damp + l_ext)                                    # (N-1, 3)

    # Precomputed inverse (avoids N linalg.solve calls):
    domega_active = np.einsum('nij,nj->ni', I_rod_inv[1:], rhs) # (N-1, 3)

    # ── Assemble output (node 0 stays zero = clamp constraint) ───────────
    dv_dt     = np.zeros((N, 3))
    domega_dt = np.zeros((N, 3))
    dv_dt[1:]     = dv_active
    domega_dt[1:] = domega_active

    return dv_dt, domega_dt


# ─────────────────────────────────────────────────────────────────────────────
# Kinematic rates  (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def rod_kinematics(R, v, omega):
    """
    dr_dt : (N, 3)     ṙ = R v
    dR_dt : (N, 3, 3)  Ṙ = R skew(ω)
    """
    dr_dt = np.einsum('nij,nj->ni', R, v)                       # (N, 3)
    dR_dt = np.matmul(R, batch_skew_local(omega))                # (N, 3, 3)
    return dr_dt, dR_dt

def batch_skew_local(v):
    """Local batch skew to avoid circular import from gpu_utils."""
    z    = np.zeros_like(v[:, 0])
    r0   = np.stack([ z,        -v[:, 2],  v[:, 1]], axis=-1)
    r1   = np.stack([ v[:, 2],  z,        -v[:, 0]], axis=-1)
    r2   = np.stack([-v[:, 1],  v[:, 0],  z        ], axis=-1)
    return np.stack([r0, r1, r2], axis=-2)


# ─────────────────────────────────────────────────────────────────────────────
# Clamp reactions (elastic only)
# ─────────────────────────────────────────────────────────────────────────────

def clamp_reactions(r, R, rod_id):
    _, _, n_seg, m_seg = compute_strains(r, R, rod_id)
    return n_seg[0].copy(), m_seg[0].copy()