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

ETA_S = 5e-4    
ETA_B = 5e-4    

# ─────────────────────────────────────────────────────────────────────────────
# Scalar helpers (kept for body_mechanics / single-matrix use)
# ─────────────────────────────────────────────────────────────────────────────

def skew(v):
    K = np.zeros((3, 3))
    K[0, 1] = -v[2];  K[0, 2] =  v[1]
    K[1, 0] =  v[2];  K[1, 2] = -v[0]
    K[2, 0] = -v[1];  K[2, 1] =  v[0]
    return K

def vect(S):
    out = np.zeros(3)
    out[0] = S[2, 1];  out[1] = S[0, 2];  out[2] = S[1, 0]
    return out

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
    """
    Parameters
    ----------
    r  : (N, 3)    node positions (world frame)
    R  : (N, 3, 3) director matrices

    Returns
    -------
    u  : (N-1, 3)  linear strain
    kp : (N-1, 3)  curvature
    n  : (N-1, 3)  force resultant
    m  : (N-1, 3)  moment resultant
    """
    u0 = u0_all[rod_id]   
    k0 = k0_all[rod_id]   

    dr = (r[1:] - r[:-1]) / ds                          
    u = np.einsum('nki,nk->ni', R[:N-1], dr)            

    R_rel = np.einsum('nki,nkj->nij', R[:N-1], R[1:])  
    kp    = batch_log_SO3(R_rel) / ds                   

    # ── Constitutive ────────────────────────────────────────────────────
    n = np.einsum('nij,nj->ni', Ks_arr[:N-1], u  - u0)
    m = np.einsum('nij,nj->ni', Kb_arr[:N-1], kp - k0)

    return u, kp, n, m


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


def rod_rhs(r, R, v, omega, f_contact, f_ext_moments, rod_id,
            clamp_v, clamp_omega):
    """
    Returns dv_dt, domega_dt : (N, 3)
    Row 0 is zero — integrator enforces clamp constraint.
    """
    u, kp, n_seg_e, m_seg_e = compute_strains(r, R, rod_id)

    n_kv, m_kv = _kv_corrections(v, omega)
    n_seg = n_seg_e + n_kv    
    m_seg = m_seg_e + m_kv     

    n_diff = (n_seg[1:] - n_seg[:-1]) / ds              
    n_tip  = -n_seg[-1:] / ds                           
    ds_n   = np.concatenate([n_diff, n_tip], axis=0)   

    m_diff = (m_seg[1:] - m_seg[:-1]) / ds
    m_tip  = -m_seg[-1:] / ds
    ds_m   = np.concatenate([m_diff, m_tip], axis=0)    # (N-1, 3)

    u_j  = np.concatenate([0.5*(u[:-1]     + u[1:]),     u[-1:] ],  axis=0)
    n_j  = np.concatenate([0.5*(n_seg[:-1] + n_seg[1:]), n_seg[-1:]], axis=0)
    kp_j = np.concatenate([0.5*(kp[:-1]    + kp[1:]),    kp[-1:]],   axis=0)
    m_j  = np.concatenate([0.5*(m_seg[:-1] + m_seg[1:]), m_seg[-1:]], axis=0)

    R_a   = R[1:]              
    v_a   = v[1:]              
    om_a  = omega[1:]         
    rA    = rhoA[1:]           

    f_grav = np.einsum('nki,k->ni', R_a, G_VEC) * rA[:, None]  

    f_ext  = np.einsum('nki,nk->ni', R_a, f_contact[1:]) / ds  

    f_damp = -C_LIN * rA[:, None] * v_a                        

    l_ext  = np.einsum('nki,nk->ni', R_a, f_ext_moments[1:]) / ds  

    om_cross_rAv = np.cross(om_a, rA[:, None] * v_a)           
    dv_active = (ds_n
                 + np.cross(kp_j, n_j)
                 - om_cross_rAv
                 + f_grav + f_ext + f_damp) / rA[:, None]      

    Iomj   = np.einsum('nij,nj->ni', I_rod_arr[1:], om_a)      
    t_damp = -C_ANG * Iomj                                      

    rhs = (ds_m
           + np.cross(kp_j, m_j)
           + np.cross(u_j,  n_j)
           - np.cross(om_a, Iomj)
           + t_damp + l_ext)                                    


    domega_active = np.einsum('nij,nj->ni', I_rod_inv[1:], rhs) 


    dv_dt     = np.zeros((N, 3))
    domega_dt = np.zeros((N, 3))
    dv_dt[1:]     = dv_active
    domega_dt[1:] = domega_active

    return dv_dt, domega_dt

def rod_kinematics(R, v, omega):
    """
    dr_dt : (N, 3)     ṙ = R v
    dR_dt : (N, 3, 3)  Ṙ = R skew(ω)
    """
    dr_dt = np.einsum('nij,nj->ni', R, v)                      
    dR_dt = np.matmul(R, batch_skew_local(omega))                
    return dr_dt, dR_dt

def batch_skew_local(v):
    """Local batch skew to avoid circular import from gpu_utils."""
    z    = np.zeros_like(v[:, 0])
    r0   = np.stack([ z,        -v[:, 2],  v[:, 1]], axis=-1)
    r1   = np.stack([ v[:, 2],  z,        -v[:, 0]], axis=-1)
    r2   = np.stack([-v[:, 1],  v[:, 0],  z        ], axis=-1)
    return np.stack([r0, r1, r2], axis=-2)

def clamp_reactions(r, R, rod_id):
    _, _, n_seg, m_seg = compute_strains(r, R, rod_id)
    return n_seg[0].copy(), m_seg[0].copy()