

from gpu_utils import xp as np
from parameters import (
    N,
    MU_CONTACT, GROUND_Z, K_CONTACT, D_CONTACT,
    TENDON_ANGLE, R0_TEN, TENDON_NODES,
    THICK_FIXED, THICK_TIP,
    tendon_magnitude,
    R_I0,
)

import math as _math

_ca_ten = _math.cos(TENDON_ANGLE)
_sa_ten = _math.sin(TENDON_ANGLE)

_thickness = THICK_FIXED + (THICK_TIP - THICK_FIXED) * np.linspace(0.0, 1.0, N)

def ground_contact_forces(r, v_world):
    """
    r       : (N, 3)  node positions
    v_world : (N, 3)  world-frame velocities
    Returns   (N, 3)  world-frame contact force
    """
    pen  = GROUND_Z - r[:, 2]                              # (N,) penetration
    mask = pen > 0.0                                       # (N,) bool

    # Normal force
    F_n = np.maximum(K_CONTACT * pen - D_CONTACT * v_world[:, 2], 0.0)
    F_n = np.where(mask, F_n, 0.0)                         # (N,)

    # Tangential speed
    vx, vy = v_world[:, 0], v_world[:, 1]
    v_tan  = np.sqrt(vx**2 + vy**2)                       # (N,)
    in_mot = mask & (v_tan > 1.0e-8)

    safe_vt = np.where(v_tan > 1.0e-8, v_tan, 1.0)
    F_fx = np.where(in_mot, -MU_CONTACT * F_n * vx / safe_vt, 0.0)
    F_fy = np.where(in_mot, -MU_CONTACT * F_n * vy / safe_vt, 0.0)

    return np.stack([F_fx, F_fy, F_n], axis=-1)            # (N, 3)

def tendon_forces_and_moments(t, rod_R):
    """
    t     : float
    rod_R : (N, 3, 3)

    Returns F_tend, M_tend : (N, 3) world-frame
    """
    F_tend = np.zeros((N, 3))
    M_tend = np.zeros((N, 3))

    T_total = tendon_magnitude(t)
    if T_total < 1.0e-12:
        return F_tend, M_tend

    T_per = T_total / TENDON_NODES

    R_a  = rod_R[:TENDON_NODES]                           # (K, 3, 3)
    d1   = R_a[:, :, 0]                                   # (K, 3)
    d3   = R_a[:, :, 2]                                   # (K, 3)

    t_vec = _ca_ten * d3 + _sa_ten * d1                   # (K, 3)
    norm  = np.linalg.norm(t_vec, axis=-1, keepdims=True) # (K, 1)
    norm  = np.where(norm > 1.0e-12, norm, 1.0)
    t_hat = t_vec / norm                                   # (K, 3)

    F_j   = T_per * t_hat                                 # (K, 3)

    half_t = (_thickness[:TENDON_NODES])[:, None]         # (K, 1)
    r_off  = -half_t * d1                                 # (K, 3)
    M_j    = np.cross(r_off, F_j)                         # (K, 3)

    F_tend = np.zeros((N, 3))
    M_tend = np.zeros((N, 3))
    F_tend[:TENDON_NODES] = F_j
    M_tend[:TENDON_NODES] = M_j

    return F_tend, M_tend


def external_forces_and_moments(t, rod_id, rod_r, rod_R, v_world):
    F_c        = ground_contact_forces(rod_r, v_world)
    F_t, M_t   = tendon_forces_and_moments(t, rod_R)
    return F_c + F_t, M_t



def apply_floor_clamp(r, v_world):
    below = r[:, 2] < GROUND_Z
    if np.any(below):
        r[:, 2] = np.where(below, GROUND_Z, r[:, 2])