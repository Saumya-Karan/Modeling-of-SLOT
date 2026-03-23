

from gpu_utils import xp as np        
from parameters import (
    MB, IB, IB_INV,
    P_ATTACH, R_I0,
    G_VEC,
    C_LIN, C_ANG,
)
from rod_mechanics import skew, clamp_reactions


def gravity_body(Rb):
    return Rb.T @ G_VEC


def aggregate_rod_loads(rod_rs, rod_Rs):
    F_rod_b = np.zeros(3)
    T_rod_b = np.zeros(3)
    for i in range(4):
        ni0, mi0 = clamp_reactions(rod_rs[i], rod_Rs[i], i)
        Ri0 = R_I0[i];  pi = P_ATTACH[i]
        f_b_i   = Ri0 @ ni0
        tau_b_i = np.cross(pi, f_b_i) - Ri0 @ mi0 
        F_rod_b += f_b_i
        T_rod_b += tau_b_i
    return F_rod_b, T_rod_b


def body_rhs(Rb, vb, wb, rod_rs, rod_Rs,
             F_ext_world=None, tau_ext_world=None):
    if F_ext_world   is None: F_ext_world   = np.zeros(3)
    if tau_ext_world is None: tau_ext_world = np.zeros(3)

    g_b       = gravity_body(Rb)
    F_b_ext   = Rb.T @ F_ext_world
    tau_b_ext = Rb.T @ tau_ext_world
    F_rod_b, T_rod_b = aggregate_rod_loads(rod_rs, rod_Rs)

    f_damp_b = -C_LIN * MB * vb
    t_damp_b = -C_ANG * IB @ wb

    dvb_dt = (
        - np.cross(wb, MB * vb)
        + F_rod_b + MB * g_b + F_b_ext + f_damp_b
    ) / MB

    Ib_wb  = IB @ wb
    dwb_dt = IB_INV @ (
        - np.cross(wb, Ib_wb)
        + T_rod_b + tau_b_ext + t_damp_b
    )
    return dvb_dt, dwb_dt


def body_kinematics(Rb, vb, wb):
    return Rb @ vb, Rb @ skew(wb)


def clamp_kinematics(vb, wb, rod_id):
    Ri0 = R_I0[rod_id];  pi = P_ATTACH[rod_id]
    v_clamp = Ri0.T @ (vb + np.cross(wb, pi))
    w_clamp = Ri0.T @ wb
    return v_clamp, w_clamp


def enforce_clamp(x, Rb, rod_r, rod_R, rod_id):
    pi  = P_ATTACH[rod_id]
    Ri0 = R_I0[rod_id]
    rod_r[0] = x + Rb @ pi
    rod_R[0] = Rb @ Ri0