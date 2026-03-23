

from gpu_utils import xp as np
from gpu_utils import (batch_cayley, batch_rodrigues,
                        batch_project_SO3, batch_vect_from_RtdR)
from parameters import N, DT, MB, r0_local_all, P_ATTACH, R_I0, K_RESTORE
from rod_mechanics   import rod_rhs, rod_kinematics, skew
from body_mechanics  import (body_rhs, body_kinematics,
                              clamp_kinematics, enforce_clamp)
from contact_tendon  import external_forces_and_moments, apply_floor_clamp

# ─────────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────────

def pack_state(x, Rb, vb, wb, rod_rs, rod_Rs, rod_vs, rod_oms):
    return {"x": x.copy(), "Rb": Rb.copy(), "vb": vb.copy(), "wb": wb.copy(),
            "rod_rs":  [r.copy() for r in rod_rs],
            "rod_Rs":  [R.copy() for R in rod_Rs],
            "rod_vs":  [v.copy() for v in rod_vs],
            "rod_oms": [o.copy() for o in rod_oms]}

def unpack_state(state):
    return (state["x"], state["Rb"], state["vb"], state["wb"],
            state["rod_rs"], state["rod_Rs"], state["rod_vs"], state["rod_oms"])

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vect_from_skew(S):
    """Scalar vect() for body rotation (3×3 → 3)."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def _om_body_single(R0, dR):
    """ω = vect(R0ᵀ dR)  for a single (3,3) pair."""
    RtdR = R0.T @ dR
    return np.array([RtdR[2, 1], RtdR[0, 2], RtdR[1, 0]])

# ─────────────────────────────────────────────────────────────────────────────
# Clamp velocity constraint  (Bug 10)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_clamp_velocity(rod_vs_list, rod_oms_list, vb, wb):
    for i in range(4):
        cv, co = clamp_kinematics(vb, wb, i)
        rod_vs_list[i][0] = cv
        rod_oms_list[i][0] = co


def _apply_tip_ground_constraint(rod_rs_list, rod_Rs_list, rod_vs_list):
    from parameters import GROUND_Z as _GZ
    for i in range(4):
        rod_rs_list[i][-1, 2] = _GZ
        R_tip   = rod_Rs_list[i][-1]           # (3, 3) director at tip
        v_body  = rod_vs_list[i][-1]           # (3,)   tip velocity, body frame
        v_world = R_tip @ v_body               # (3,)   world frame
        v_world_no_z = np.stack([v_world[0], v_world[1],
                                  np.zeros_like(v_world[2])])
        rod_vs_list[i][-1] = R_tip.T @ v_world_no_z   # back to body frame

# ─────────────────────────────────────────────────────────────────────────────
# Full system derivatives
# ─────────────────────────────────────────────────────────────────────────────

def _derivatives(t, state):
    (x, Rb, vb, wb,
     rod_rs, rod_Rs, rod_vs, rod_oms) = unpack_state(state)

    # 1. Enforce geometric clamp
    for i in range(4):
        enforce_clamp(x, Rb, rod_rs[i], rod_Rs[i], i)

    # 2. World-frame node velocities  (vectorised — was: for j in range(N))
    rod_vw = []
    for i in range(4):
        # einsum 'nij,nj->ni'  ≡  R[j] @ v[j]  for all j
        rod_vw.append(np.einsum('nij,nj->ni', rod_Rs[i], rod_vs[i]))  # (N,3)

    # 3. External forces + moments
    rod_fext, rod_mext = [], []
    for i in range(4):
        F, M = external_forces_and_moments(t, i, rod_rs[i], rod_Rs[i], rod_vw[i])
        rod_fext.append(F);  rod_mext.append(M)

    for i in range(4):
        # Rb @ R_I0[i]: rotate leg-frame offsets to world frame
        Rb_Ri0 = Rb @ R_I0[i]                                    # (3, 3)
        p_world = Rb @ P_ATTACH[i]                               # (3,)
        # r_desired: (N, 3)
        r_desired = (x[None, :]
                     + p_world[None, :]
                     + np.einsum('ij,kj->ki', Rb_Ri0, r0_local_all[i]))
        f_restore = -K_RESTORE * (rod_rs[i] - r_desired)         # (N, 3)
        rod_fext[i] = rod_fext[i] + f_restore

    # 4. Clamp BCs
    clamp_vs, clamp_oms = [], []
    for i in range(4):
        cv, co = clamp_kinematics(vb, wb, i)
        clamp_vs.append(cv);  clamp_oms.append(co)

    # 5. Rod PDE RHS
    drod_vs, drod_oms = [], []
    for i in range(4):
        dv, dom = rod_rhs(rod_rs[i], rod_Rs[i], rod_vs[i], rod_oms[i],
                          rod_fext[i], rod_mext[i], i,
                          clamp_vs[i], clamp_oms[i])
        drod_vs.append(dv);  drod_oms.append(dom)

    # 6. Body EOM
    dvb, dwb = body_rhs(Rb, vb, wb, rod_rs, rod_Rs)

    # 7. Kinematic rates
    dx, dRb = body_kinematics(Rb, vb, wb)
    drod_rs, drod_Rs = [], []
    for i in range(4):
        dr, dR = rod_kinematics(rod_Rs[i], rod_vs[i], rod_oms[i])
        drod_rs.append(dr);  drod_Rs.append(dR)

    return {"x": dx, "Rb": dRb, "vb": dvb, "wb": dwb,
            "rod_rs": drod_rs, "rod_Rs": drod_Rs,
            "rod_vs": drod_vs, "rod_oms": drod_oms}

# ─────────────────────────────────────────────────────────────────────────────
# Provisional state for RK4 intermediate stages  (vectorised Cayley)
# ─────────────────────────────────────────────────────────────────────────────

def _add_scaled(state, deriv, h):
    (x, Rb, vb, wb,
     rod_rs, rod_Rs, rod_vs, rod_oms) = unpack_state(state)

    new_x  = x  + h * deriv["x"]
    new_vb = vb + h * deriv["vb"]
    new_wb = wb + h * deriv["wb"]

    # Body rotation — Bug 5: R^T @ Ṙ → body-frame ω, scalar Cayley
    om_b   = _om_body_single(Rb, deriv["Rb"])
    # rodrigues/cayley for single matrix via batch op
    new_Rb = Rb @ batch_cayley(om_b[None], h)[0]

    new_rod_rs, new_rod_Rs, new_rod_vs, new_rod_oms = [], [], [], []
    for i in range(4):
        new_rod_rs.append( rod_rs[i]  + h * deriv["rod_rs"][i])
        new_rod_vs.append( rod_vs[i]  + h * deriv["rod_vs"][i])
        new_rod_oms.append(rod_oms[i] + h * deriv["rod_oms"][i])

        # Vectorised Cayley update for all N nodes  (was: for j in range(N))
        # Bug 6: R^T @ Ṙ gives body-frame ω per node
        om_all = batch_vect_from_RtdR(rod_Rs[i],          # (N, 3)
                                       deriv["rod_Rs"][i])
        dR_cay = batch_cayley(om_all, h)                   # (N, 3, 3)
        new_Ri = np.matmul(rod_Rs[i], dR_cay)             # (N, 3, 3)
        new_rod_Rs.append(new_Ri)

    # Bug 10: overwrite node 0 with constraint velocity
    _apply_clamp_velocity(new_rod_vs, new_rod_oms, new_vb, new_wb)
    # Tip ground pin: node N-1 stays at z=0, no z-velocity (mirrors clamp at node 0)
    _apply_tip_ground_constraint(new_rod_rs, new_rod_Rs, new_rod_vs)

    return pack_state(new_x, new_Rb, new_vb, new_wb,
                      new_rod_rs, new_rod_Rs, new_rod_vs, new_rod_oms)

# ─────────────────────────────────────────────────────────────────────────────
# Geometric RK4 step  (vectorised final accumulation)
# ─────────────────────────────────────────────────────────────────────────────

def rk4_step(t, state, dt):
    (x0, Rb0, vb0, wb0,
     rod_rs0, rod_Rs0, rod_vs0, rod_oms0) = unpack_state(state)

    k1 = _derivatives(t,           state)
    k2 = _derivatives(t + 0.5*dt,  _add_scaled(state, k1, 0.5*dt))
    k3 = _derivatives(t + 0.5*dt,  _add_scaled(state, k2, 0.5*dt))
    k4 = _derivatives(t +     dt,  _add_scaled(state, k3,     dt))

    def _w(a, b, c, d): return (a + 2.0*b + 2.0*c + d) / 6.0

    # ── Euclidean ─────────────────────────────────────────────────────────
    new_x  = x0  + dt * _w(k1["x"],  k2["x"],  k3["x"],  k4["x"])
    new_vb = vb0 + dt * _w(k1["vb"], k2["vb"], k3["vb"], k4["vb"])
    new_wb = wb0 + dt * _w(k1["wb"], k2["wb"], k3["wb"], k4["wb"])

    # ── Body SO(3) — scalar Rodrigues + Newton project ────────────────────
    om_b1 = _om_body_single(Rb0, k1["Rb"])
    om_b2 = _om_body_single(Rb0, k2["Rb"])
    om_b3 = _om_body_single(Rb0, k3["Rb"])
    om_b4 = _om_body_single(Rb0, k4["Rb"])
    om_eff_b = _w(om_b1, om_b2, om_b3, om_b4)
    dR_b     = batch_rodrigues(om_eff_b[None], dt)[0]
    new_Rb   = batch_project_SO3((Rb0 @ dR_b)[None])[0]

    # ── Rod nodes — vectorised Rodrigues + Newton project ─────────────────
    new_rod_rs, new_rod_Rs, new_rod_vs, new_rod_oms = [], [], [], []
    for i in range(4):
        new_rod_rs.append(rod_rs0[i] + dt * _w(k1["rod_rs"][i], k2["rod_rs"][i],
                                                k3["rod_rs"][i], k4["rod_rs"][i]))
        new_rod_vs.append(rod_vs0[i] + dt * _w(k1["rod_vs"][i], k2["rod_vs"][i],
                                                k3["rod_vs"][i], k4["rod_vs"][i]))
        new_rod_oms.append(rod_oms0[i] + dt * _w(k1["rod_oms"][i], k2["rod_oms"][i],
                                                  k3["rod_oms"][i], k4["rod_oms"][i]))

        # Vectorised ω per node across all 4 stages
        om1 = batch_vect_from_RtdR(rod_Rs0[i], k1["rod_Rs"][i])
        om2 = batch_vect_from_RtdR(rod_Rs0[i], k2["rod_Rs"][i])
        om3 = batch_vect_from_RtdR(rod_Rs0[i], k3["rod_Rs"][i])
        om4 = batch_vect_from_RtdR(rod_Rs0[i], k4["rod_Rs"][i])
        om_eff = _w(om1, om2, om3, om4)                   # (N, 3)

        dR_rod  = batch_rodrigues(om_eff, dt)              # (N, 3, 3)
        new_Ri  = np.matmul(rod_Rs0[i], dR_rod)           # (N, 3, 3)
        new_Ri  = batch_project_SO3(new_Ri)                # (N, 3, 3) 
        new_rod_Rs.append(new_Ri)

    new_state = pack_state(new_x, new_Rb, new_vb, new_wb,
                           new_rod_rs, new_rod_Rs, new_rod_vs, new_rod_oms)

    # ── Post-step corrections ─────────────────────────────────────────────
    for i in range(4):
        enforce_clamp(new_state["x"], new_state["Rb"],
                      new_state["rod_rs"][i], new_state["rod_Rs"][i], i)

    _apply_clamp_velocity(new_state["rod_vs"], new_state["rod_oms"],
                          new_state["vb"], new_state["wb"])

    # Tip ground pin (node N-1) — bilateral: tip cannot lift off ground
    _apply_tip_ground_constraint(new_state["rod_rs"], new_state["rod_Rs"],
                                  new_state["rod_vs"])

    for i in range(4):
        # Vectorised world velocities
        vw = np.einsum('nij,nj->ni', new_state["rod_Rs"][i],
                       new_state["rod_vs"][i])              # (N, 3)
        apply_floor_clamp(new_state["rod_rs"][i], vw)
        # Write corrected velocities back to body frame
        new_state["rod_vs"][i] = np.einsum('nji,nj->ni',
                                            new_state["rod_Rs"][i], vw)  # Rᵀ vw

    return new_state