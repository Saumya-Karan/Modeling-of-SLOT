

from gpu_utils import xp as np   # GPU if CuPy available, else NumPy

# ─────────────────────────────────────────────────────────────────────────────
# Rod discretisation
# ─────────────────────────────────────────────────────────────────────────────
L  = 0.180
N  = 21
ds = L / (N - 1)
s  = np.linspace(0.0, L, N)

# ─────────────────────────────────────────────────────────────────────────────
# Cross-section (tapered rectangular — from rod36.py)
# ─────────────────────────────────────────────────────────────────────────────
THICK_FIXED = 0.0135
THICK_TIP   = 0.0035
WIDTH       = 0.020            # constant width 2 cm  (rod36.py: width = 0.02)

thickness = THICK_FIXED + (THICK_TIP - THICK_FIXED) * np.linspace(0.0, 1.0, N)
A_cs      = thickness * WIDTH                      # rod36.py: A = thickness * width
I_bend    = WIDTH * thickness**3 / 12.0            # rod36.py: I = (width * thickness**3) / 12.0
J_tor     = WIDTH * thickness**3 / 3.0

# ─────────────────────────────────────────────────────────────────────────────
# Material properties — rod36.py values
#   E_default = 10e6 Pa (10 MPa flexible TPU)
#   rho = 1200 kg/m³
# ─────────────────────────────────────────────────────────────────────────────
E_MOD = 10.0e6           # rod36.py: E_default = 10e6
NU    = 0.3
G_MOD = E_MOD / (2.0 * (1.0 + NU))
RHO   = 1200.0           # rod36.py: rho = 1200

rhoA = RHO * A_cs

# Mass per node — matches rod36.py exactly
#   m[1:-1] = (rho*A[:-2]*dx + rho*A[1:-1]*dx) / 2
#   m[0]    = rho*A[0]*dx/2
#   m[-1]   = rho*A[-1]*dx/2
node_mass       = np.zeros(N)
node_mass[1:-1] = 0.5 * (rhoA[:-2] + rhoA[1:-1]) * ds   # rod36.py formula
node_mass[0]    = 0.5 * rhoA[0]  * ds
node_mass[-1]   = 0.5 * rhoA[-1] * ds

# ─────────────────────────────────────────────────────────────────────────────
# Stiffness matrices
# ─────────────────────────────────────────────────────────────────────────────
Ks_arr = np.zeros((N, 3, 3))
Kb_arr = np.zeros((N, 3, 3))
for j in range(N):
    Ks_arr[j] = np.diag(np.array([G_MOD*float(A_cs[j]), G_MOD*float(A_cs[j]), E_MOD*float(A_cs[j])]))
    Kb_arr[j] = np.diag(np.array([E_MOD*float(I_bend[j]), E_MOD*float(I_bend[j]), G_MOD*float(J_tor[j])]))

I_rod_arr = np.zeros((N, 3, 3))
for j in range(N):
    rA  = float(rhoA[j])
    I11 = rA * WIDTH**2             / 12.0
    I22 = rA * float(thickness[j])**2 / 12.0
    I_rod_arr[j] = np.diag(np.array([I11, I22, I11 + I22]))

# Precomputed diagonal inverse — used by rod_mechanics.py to replace
# N individual linalg.solve calls with a single batched einsum.
# Safe because I_rod_arr is strictly diagonal with positive entries.
I_rod_inv = np.zeros((N, 3, 3))
for j in range(N):
    d = np.array([float(I_rod_arr[j, 0, 0]),
                  float(I_rod_arr[j, 1, 1]),
                  float(I_rod_arr[j, 2, 2])])
    I_rod_inv[j] = np.diag(1.0 / d)

# ─────────────────────────────────────────────────────────────────────────────
# Rigid body
# ─────────────────────────────────────────────────────────────────────────────
MB = 0.500
BX = 0.1255
BY = 0.0855
BZ = 0.034

IB     = MB/12.0 * np.diag([BY**2+BZ**2, BX**2+BZ**2, BX**2+BY**2])
IB_INV = np.linalg.inv(IB)

hx    = BX / 2.0
hy    = BY / 2.0
hz    = BZ / 2.0
z_off = -hz + THICK_FIXED / 2.0    # leg flush with cuboid bottom face

P_ATTACH = np.array([
    [ hx,  hy, z_off],
    [-hx,  hy, z_off],
    [-hx, -hy, z_off],
    [ hx, -hy, z_off],
])

SPLAY_DEG = np.array([45.0, 135.0, -135.0, -45.0])
SPLAY_RAD = np.deg2rad(SPLAY_DEG)

# ─────────────────────────────────────────────────────────────────────────────
# R_i0: d2=[sa,-ca,0] → d1=d2×d3=[0,0,+1] UPWARD  (Bug 11)
# ─────────────────────────────────────────────────────────────────────────────
def make_Ri0(alpha):
    ca, sa = np.cos(alpha), np.sin(alpha)
    d3 = np.array([ca,  sa, 0.0])
    d2 = np.array([sa, -ca, 0.0])
    d1 = np.cross(d2, d3);  d1 /= np.linalg.norm(d1)
    return np.column_stack([d1, d2, d3])

R_I0 = [make_Ri0(a) for a in SPLAY_RAD]

# ─────────────────────────────────────────────────────────────────────────────
# Initial rod shape — cubic profile (rod36.py coefficients)
#   z(s) = a*s³ + b*s² + d  with a=6.822, b=-1.844, d=0.02175 (rod36.py)
#   Derived from BCs: z(0)=Z0_FIXED, z'(0)=0, z(L)=0, z'(L)=0
# ─────────────────────────────────────────────────────────────────────────────
Z_BODY_INIT = 0.032      # was 0.0455 — see comment below
# Root cause of body-not-rising bug:
#   z_off = -hz + THICK_FIXED/2 = -0.01025 m
#   We need Z0_FIXED = 0.02175 m  (rod36.py: d=0.02175)
#   → Z_BODY_INIT = Z0_FIXED - z_off = 0.02175 + 0.01025 = 0.032 m
#
#   Old value 0.0455 m gave Z0_FIXED = 0.03525 m — a 62% taller arch than
#   rod36.py. The natural equilibrium of that arch was ~17 mm, so the body
#   started 28 mm ABOVE equilibrium and fell the entire 20 N tendon hold.
Z0_FIXED    = Z_BODY_INIT + z_off   # = 0.032 - 0.01025 = 0.02175 m ✓

A_CUB = 2.0 * Z0_FIXED / L**3
B_CUB = -3.0 * Z0_FIXED / L**2

def cubic_z(sv):  return A_CUB*sv**3 + B_CUB*sv**2 + Z0_FIXED
def cubic_dz(sv): return 3.0*A_CUB*sv**2 + 2.0*B_CUB*sv

def _log_SO3(R):
    cos_theta = np.clip(0.5*(np.trace(R)-1.0), -1.0, 1.0)
    theta     = np.arccos(cos_theta)
    if theta < 1.0e-8: return np.zeros(3)
    S = (R - R.T) / (2.0*np.sin(theta))
    return theta * np.array([S[2,1], S[0,2], S[1,0]])

def _build_initial_rod(i):
    alpha  = SPLAY_RAD[i]
    ca, sa = np.cos(alpha), np.sin(alpha)
    d2_rod = np.array([sa, -ca, 0.0])
    clamp_w    = P_ATTACH[i].copy();  clamp_w[2] += Z_BODY_INIT
    z_s, dz_s  = cubic_z(s), cubic_dz(s)
    r_rod = np.zeros((N, 3));  R_rod = np.zeros((N, 3, 3))
    for j in range(N):
        r_rod[j] = clamp_w + np.array([s[j]*ca, s[j]*sa, z_s[j]-Z0_FIXED])
        L_j      = np.sqrt(1.0 + dz_s[j]**2)
        d3_j     = np.array([ca/L_j, sa/L_j, dz_s[j]/L_j])
        d1_j     = np.cross(d2_rod, d3_j);  d1_j /= np.linalg.norm(d1_j)
        R_rod[j] = np.column_stack([d1_j, d2_rod, d3_j])
    return r_rod, R_rod

def _compute_ref_strains(r_rod, R_rod):
    u0 = np.zeros((N-1, 3));  k0 = np.zeros((N-1, 3))
    for j in range(N-1):
        u0[j] = R_rod[j].T @ (r_rod[j+1] - r_rod[j]) / ds
        k0[j] = _log_SO3(R_rod[j].T @ R_rod[j+1]) / ds
    return u0, k0

r0_all = np.zeros((4, N, 3));   R0_all = np.zeros((4, N, 3, 3))
u0_all = np.zeros((4, N-1, 3)); k0_all = np.zeros((4, N-1, 3))
for _i in range(4):
    r0_all[_i], R0_all[_i] = _build_initial_rod(_i)
    u0_all[_i], k0_all[_i] = _compute_ref_strains(r0_all[_i], R0_all[_i])

# ─────────────────────────────────────────────────────────────────────────────
# Shape-restoring spring reference positions  (body-relative frame)
#
#   r0_local_all[i][j] = R_I0[i].T @ (r0_all[i][j] − x0 − P_ATTACH[i])
#
#   Stores each node's reference position in the leg-attached frame so the
#   spring target tracks body motion:
#     r_desired[j] = x + Rb @ P_ATTACH[i] + Rb @ R_I0[i] @ r0_local[i][j]
#
#   Restoring force per node: F = −K_RESTORE × (r_actual − r_desired)
#   Equivalent to rod36.py's:  Fx[j] -= 6e3*(x[j]-x0[j])
#                              Fz[j] -= 6e3*(z[j]-z0[j])
#   but body-motion aware.
# ─────────────────────────────────────────────────────────────────────────────
_x0_world = np.array([0.0, 0.0, Z_BODY_INIT])   # initial body CoM
r0_local_all = np.zeros((4, N, 3))
for _i in range(4):
    for _j in range(N):
        _dr = r0_all[_i, _j] - _x0_world - P_ATTACH[_i]
        r0_local_all[_i, _j] = R_I0[_i].T @ _dr

K_RESTORE = 300.0   # [N/m] per node — resists upward curling from tendon d1 component

# ─────────────────────────────────────────────────────────────────────────────
# Simulation settings
# ─────────────────────────────────────────────────────────────────────────────
DT         = 1.0e-6
T_TOTAL    = 0.08    # extended from 0.25 s — allows full sit/stand/settle cycle
SAVE_EVERY = 200

G_VEC = np.array([0.0, 0.0, -9.81])

T_BASE      = 1.8    # baseline [N] — 4 × 1.8 × sin(135°) = 5.1 N ≈ body weight 4.9 N
                     #   → tips barely touch ground, no free-fall
T_EXTRA     = 15    # extra for standing [N]  → peak = 1.8 + 3.2 = 5.0 N
                    
TENDON_FMAX = T_BASE + T_EXTRA   # = 5.0 N

TENDON_ANGLE  = float(np.deg2rad(135.0))   # plain float — used with math.cos
R0_TEN        = 0.04                       # rod36.py: R0 = 0.04
TENDON_NODES  = int(2 * N / 3)             # rod36.py: tension_length = int(2*N/3)

# Schedule timings
T_RAMP_BASE = 0.020   # 0 → T_BASE  ramp ends [s]   (20 ms settling)
T_RAMP_END  = 0.035   # T_BASE → 20N ramp ends [s]   (15 ms rise)
T_HOLD_END  = 0.060   # hold at 20N ends [s]          (25 ms hold)
T_DN_END    = 0.080   # 20N → 0 ramp ends [s]         (20 ms sit)

# Aliases used by main.py
T_SETTLE   = T_RAMP_BASE
T_REL_END  = T_DN_END

def tendon_magnitude(t: float) -> float:
    if t <= 0.0:
        return 0.0
    elif t < T_RAMP_BASE:
        tau = t / T_RAMP_BASE
        return T_BASE * 0.5 * (1.0 - np.cos(np.pi * tau))
    elif t < T_RAMP_END:
        tau = (t - T_RAMP_BASE) / (T_RAMP_END - T_RAMP_BASE)
        return T_BASE + T_EXTRA * 0.5 * (1.0 - np.cos(np.pi * tau))
    elif t < T_HOLD_END:
        return TENDON_FMAX
    elif t < T_DN_END:
        tau = (t - T_HOLD_END) / (T_DN_END - T_HOLD_END)
        return TENDON_FMAX * 0.5 * (1.0 + np.cos(np.pi * tau))
    else:
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Ground contact
#   rod36.py: mu = 0.1 (tip), mu_legs = [0.6, 0.1, 0.1, 0.6]
#   Here: uniform mu = 0.4 (penalty method; rod36.py used sign-velocity friction)
# ─────────────────────────────────────────────────────────────────────────────
MU_CONTACT = 0.4
GROUND_Z   = 0.0
K_CONTACT  = 100000.0   # [N/m]  matches reference contact_forces(k=1e5)
D_CONTACT  = 10.0       # [N·s/m]  near critically damped

# ─────────────────────────────────────────────────────────────────────────────
# Material damping (Rayleigh-type)
#   rod36.py used c=0.1 N·s/m scalar damping + vx/vz *= 0.95 per step.
#   Here: equivalent body-frame Rayleigh damping.
# ─────────────────────────────────────────────────────────────────────────────
C_LIN = 8.0    # [s⁻¹]  increased from 2.0 — damps post-actuation oscillation
C_ANG = 8.0    # [s⁻¹]  prevents body from spinning after landing