import math as _math               

from gpu_utils import xp as np     

# ─────────────────────────────────────────────────────────────────────────────
# Rod discretisation
# ─────────────────────────────────────────────────────────────────────────────
L  = 0.180
N  = 21
ds = L / (N - 1)
s  = np.linspace(0.0, L, N)

# ─────────────────────────────────────────────────────────────────────────────
# Cross-section 
# ─────────────────────────────────────────────────────────────────────────────
THICK_FIXED = 0.0135
THICK_TIP   = 0.0035
WIDTH       = 0.020           

thickness = THICK_FIXED + (THICK_TIP - THICK_FIXED) * np.linspace(0.0, 1.0, N)
A_cs      = thickness * WIDTH                      
I_bend    = WIDTH * thickness**3 / 12.0            
J_tor     = WIDTH * thickness**3 / 3.0

# ─────────────────────────────────────────────────────────────────────────────
# Material properties — rod36.py values
#   E_default = 10e6 Pa (10 MPa flexible TPU)
#   rho = 1200 kg/m³
# ─────────────────────────────────────────────────────────────────────────────
E_MOD = 10.0e6          
NU    = 0.3
G_MOD = E_MOD / (2.0 * (1.0 + NU))
RHO   = 1200.0          

rhoA = RHO * A_cs

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
    I11 = rA * WIDTH**2               / 12.0
    I22 = rA * float(thickness[j])**2 / 12.0
    I_rod_arr[j] = np.diag(np.array([I11, I22, I11 + I22]))


I_rod_inv = np.zeros((N, 3, 3))
for j in range(N):
    d = np.array([float(I_rod_arr[j, 0, 0]),
                  float(I_rod_arr[j, 1, 1]),
                  float(I_rod_arr[j, 2, 2])])
    I_rod_inv[j] = np.diag(1.0 / d)


MB = 0.500
BX = 0.1255
BY = 0.0855
BZ = 0.034

IB     = MB/12.0 * np.diag([BY**2+BZ**2, BX**2+BZ**2, BX**2+BY**2])
IB_INV = np.linalg.inv(IB)

hx    = BX / 2.0
hy    = BY / 2.0
hz    = BZ / 2.0
z_off = -hz + THICK_FIXED / 2.0    

P_ATTACH = np.array([
    [ hx,  hy, z_off],
    [-hx,  hy, z_off],
    [-hx, -hy, z_off],
    [ hx, -hy, z_off],
])

SPLAY_DEG = np.array([45.0, 135.0, -135.0, -45.0])
SPLAY_RAD = np.deg2rad(SPLAY_DEG)


def make_Ri0(alpha):
    a  = float(alpha)
    ca = _math.cos(a)
    sa = _math.sin(a)
    d3 = np.array([ca,  sa, 0.0])
    d2 = np.array([sa, -ca, 0.0])
    d1 = np.cross(d2, d3);  d1 /= np.linalg.norm(d1)
    return np.column_stack([d1, d2, d3])

R_I0 = [make_Ri0(a) for a in SPLAY_RAD]


Z_BODY_INIT = 0.032      

Z0_FIXED    = Z_BODY_INIT + z_off   

A_CUB = 2.0 * Z0_FIXED / L**3
B_CUB = -3.0 * Z0_FIXED / L**2

def cubic_z(sv):  return A_CUB*sv**3 + B_CUB*sv**2 + Z0_FIXED
def cubic_dz(sv): return 3.0*A_CUB*sv**2 + 2.0*B_CUB*sv


def _log_SO3(R):
    
    cos_theta = _math.acos(max(-1.0, min(1.0,
                    float(0.5 * (float(np.trace(R)) - 1.0)))))
    theta = cos_theta                          
    if theta < 1.0e-8:
        return np.zeros(3)
    sin_theta = _math.sin(theta)
    S = (R - R.T) / (2.0 * sin_theta)
    return theta * np.array([float(S[2,1]), float(S[0,2]), float(S[1,0])])


def _build_initial_rod(i):
    
    alpha  = float(SPLAY_RAD[i])
    ca     = _math.cos(alpha)
    sa     = _math.sin(alpha)
    d2_rod = np.array([sa, -ca, 0.0])         

    clamp_w    = P_ATTACH[i].copy();  clamp_w[2] += Z_BODY_INIT
    z_s, dz_s  = cubic_z(s), cubic_dz(s)       
    r_rod = np.zeros((N, 3));  R_rod = np.zeros((N, 3, 3))
    for j in range(N):
        sj   = float(s[j])
        zsj  = float(z_s[j])
        dzsj = float(dz_s[j])
        
        r_rod[j] = clamp_w + np.array([sj*ca, sj*sa, zsj - Z0_FIXED])
        L_j      = _math.sqrt(1.0 + dzsj**2)
        d3_j     = np.array([ca/L_j, sa/L_j, dzsj/L_j])
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


DT         = 5.0e-6       
T_TOTAL    = 2.00         
SAVE_EVERY = 200

G_VEC = np.array([0.0, 0.0, -9.81])


T_BASE      = 3.0          

TENDON_FMAX = 3.6         

T_EXTRA     = TENDON_FMAX - T_BASE

TENDON_ANGLE  = _math.radians(135.0) 
R0_TEN        = 0.04                 
TENDON_NODES  = int(2 * N / 3)       


T_RAMP_BASE = 0.000                    
T_RAMP_END  = T_RAMP_BASE + 0.960     
T_HOLD_END  = T_RAMP_END               
T_DN_END    = T_HOLD_END  + 0.600     

# Aliases used by main.py
T_SETTLE   = T_RAMP_BASE
T_REL_END  = T_DN_END


def tendon_magnitude(t: float) -> float:

    if t <= 0.0:
        return float(T_BASE)
    elif t < T_RAMP_END:
        tau = t / T_RAMP_END
        return T_BASE + T_EXTRA * 0.5 * (1.0 - _math.cos(_math.pi * tau))
    elif t < T_HOLD_END:
        return float(TENDON_FMAX)
    elif t < T_DN_END:
        tau = (t - T_HOLD_END) / (T_DN_END - T_HOLD_END)
        return T_BASE + T_EXTRA * 0.5 * (1.0 + _math.cos(_math.pi * tau))
    else:
        return float(T_BASE)   


MU_CONTACT = 0.4
GROUND_Z   = 0.0
K_CONTACT  = 100000.0   
D_CONTACT  = 10.0       


C_LIN = 40.0   
C_ANG = 40.0   