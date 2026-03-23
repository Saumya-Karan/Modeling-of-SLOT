import numpy as _np

import numpy as _np

# ─────────────────────────────────────────────────────────────
# USER CONTROL (SET THIS MANUALLY)
# ─────────────────────────────────────────────────────────────
USE_GPU = False  # ← Set True only if CuPy is installed (pip install cupy-cuda12x)

# ─────────────────────────────────────────────────────────────
# Backend selection
# ─────────────────────────────────────────────────────────────
if USE_GPU:
    try:
        import cupy as _cp
        _cp.zeros(1)  # test
        xp = _cp
        print("  [gpu_utils] Using GPU (CuPy)")
    except Exception as e:
        print("  [gpu_utils] GPU requested but CuPy failed → fallback to CPU")
        print("   Reason:", e)
        xp = _np
        USE_GPU = False
else:
    xp = _np
    print("  [gpu_utils] Using CPU (NumPy)")


# ─────────────────────────────────────────────────────────────────────────────
# Transfer helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_host(arr):
    """Return a plain NumPy array (CPU), copying from GPU if needed."""
    if USE_GPU and isinstance(arr, xp.ndarray):
        return arr.get()
    return _np.asarray(arr)


def to_device(arr):
    """Transfer a NumPy array to the active device (GPU or no-op on CPU)."""
    if USE_GPU:
        return xp.asarray(arr)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Batch skew-symmetric matrix  (..., 3) → (..., 3, 3)
# ─────────────────────────────────────────────────────────────────────────────

def batch_skew(v):
    """
    Build skew-symmetric matrices for a batch of 3-vectors.

    v : (..., 3) → K : (..., 3, 3)  where K @ a = v × a
    """
    z = xp.zeros_like(v[..., 0])
    row0 = xp.stack([ z,        -v[..., 2],  v[..., 1]], axis=-1)
    row1 = xp.stack([ v[..., 2], z,         -v[..., 0]], axis=-1)
    row2 = xp.stack([-v[..., 1], v[..., 0],  z        ], axis=-1)
    return xp.stack([row0, row1, row2], axis=-2)   # (..., 3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Batch SO(3) log-map  (..., 3, 3) → (..., 3)
# ─────────────────────────────────────────────────────────────────────────────

def batch_log_SO3(R):
    """
    Exact Rodrigues inverse for a batch of rotation matrices.

    R     : (..., 3, 3)
    return: (..., 3)  axial vector ω such that expm(ω∧) = R
    """
    trace     = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = xp.clip(0.5 * (trace - 1.0), -1.0, 1.0)
    theta     = xp.arccos(cos_theta)                     # (...,)

    sin_theta = xp.sin(theta)
    # Avoid division by zero at θ≈0; result will be zeroed out anyway
    safe_sin  = xp.where(xp.abs(sin_theta) < 1.0e-10,
                         xp.ones_like(sin_theta) * 1.0e-10,
                         sin_theta)
    scale = theta / (2.0 * safe_sin)                     # (...,)

    omega = scale[..., None] * xp.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], axis=-1)                                           # (..., 3)

    # Zero out near-identity cases
    return xp.where(theta[..., None] < 1.0e-8,
                    xp.zeros_like(omega), omega)


# ─────────────────────────────────────────────────────────────────────────────
# Batch Rodrigues  (..., 3) × scalar → (..., 3, 3)
# ─────────────────────────────────────────────────────────────────────────────

def batch_rodrigues(omega, h):
    """
    expm(h · omega∧) for a batch of angular velocity vectors.

    omega : (..., 3)
    h     : scalar (same for all)
    return: (..., 3, 3)
    """
    ax    = omega * h                                     # (..., 3)
    theta = xp.linalg.norm(ax, axis=-1)                  # (...,)

    # Cap large rotations (safety)
    cap       = 1.0e2
    too_large = theta > cap
    ax        = xp.where(too_large[..., None],
                         ax / theta[..., None] * cap, ax)
    theta     = xp.where(too_large, cap * xp.ones_like(theta), theta)

    safe_th  = xp.where(theta < 1.0e-10,
                        xp.ones_like(theta) * 1.0e-10, theta)
    ax_hat   = ax / safe_th[..., None]                   # (..., 3) unit axes
    K        = batch_skew(ax_hat)                        # (..., 3, 3)

    sin_t = xp.sin(theta)[..., None, None]               # (..., 1, 1)
    cos_t = (1.0 - xp.cos(theta))[..., None, None]

    sh = ax_hat.shape[:-1] + (3, 3)
    I_b = xp.broadcast_to(xp.eye(3, dtype=omega.dtype), sh)

    R = I_b + sin_t * K + cos_t * xp.matmul(K, K)       # (..., 3, 3)

    # Near-zero: return identity
    return xp.where(theta[..., None, None] < 1.0e-10, I_b, R)


# ─────────────────────────────────────────────────────────────────────────────
# Batch Cayley map  (..., 3) × scalar → (..., 3, 3)
#   Cheap 2nd-order approximation to Rodrigues, used for intermediate stages
# ─────────────────────────────────────────────────────────────────────────────

def batch_cayley(omega, h):
    """
    Cayley map  ≈  expm(h · omega∧)  for a batch of angular velocities.

    omega : (..., 3)
    h     : scalar
    return: (..., 3, 3)
    """
    psi   = 0.5 * h * omega                              # (..., 3)
    norm2 = xp.sum(psi ** 2, axis=-1)                    # (...,)
    denom = 1.0 + norm2                                   # (...,)

    K     = batch_skew(psi)                              # (..., 3, 3)
    KpKK  = K + xp.matmul(K, K)                         # (..., 3, 3)

    sh = psi.shape[:-1] + (3, 3)
    I_b = xp.broadcast_to(xp.eye(3, dtype=omega.dtype), sh)

    return I_b + (2.0 / denom)[..., None, None] * KpKK


# ─────────────────────────────────────────────────────────────────────────────
# Batch project onto SO(3)  (..., 3, 3) → (..., 3, 3)
#   Uses one Newton step: R ← R @ (1.5 I − 0.5 Rᵀ R)
#   Much cheaper than SVD; works well when R is already near SO(3)
#   (which is always the case after a small integration step).
# ─────────────────────────────────────────────────────────────────────────────

def batch_project_SO3(R):
    """
    Newton-step re-orthogonalisation: R ← R (1.5 I − 0.5 Rᵀ R).

    One step reduces the error from O(ε) to O(ε²).
    No SVD — ≈ 10× cheaper per call.

    R : (..., 3, 3)
    """
    RtR = xp.matmul(R.swapaxes(-1, -2), R)              # (..., 3, 3)
    sh  = R.shape
    I_b = xp.broadcast_to(xp.eye(3, dtype=R.dtype), sh)
    return xp.matmul(R, 1.5 * I_b - 0.5 * RtR)


# ─────────────────────────────────────────────────────────────────────────────
# Extract body-frame ω from (R₀, Ṙ) pair — vectorised over batch dim
#   ω = vect(R₀ᵀ Ṙ)  →  [R₀ᵀ Ṙ]_{2,1}, [R₀ᵀ Ṙ]_{0,2}, [R₀ᵀ Ṙ]_{1,0}
# ─────────────────────────────────────────────────────────────────────────────

def batch_vect_from_RtdR(R0, dR):
    """
    R0 : (..., 3, 3)   current rotation
    dR : (..., 3, 3)   time/stage derivative  Ṙ = R skew(ω)
    return: (..., 3)   body-frame angular velocity ω
    """
    RtdR = xp.einsum('...ki,...kj->...ij', R0, dR)      # R₀ᵀ dR
    return xp.stack([RtdR[..., 2, 1],
                     RtdR[..., 0, 2],
                     RtdR[..., 1, 0]], axis=-1)