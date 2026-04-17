import cupy as _cp

try:
    _cp.zeros(1)  # basic allocation test

    if _cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("No CUDA device found.")

    xp = _cp
    USE_GPU = True
    print("[gpu_utils] Using GPU")

except Exception as e:
    raise RuntimeError(
        "[gpu_utils] CUDA / CuPy is REQUIRED but not available.\n"
        f"Reason: {e}"
    )


def to_host(arr):
    if USE_GPU and isinstance(arr, xp.ndarray):
        return arr.get()
    return _np.asarray(arr)


def to_device(arr):
    if USE_GPU:
        return xp.asarray(arr)
    return arr


def batch_skew(v):
    z = xp.zeros_like(v[..., 0])
    row0 = xp.stack([ z,        -v[..., 2],  v[..., 1]], axis=-1)
    row1 = xp.stack([ v[..., 2], z,         -v[..., 0]], axis=-1)
    row2 = xp.stack([-v[..., 1], v[..., 0],  z        ], axis=-1)
    return xp.stack([row0, row1, row2], axis=-2)   # (..., 3, 3)


def batch_log_SO3(R):
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


def batch_rodrigues(omega, h):
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

def batch_cayley(omega, h):
    psi   = 0.5 * h * omega                              # (..., 3)
    norm2 = xp.sum(psi ** 2, axis=-1)                    # (...,)
    denom = 1.0 + norm2                                   # (...,)

    K     = batch_skew(psi)                              # (..., 3, 3)
    KpKK  = K + xp.matmul(K, K)                         # (..., 3, 3)

    sh = psi.shape[:-1] + (3, 3)
    I_b = xp.broadcast_to(xp.eye(3, dtype=omega.dtype), sh)

    return I_b + (2.0 / denom)[..., None, None] * KpKK

def batch_project_SO3(R):
    RtR = xp.matmul(R.swapaxes(-1, -2), R)              # (..., 3, 3)
    sh  = R.shape
    I_b = xp.broadcast_to(xp.eye(3, dtype=R.dtype), sh)
    return xp.matmul(R, 1.5 * I_b - 0.5 * RtR)

def batch_vect_from_RtdR(R0, dR):
    RtdR = xp.einsum('...ki,...kj->...ij', R0, dR)      # R₀ᵀ dR
    return xp.stack([RtdR[..., 2, 1],
                     RtdR[..., 0, 2],
                     RtdR[..., 1, 0]], axis=-1)