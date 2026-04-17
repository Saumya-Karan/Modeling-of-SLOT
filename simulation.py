import numpy as np          
import time as _time

from gpu_utils import to_host        

from parameters import (
    N, DT, T_TOTAL, SAVE_EVERY,
    r0_all, R0_all,
    Z_BODY_INIT,
    tendon_magnitude,
)
from integrator      import pack_state, unpack_state, rk4_step
from rod_mechanics   import clamp_reactions
from body_mechanics  import aggregate_rod_loads

def build_initial_state():
    from gpu_utils import xp
    x0  = xp.array([0.0, 0.0, Z_BODY_INIT])
    Rb0 = xp.eye(3)
    vb0 = xp.zeros(3)
    wb0 = xp.zeros(3)

    rod_rs0  = [r0_all[i].copy() for i in range(4)]
    rod_Rs0  = [R0_all[i].copy() for i in range(4)]
    rod_vs0  = [xp.zeros((N, 3)) for _ in range(4)]
    rod_oms0 = [xp.zeros((N, 3)) for _ in range(4)]

    return pack_state(x0, Rb0, vb0, wb0,
                      rod_rs0, rod_Rs0, rod_vs0, rod_oms0)

def _allocate_history(n_frames):
    return {
        "time"        : np.zeros(n_frames),
        "x"           : np.zeros((n_frames, 3)),
        "Rb"          : np.zeros((n_frames, 3, 3)),
        "vb"          : np.zeros((n_frames, 3)),
        "wb"          : np.zeros((n_frames, 3)),
        "rod_rs"      : np.zeros((4, n_frames, N, 3)),
        "rod_Rs"      : np.zeros((4, n_frames, N, 3, 3)),
        "rod_vs"      : np.zeros((4, n_frames, N, 3)),
        "rod_oms"     : np.zeros((4, n_frames, N, 3)),
        "n0"          : np.zeros((4, n_frames, 3)),
        "m0"          : np.zeros((4, n_frames, 3)),
        "F_rod_b"     : np.zeros((n_frames, 3)),
        "T_rod_b"     : np.zeros((n_frames, 3)),
        "tendon_mag"  : np.zeros(n_frames),
        "body_height" : np.zeros(n_frames),
    }


def _record(hist, frame, t, state):
    """Save current (GPU) state to CPU history buffers."""
    (x, Rb, vb, wb,
     rod_rs, rod_Rs, rod_vs, rod_oms) = unpack_state(state)

    hist["time"][frame]        = float(t)
    hist["x"][frame]           = to_host(x)
    hist["Rb"][frame]          = to_host(Rb)
    hist["vb"][frame]          = to_host(vb)
    hist["wb"][frame]          = to_host(wb)
    hist["body_height"][frame] = float(to_host(x)[2])
    hist["tendon_mag"][frame]  = tendon_magnitude(t)

    for i in range(4):
        hist["rod_rs"][i, frame]  = to_host(rod_rs[i])
        hist["rod_Rs"][i, frame]  = to_host(rod_Rs[i])
        hist["rod_vs"][i, frame]  = to_host(rod_vs[i])
        hist["rod_oms"][i, frame] = to_host(rod_oms[i])

        ni0, mi0 = clamp_reactions(rod_rs[i], rod_Rs[i], i)
        hist["n0"][i, frame] = to_host(ni0)
        hist["m0"][i, frame] = to_host(mi0)

    F_rod_b, T_rod_b = aggregate_rod_loads(rod_rs, rod_Rs)
    hist["F_rod_b"][frame] = to_host(F_rod_b)
    hist["T_rod_b"][frame] = to_host(T_rod_b)


def run_simulation(verbose=True):
    total_steps = int(T_TOTAL / DT)
    n_frames    = total_steps // SAVE_EVERY + 1

    if verbose:
        from gpu_utils import USE_GPU
        device = "GPU (CuPy)" if USE_GPU else "CPU (NumPy vectorised)"
        print("=" * 60)
        print(f"  3D Cosserat Quadruped Simulation  [{device}]")
        print("=" * 60)
        print(f"  Total time   : {T_TOTAL:.3f} s")
        print(f"  Time step    : {DT:.2e} s")
        print(f"  Total steps  : {total_steps:,}")
        print(f"  Saved frames : {n_frames:,}  (every {SAVE_EVERY} steps)")
        print(f"  Nodes/rod    : {N}")
        print("-" * 60)

    state      = build_initial_state()
    hist       = _allocate_history(n_frames)
    frame      = 0
    t          = 0.0
    wall_start = _time.perf_counter()
    PRINT_EVERY = 5000

    _record(hist, frame, t, state);  frame += 1

    for step in range(1, total_steps + 1):
        t     = step * DT
        state = rk4_step(t - DT, state, DT)

        x_now_cpu = to_host(state["x"])
        if not np.all(np.isfinite(x_now_cpu)):
            print(f"\n  !! NaN/Inf at step {step} (t={t:.5f} s). Aborting.")
            for key in hist:
                val = hist[key]
                if isinstance(val, np.ndarray) and val.ndim >= 1:
                    hist[key] = val[:frame]
            break

        if step % SAVE_EVERY == 0 and frame < n_frames:
            _record(hist, frame, t, state);  frame += 1

        if verbose and step % PRINT_EVERY == 0:
            elapsed = _time.perf_counter() - wall_start
            frac    = step / total_steps
            eta     = elapsed / frac * (1.0 - frac) if frac > 0 else 0.0
            z_now   = float(x_now_cpu[2])
            T_now   = tendon_magnitude(t)
            rate    = step / elapsed if elapsed > 0 else 0.0
            print(f"  step {step:>7,}/{total_steps:,}"
                  f"  ({frac*100:5.1f}%)"
                  f"  t={t*1000:6.2f}ms"
                  f"  z={z_now*1000:6.2f}mm"
                  f"  T={T_now:5.1f}N"
                  f"  {rate:,.0f}steps/s"
                  f"  ETA {eta:5.0f}s")

    if verbose:
        elapsed = _time.perf_counter() - wall_start
        valid   = frame
        z_arr   = hist["body_height"][:valid]
        print("-" * 60)
        print(f"  Done in {elapsed:.1f} s  |  "
              f"avg {int(total_steps/elapsed):,} steps/s  |  "
              f"rise = {(z_arr.max()-z_arr[0])*1000:.2f} mm")
        print("=" * 60)

    # Trim
    for key in hist:
        val = hist[key]
        if isinstance(val, np.ndarray) and val.shape[0] == n_frames:
            hist[key] = val[:frame]
        elif isinstance(val, np.ndarray) and val.ndim > 1 and val.shape[1] == n_frames:
            idx = [slice(None)] * val.ndim;  idx[1] = slice(None, frame)
            hist[key] = val[tuple(idx)]
    return hist


def save_history(hist, path="sim_history.npz"):
    np.savez_compressed(path, **hist)
    print(f"  History saved → {path}")

def load_history(path="sim_history.npz"):
    return dict(np.load(path, allow_pickle=False))