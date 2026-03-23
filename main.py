

import argparse
import sys
import os
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='3D Cosserat Quadruped — Sit/Stand Simulation',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=__doc__
    )
    p.add_argument('--load',       action='store_true',
                   help='Load existing sim_history.npz instead of re-simulating')
    p.add_argument('--no-anim',    action='store_true',
                   help='Skip the 3D animation, show only state plots')
    p.add_argument('--save-anim',  action='store_true',
                   help='Save animation to quadruped.mp4 (requires ffmpeg)')
    p.add_argument('--shape-only', action='store_true',
                   help='Plot initial rod shapes and exit (no simulation)')
    p.add_argument('--no-plots',   action='store_true',
                   help='Skip state plots')
    p.add_argument('--history',    type=str, default='sim_history.npz',
                   help='Path to history file (default: sim_history.npz)')
    p.add_argument('--dt',         type=float, default=None,
                   help='Override time step DT [s]  (e.g. 1e-5)')
    p.add_argument('--t-total',    type=float, default=None,
                   help='Override total simulation time [s]')
    p.add_argument('--save-every', type=int, default=None,
                   help='Override SAVE_EVERY (frames saved every N steps)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Runtime parameter overrides
# ─────────────────────────────────────────────────────────────────────────────

def apply_overrides(args):
    """Patch parameters module at runtime if CLI overrides given."""
    import parameters as P

    if args.dt is not None:
        P.DT = args.dt
        print(f"  [override] DT         = {P.DT:.2e} s")

    if args.t_total is not None:
        P.T_TOTAL = args.t_total
        print(f"  [override] T_TOTAL    = {P.T_TOTAL:.3f} s")

    if args.save_every is not None:
        P.SAVE_EVERY = args.save_every
        print(f"  [override] SAVE_EVERY = {P.SAVE_EVERY}")


# ─────────────────────────────────────────────────────────────────────────────
# Print system summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    from parameters import (
        N, L, ds, DT, T_TOTAL, SAVE_EVERY,
        MB, BX, BY, BZ,
        T_BASE, T_EXTRA, TENDON_FMAX,
        T_SETTLE, T_RAMP_END, T_HOLD_END, T_REL_END,
        MU_CONTACT, K_CONTACT,
        Z_BODY_INIT,
        E_MOD, RHO,
    )

    total_steps  = int(T_TOTAL / DT)
    n_frames     = total_steps // SAVE_EVERY + 1
    mem_est_mb   = (4 * n_frames * N * 3 * 8) / 1e6  # rod_rs alone

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      3D Cosserat Quadruped — Simulation Parameters       ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Rods          : 4 × N={N} nodes,  L={L*100:.1f} cm,  ds={ds*1000:.2f} mm   ║")
    print(f"║  Material      : E={E_MOD/1e6:.0f} MPa,  ρ={RHO:.0f} kg/m³              ║")
    print(f"║  Body          : {BX*100:.1f}×{BY*100:.1f}×{BZ*100:.1f} cm,  M={MB:.3f} kg         ║")
    print(f"║  Init height   : {Z_BODY_INIT*1000:.1f} mm                            ║")
    print(f"║  Integrator    : Geometric RK4 (Cayley+Rodrigues/SO(3)) ║")
    print(f"║  DT            : {DT:.2e} s                              ║")
    print(f"║  T_TOTAL       : {T_TOTAL:.3f} s                              ║")
    print(f"║  Steps         : {total_steps:,}                          ║")
    print(f"║  Saved frames  : {n_frames:,}  (every {SAVE_EVERY} steps)          ║")
    print(f"║  Approx memory : {mem_est_mb:.0f} MB (rod positions only)          ║")
    print(f"║  Tendon base   : {T_BASE:.1f} N  |  stand extra : {T_EXTRA:.1f} N  |  peak : {TENDON_FMAX:.1f} N  ║")
    print(f"║  Schedule      : settle {T_SETTLE*1000:.0f} ms → ramp {T_RAMP_END*1000:.0f} ms → "
          f"hold {T_HOLD_END*1000:.0f} ms → sit {T_REL_END*1000:.0f} ms  ║")
    print(f"║  Contact       : penalty K={K_CONTACT:.0e} N/m,  μ={MU_CONTACT:.2f}       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Shape-only mode  (initial geometry, no sim)
# ─────────────────────────────────────────────────────────────────────────────

def shape_only_mode():
    """Plot initial rod geometry and exit."""
    from simulation     import build_initial_state
    from visualization  import plot_rod_shape

    print("  Building initial state …")
    state = build_initial_state()

    # Wrap in a minimal history-compatible dict for plot_rod_shape
    from parameters import N
    hist = {
        "time"   : np.array([0.0]),
        "x"      : state["x"][np.newaxis, :],
        "Rb"     : state["Rb"][np.newaxis, :, :],
        "rod_rs" : np.array([r[np.newaxis] for r in state["rod_rs"]]),
        "rod_Rs" : np.array([R[np.newaxis] for R in state["rod_Rs"]]),
    }
    print("  Plotting initial rod shapes …")
    plot_rod_shape(hist, frame=0)
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Parameter overrides ───────────────────────────────────────────────
    apply_overrides(args)

    # ── Print config summary ──────────────────────────────────────────────
    print_summary()

    # ── Shape-only shortcut ───────────────────────────────────────────────
    if args.shape_only:
        shape_only_mode()

    # ── Load or run simulation ────────────────────────────────────────────
    from simulation import run_simulation, save_history, load_history

    if args.load:
        if not os.path.exists(args.history):
            print(f"  ERROR: history file '{args.history}' not found.")
            print("  Run without --load to generate it first.")
            sys.exit(1)
        print(f"  Loading history from '{args.history}' …")
        hist = load_history(args.history)
        print(f"  Loaded {len(hist['time'])} frames.")
    else:
        hist = run_simulation(verbose=True)
        save_history(hist, path=args.history)

    # ── Quick sanity checks ───────────────────────────────────────────────
    _sanity_check(hist)

    # ── 3-D Animation ─────────────────────────────────────────────────────
    if not args.no_anim:
        from visualization import animate

        save_path = 'quadruped.mp4' if args.save_anim else None

        if save_path:
            print(f"  Rendering animation → {save_path} …")
        else:
            print("  Launching 3-D animation …")
            print("  (close the window to continue to state plots)")

        animate(hist, interval_ms=30, save_path=save_path)

    # ── State plots ───────────────────────────────────────────────────────
    if not args.no_plots:
        from visualization import plot_states, plot_rod_shape
        print("  Plotting state histories …")
        plot_states(hist)

        # Rod shapes at 3 key moments: start, peak, end
        n_frames = len(hist["time"])
        key_frames = {
            'Initial (t=0)'     : 0,
            'Peak height'       : int(np.argmax(hist["body_height"])),
            'Final (t=T_TOTAL)' : n_frames - 1,
        }
        for label, frame_idx in key_frames.items():
            print(f"  Rod shape @ {label} …")
            plot_rod_shape(hist, frame=frame_idx)

    print("\n  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks on history
# ─────────────────────────────────────────────────────────────────────────────

def _sanity_check(hist: dict) -> None:
    """Print a quick health report on the simulation history."""
    print()
    print("  ── Sanity checks ──────────────────────────────────────")

    z     = hist["body_height"]
    z_min = z.min()
    z_max = z.max()
    z0    = z[0]

    # "Rise" = how far the body goes above its settled equilibrium.
    # We measure from the POST-SETTLING minimum (after the first fall),
    # not from z_arr[0] which is the stress-free initial overshoot height.
    # Find the minimum after t > T_RAMP_BASE (settle phase ends).
    from parameters import T_RAMP_BASE, DT, SAVE_EVERY
    settle_frame = int(T_RAMP_BASE / (DT * SAVE_EVERY))
    z_settled    = z[settle_frame:]
    z_eq         = z_settled.min()          # approximate equilibrium = min after settling
    rise         = z_settled.max() - z_eq   # how far body rose above equilibrium

    # Check 1: Did the body actually rise?
    if rise > 1e-3:
        print(f"  ✓  Body rose by {rise*1000:.2f} mm above equilibrium"
              f"  (eq≈{z_eq*1000:.1f} mm → peak≈{z_settled.max()*1000:.1f} mm)")
    else:
        print(f"  ✗  Body did NOT rise above equilibrium  (rise = {rise*1000:.3f} mm)")
        print("     Check tendon parameters or DT stability.")

    # Check 2: No NaN/Inf in body position
    if np.all(np.isfinite(hist["x"])):
        print("  ✓  Body CoM: no NaN/Inf detected")
    else:
        print("  ✗  Body CoM: NaN or Inf found — simulation may have diverged")

    # Check 3: No NaN in rod positions
    rod_ok = all(np.all(np.isfinite(hist["rod_rs"][i]))
                 for i in range(4))
    if rod_ok:
        print("  ✓  Rod positions: no NaN/Inf detected")
    else:
        print("  ✗  Rod positions: NaN or Inf found")

    # Check 4: SO(3) drift — check that R^T R ≈ I for body and rod 0 node 0
    Rb_last = hist["Rb"][-1]
    ortho_err_body = np.linalg.norm(Rb_last.T @ Rb_last - np.eye(3))
    R0_last  = hist["rod_Rs"][0, -1, 0]
    ortho_err_rod  = np.linalg.norm(R0_last.T @ R0_last - np.eye(3))
    tol = 1e-5
    sym = "✓" if ortho_err_body < tol else "✗"
    print(f"  {sym}  SO(3) drift body Rb: ‖R^T R - I‖ = {ortho_err_body:.2e}"
          f"  (tol {tol:.0e})")
    sym = "✓" if ortho_err_rod < tol else "✗"
    print(f"  {sym}  SO(3) drift rod0/node0: ‖R^T R - I‖ = {ortho_err_rod:.2e}"
          f"  (tol {tol:.0e})")

    # Check 5: Clamp constraint — r_i(0) should equal x + Rb p_i
    from parameters import P_ATTACH
    clamp_errs = []
    x_last  = hist["x"][-1]
    for i in range(4):
        r0_i  = hist["rod_rs"][i, -1, 0]
        r0_expected = x_last + Rb_last @ P_ATTACH[i]
        clamp_errs.append(np.linalg.norm(r0_i - r0_expected))
    max_clamp_err = max(clamp_errs)
    sym = "✓" if max_clamp_err < 1e-6 else "⚠"
    print(f"  {sym}  Clamp constraint: max |r_i(0) - x - Rb pi| = "
          f"{max_clamp_err:.2e} m")

    # Check 6: Ground penetration
    rod_rs_all = hist["rod_rs"]          # (4, F, N, 3)
    min_z_rod  = rod_rs_all[:, :, :, 2].min()
    sym = "✓" if min_z_rod >= -1e-4 else "⚠"
    print(f"  {sym}  Min node z = {min_z_rod*1000:.3f} mm  "
          f"(ground at 0 mm, tolerance 0.1 mm)")

    # Check 7: Energy proxy — body KE at end should be near 0 (at rest)
    vb_last  = hist["vb"][-1]
    from parameters import MB
    KE_body  = 0.5 * MB * np.dot(vb_last, vb_last)
    sym = "✓" if KE_body < 1e-3 else "ℹ"
    print(f"  {sym}  Body KE (final) = {KE_body*1000:.3f} mJ")

    print("  ───────────────────────────────────────────────────────")
    print()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()