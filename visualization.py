import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from parameters import (
    N, L, ds,
    BX, BY, BZ, P_ATTACH,
    THICK_FIXED, THICK_TIP, WIDTH,
    TENDON_ANGLE, R0_TEN, TENDON_NODES,
    SPLAY_RAD, R_I0,
    SAVE_EVERY, DT,
)

LEG_COLOR    = '#1565C0'        
LEG_NAMES    = ['Leg 0 (+x+y)', 'Leg 1 (−x+y)', 'Leg 2 (−x−y)', 'Leg 3 (+x−y)']
BODY_COLOR   = '#FFC107'
BODY_EDGE    = '#E65100'
TENDON_COLOR = '#C62828'
COM_COLOR    = '#F44336'
PULLEY_COLOR = '#FF6F00'

# Per-leg plot colours for state plots (kept distinct for readability)
PLOT_COLORS  = ['#1565C0', '#2E7D32', '#BF360C', '#6A1B9A']

def _thick(j: int) -> float:
    return THICK_FIXED + (THICK_TIP - THICK_FIXED) * j / (N - 1)

def _rod_tube_faces(r: np.ndarray, R: np.ndarray) -> list:
    """
    Build quad faces for the tapered rod tube.
    Columns of R[j]: [d1 | d2 | d3]  where d1≈up, d2≈sideways, d3≈tangent.
    """
    faces    = []
    half_w   = WIDTH / 2.0

    def corners(j):
        d1 = R[j, :, 0];  d2 = R[j, :, 1];  c = r[j]
        t  = _thick(j) / 2.0
        return [c + t*d1 + half_w*d2,
                c + t*d1 - half_w*d2,
                c - t*d1 - half_w*d2,
                c - t*d1 + half_w*d2]

    for j in range(N - 1):
        cj, cj1 = corners(j), corners(j + 1)
        faces.append([cj[0], cj[1], cj1[1], cj1[0]])   # top
        faces.append([cj[1], cj[2], cj1[2], cj1[1]])   # right
        faces.append([cj[2], cj[3], cj1[3], cj1[2]])   # bottom
        faces.append([cj[3], cj[0], cj1[0], cj1[3]])   # left

    faces.append(corners(0))        # start cap
    faces.append(corners(N - 1))    # end cap
    return faces

def _cuboid_faces(x: np.ndarray, Rb: np.ndarray) -> list:
    hx, hy, hz = BX/2, BY/2, BZ/2
    offs = np.array([[ hx, hy, hz],[ hx,-hy, hz],[-hx,-hy, hz],[-hx, hy, hz],
                     [ hx, hy,-hz],[ hx,-hy,-hz],[-hx,-hy,-hz],[-hx, hy,-hz]])
    v = np.array([x + Rb @ o for o in offs])
    return [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],
            [v[0],v[1],v[5],v[4]],[v[1],v[2],v[6],v[5]],
            [v[2],v[3],v[7],v[6]],[v[3],v[0],v[4],v[7]]]

def _pulley_pos(r0: np.ndarray, R0: np.ndarray) -> np.ndarray:
    d1 = R0[:, 0]   
    d3 = R0[:, 2]   
    ca, sa = np.cos(TENDON_ANGLE), np.sin(TENDON_ANGLE)
    return r0 + R0_TEN * (ca * d3 + sa * d1)

def _tendon_line(r: np.ndarray, R: np.ndarray) -> tuple:
    pulley = _pulley_pos(r[0], R[0])

    xs = [pulley[0]]
    ys = [pulley[1]]
    zs = [pulley[2]]

    for j in range(TENDON_NODES):
        d1      = R[j, :, 0]                       
        t_half  = _thick(j) / 2.0
        pt      = r[j] - t_half * d1               
        xs.append(pt[0]);  ys.append(pt[1]);  zs.append(pt[2])

    return xs, ys, zs


def animate(hist: dict,
            interval_ms: int = 30,
            save_path: str = None) -> None:
    """
    3-D animation of body + 4 legs with correct pulley and tendon geometry.
    All legs rendered in the same blue colour (Viz-1).
    """
    times    = hist["time"]
    xs       = hist["x"]
    Rbs      = hist["Rb"]
    rod_rs   = hist["rod_rs"]    
    rod_Rs   = hist["rod_Rs"]    
    T_arr    = hist["tendon_mag"]
    n_frames = len(times)

    fig = plt.figure(figsize=(14, 9))
    ax  = fig.add_subplot(111, projection='3d')

    pad = L + 0.05
    ax.set_xlim(-pad, pad);  ax.set_ylim(-pad, pad);  ax.set_zlim(-0.05, 0.25)
    ax.set_xlabel('X [m]', labelpad=6)
    ax.set_ylabel('Y [m]', labelpad=6)
    ax.set_zlabel('Z [m]', labelpad=6)
    ax.set_title('3D Cosserat Quadruped — Sit / Stand', fontsize=13, pad=12)
    ax.view_init(elev=18, azim=-55)
    ax.grid(True, alpha=0.4)

    # Ground plane
    ax.plot_surface(np.array([[-pad,pad],[-pad,pad]]),
                    np.array([[-pad,-pad],[pad,pad]]),
                    np.zeros((2,2)),
                    alpha=0.08, color='gray', zorder=0)

    rod_patches = []
    for _ in range(4):
        pc = Poly3DCollection([], facecolor=LEG_COLOR,
                              edgecolor='none', alpha=0.55, zorder=2)
        ax.add_collection3d(pc)
        rod_patches.append(pc)

    body_patch = Poly3DCollection([], facecolor=BODY_COLOR,
                                  edgecolor=BODY_EDGE, alpha=0.70, zorder=3)
    ax.add_collection3d(body_patch)

    tendon_lines = [ax.plot([], [], [], '-',
                            color=TENDON_COLOR, lw=1.5, alpha=0.9, zorder=5)[0]
                    for _ in range(4)]
    pulley_markers = [ax.scatter([], [], [], color=PULLEY_COLOR, s=60,
                                 marker='o', zorder=6, depthshade=False)
                      for _ in range(4)]

    com_scatter = ax.scatter([], [], [], color=COM_COLOR, s=80,
                             zorder=6, depthshade=False)

    tip_markers = [ax.scatter([], [], [], color=LEG_COLOR, s=20,
                              marker='o', zorder=4, depthshade=False)
                   for _ in range(4)]

    time_txt   = ax.text2D(0.02, 0.97, '', transform=ax.transAxes, fontsize=10,
                           va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    height_txt = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10,
                           va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    tendon_txt = ax.text2D(0.02, 0.83, '', transform=ax.transAxes, fontsize=10,
                           va='top', color='darkred',
                           bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    # Single legend entry for all legs
    ax.plot([], [], color=LEG_COLOR, lw=3, label='Legs (×4)')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.8)

    # ── Update ────────────────────────────────────────────────────────────
    def update(frame):
        x_f  = xs[frame]
        Rb_f = Rbs[frame]

        for i in range(4):
            r_i = rod_rs[i, frame]      # (N, 3)
            R_i = rod_Rs[i, frame]      # (N, 3, 3)

            # Rod tube
            rod_patches[i].set_verts(_rod_tube_faces(r_i, R_i))

            # Tendon (Viz-3: bottom surface)
            tx, ty, tz = _tendon_line(r_i, R_i)
            tendon_lines[i].set_data(tx, ty)
            tendon_lines[i].set_3d_properties(tz)

            # Pulley (Viz-2: inside torso)
            p = _pulley_pos(r_i[0], R_i[0])
            pulley_markers[i]._offsets3d = ([p[0]], [p[1]], [p[2]])

            # Tip
            tip = r_i[-1]
            tip_markers[i]._offsets3d = ([tip[0]], [tip[1]], [tip[2]])

        body_patch.set_verts(_cuboid_faces(x_f, Rb_f))
        com_scatter._offsets3d = ([x_f[0]], [x_f[1]], [x_f[2]])

        time_txt.set_text(f't = {times[frame]*1000:.2f} ms')
        height_txt.set_text(f'CoM z = {x_f[2]*1000:.2f} mm')
        tendon_txt.set_text(f'Tendon = {T_arr[frame]:.2f} N')

        return (rod_patches + [body_patch] + tendon_lines +
                pulley_markers + tip_markers +
                [com_scatter, time_txt, height_txt, tendon_txt])

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=interval_ms, blit=False, repeat=True)

    if save_path:
        writer = (animation.FFMpegWriter(fps=30, bitrate=1800)
                  if save_path.endswith('.mp4')
                  else animation.PillowWriter(fps=20))
        ani.save(save_path, writer=writer)
        print(f"  Animation saved → {save_path}")
    else:
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# State plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_states(hist: dict) -> None:
    """10-panel time-series of all tracked states."""
    t        = hist["time"] * 1000          # ms
    x_com    = hist["x"]
    vb       = hist["vb"]
    wb       = hist["wb"]
    n0       = hist["n0"]
    m0       = hist["m0"]
    F_rod_b  = hist["F_rod_b"]
    T_rod_b  = hist["T_rod_b"]
    T_mag    = hist["tendon_mag"]
    z_height = hist["body_height"] * 1000
    Rbs      = hist["Rb"]

    def rot_to_euler(R_arr):
        F = R_arr.shape[0];  angles = np.zeros((F, 3))
        for k in range(F):
            R = R_arr[k]
            angles[k,0] = np.rad2deg(np.arctan2( R[2,1], R[2,2]))
            angles[k,1] = np.rad2deg(np.arctan2(-R[2,0], np.sqrt(R[2,1]**2+R[2,2]**2)))
            angles[k,2] = np.rad2deg(np.arctan2( R[1,0], R[0,0]))
        return angles
    euler = rot_to_euler(Rbs)

    fig, axes = plt.subplots(5, 2, figsize=(16, 22))
    fig.suptitle('3D Cosserat Quadruped — Complete State History',
                 fontsize=14, fontweight='bold', y=0.995)

    # Row 0
    ax = axes[0,0]
    for k, lbl, c in zip(range(3), ['x','y','z'], ['tab:blue','tab:green','tab:red']):
        ax.plot(t, x_com[:,k]*1000, label=f'{lbl} [mm]', color=c)
    ax.set_title('Body CoM Position');  ax.set_ylabel('Position [mm]')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.4)

    ax = axes[0,1]
    ax.plot(t, z_height, color='tab:red', lw=2)
    ax.axhline(z_height[0], ls='--', color='gray', lw=1, label='Initial height')
    ax.fill_between(t, z_height[0], z_height, alpha=0.2, color='tab:red')
    ax.set_title('Body Height (Sit / Stand)');  ax.set_ylabel('CoM z [mm]')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.4)

    # Row 1
    ax = axes[1,0]
    for k, lbl, c in zip(range(3), ['vb_x','vb_y','vb_z'],
                          ['tab:blue','tab:green','tab:red']):
        ax.plot(t, vb[:,k]*1000, label=f'{lbl} [mm/s]', color=c)
    ax.set_title('Body Frame Velocity  vb');  ax.set_ylabel('Velocity [mm/s]')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.4)

    ax = axes[1,1]
    for k, lbl, c in zip(range(3), ['ωb_x','ωb_y','ωb_z'],
                          ['tab:blue','tab:green','tab:red']):
        ax.plot(t, np.rad2deg(wb[:,k]), label=f'{lbl} [°/s]', color=c)
    ax.set_title('Body Angular Velocity  ωb');  ax.set_ylabel('[°/s]')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.4)

    # Row 2
    ax = axes[2,0]
    for i in range(4):
        ax.plot(t, np.linalg.norm(n0[i], axis=1),
                label=f'Leg {i}', color=PLOT_COLORS[i])
    ax.set_title('Clamp Force Magnitude  |n_i(0)|')
    ax.set_ylabel('[N]');  ax.legend(fontsize=8);  ax.grid(True, alpha=0.4)

    ax = axes[2,1]
    for i in range(4):
        ax.plot(t, np.linalg.norm(m0[i], axis=1),
                label=f'Leg {i}', color=PLOT_COLORS[i])
    ax.set_title('Clamp Moment Magnitude  |m_i(0)|')
    ax.set_ylabel('[N·m]');  ax.legend(fontsize=8);  ax.grid(True, alpha=0.4)

    # Row 3
    ax = axes[3,0]
    for k, lbl, c in zip(range(3), ['Fx','Fy','Fz'],
                          ['tab:blue','tab:green','tab:red']):
        ax.plot(t, F_rod_b[:,k], label=lbl, color=c)
    ax.set_title('Net Rod Force on Body  F_rod_b (body frame)')
    ax.set_ylabel('[N]');  ax.legend(fontsize=8);  ax.grid(True, alpha=0.4)

    ax = axes[3,1]
    for k, lbl, c in zip(range(3), ['Tx','Ty','Tz'],
                          ['tab:blue','tab:green','tab:red']):
        ax.plot(t, T_rod_b[:,k]*1000, label=lbl, color=c)
    ax.set_title('Net Rod Torque on Body  T_rod_b (body frame)')
    ax.set_ylabel('[mN·m]');  ax.legend(fontsize=8);  ax.grid(True, alpha=0.4)

    # Row 4
    ax = axes[4,0]
    ax.plot(t, T_mag, color='darkred', lw=2)
    ax.fill_between(t, T_mag, alpha=0.3, color='darkred')
    ax.set_title('Tendon Force Schedule')
    ax.set_ylabel('[N]');  ax.set_xlabel('Time [ms]');  ax.grid(True, alpha=0.4)

    ax = axes[4,1]
    for k, lbl, c in zip(range(3), ['Roll','Pitch','Yaw'],
                          ['tab:blue','tab:green','tab:orange']):
        ax.plot(t, euler[:,k], label=f'{lbl} [°]', color=c)
    ax.set_title('Body Orientation — Euler Angles (ZYX)')
    ax.set_ylabel('[°]');  ax.set_xlabel('Time [ms]')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.4)

    for fig_obj in [fig]:
        fig_obj.tight_layout()

    # Per-leg clamp reactions (second figure)
    fig2, axes2 = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    fig2.suptitle('Per-Leg Clamp Reactions  n_i(0) and m_i(0)',
                  fontsize=13, fontweight='bold')
    for i in range(4):
        ax_n, ax_m = axes2[i,0], axes2[i,1]
        for k, lbl, c in zip(range(3), ['d1','d2','d3'],
                              ['tab:blue','tab:green','tab:red']):
            ax_n.plot(t, n0[i,:,k], label=f'n_{lbl}', color=c)
            ax_m.plot(t, m0[i,:,k]*1000, label=f'm_{lbl}', color=c)
        ax_n.set_ylabel(f'Leg {i}\n[N]', fontsize=9)
        ax_m.set_ylabel('[mN·m]', fontsize=9)
        ax_n.grid(True, alpha=0.4);  ax_m.grid(True, alpha=0.4)
        ax_n.legend(fontsize=7);     ax_m.legend(fontsize=7)
        if i == 0:
            ax_n.set_title('Clamp Force  n_i(0)  [rod body frame]')
            ax_m.set_title('Clamp Moment m_i(0)  [rod body frame]')
        if i == 3:
            ax_n.set_xlabel('Time [ms]');  ax_m.set_xlabel('Time [ms]')

    fig2.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Rod shape debug view
# ─────────────────────────────────────────────────────────────────────────────

def plot_rod_shape(hist: dict, frame: int = -1) -> None:
    if frame == -1:
        frame = len(hist["time"]) - 1

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Rod shapes at t = {hist["time"][frame]*1000:.2f} ms')
    ax.set_xlabel('X [m]');  ax.set_ylabel('Y [m]');  ax.set_zlabel('Z [m]')

    for i in range(4):
        r  = hist["rod_rs"][i, frame]
        R  = hist["rod_Rs"][i, frame]
        ax.plot(r[:,0], r[:,1], r[:,2], '-o', markersize=3,
                color=LEG_COLOR, lw=2,
                label='Legs' if i == 0 else '_nolegend_')
        ax.scatter(*r[0],  color=PULLEY_COLOR, s=60, marker='s', zorder=5)
        ax.scatter(*r[-1], color=LEG_COLOR,    s=60, marker='^', zorder=5)

        # Pulley
        p = _pulley_pos(r[0], R[0])
        ax.scatter(*p, color=PULLEY_COLOR, s=80, marker='o', zorder=6)

        # Tendon
        tx, ty, tz = _tendon_line(r, R)
        ax.plot(tx, ty, tz, '-', color=TENDON_COLOR, lw=1.5, alpha=0.8)

    x_f  = hist["x"][frame]
    Rb_f = hist["Rb"][frame]
    ax.scatter(*x_f, color=COM_COLOR, s=100, marker='*', zorder=6, label='CoM')

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()