"""Front-view schematic of the waist whippletree: the core's roll on the hip box slaved 1:1 to the mean of the two
leg roll angles. Dimensions from the round-7 pantograph body (ROLL_PIVOT_Y 12 mm, roll bracket arm 38 mm)."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

PIV, ARM = 1.2, 3.8            # cm: each leg's roll pivot from the hip centre, the bracket arm
Z_BR = -2.0                    # bracket bar height (hip axis z = 0)
Z_BEAM = 6.0                   # rocker height
Z_CORE = 3.5                   # core hinge (top of the hip box)
C_ARM = 3.8                    # core crank

def draw(ax, thL, thR, title, show_dims):
    d = math.radians
    th_c = (thL + thR) / 2
    # hip box
    ax.add_patch(Rectangle((-4.2, -3.5), 8.4, 7.0, fc="#e6e6e6", ec="#777", lw=1.2, zorder=0))
    ax.text(0, -3.05, "hip box (pelvis): roll servos, leg pivots, hip IMU", ha="center", fontsize=7.5, color="#555")
    # core, rotated about its hinge by th_c
    c, s = math.cos(d(th_c)), math.sin(d(th_c))
    rot = lambda x, z: (x * c - (z - Z_CORE) * s, Z_CORE + x * s + (z - Z_CORE) * c)
    core = [rot(x, z) for x, z in ((-3.8, Z_CORE + 0.4), (3.8, Z_CORE + 0.4), (3.8, Z_CORE + 14.0), (-3.8, Z_CORE + 14.0))]
    ax.add_patch(plt.Polygon(core, closed=True, fc="#f3d9d9", ec="#b55", lw=1.2, alpha=0.9, zorder=1))
    cx, cz = rot(0, Z_CORE + 10.0)
    ax.text(cx, cz, "core\nbattery, electronics,\nfans, core IMU", ha="center", va="center", fontsize=8, color="#933")
    ax.add_patch(Circle((0, Z_CORE), 0.35, fc="white", ec="k", lw=1.5, zorder=5))
    ax.annotate("waist hinge\n(fore-aft axis)", (0, Z_CORE), (-7.6, Z_CORE + 1.0), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#444"))
    # core crank: 3.8 cm toward +y (left), rotates with the core
    tip_c = rot(C_ARM, Z_CORE)
    ax.plot([0, tip_c[0]], [Z_CORE, tip_c[1]], color="#b55", lw=4, solid_capstyle="round", zorder=4)
    # legs: left (+y, drawn on the left as seen from the front? we draw robot's left on the viewer's right)
    # convention here: x axis of the plot = robot +y (robot's left) to the RIGHT of the page
    def leg(pivot_y, th, pick_dir, label):
        cth, sth = math.cos(d(th)), math.sin(d(th))
        r = lambda dy, dz: (pivot_y + dy * cth - dz * sth, dy * sth + dz * cth)   # rotate about the pivot (z=0 at Z_BR shifted)
        # bracket bar from pivot outboard (sign of pivot_y) to the arm length, at Z_BR
        out = math.copysign(ARM, pivot_y)
        p0 = (pivot_y, Z_BR)
        b_end = (p0[0] + out * cth, Z_BR + out * sth)
        ax.plot([p0[0], b_end[0]], [p0[1], b_end[1]], color="#2b6cb0", lw=5, solid_capstyle="round", zorder=3)
        # leg stub down from the bracket
        lg = (p0[0] + (out * 0.6) * cth + 4.5 * sth, Z_BR + (out * 0.6) * sth - 4.5 * cth)
        ax.plot([p0[0] + out * 0.6 * cth, lg[0]], [Z_BR + out * 0.6 * sth, lg[1]], color="#2b6cb0", lw=3, ls=(0, (3, 2)), zorder=2)
        ax.text(lg[0], lg[1] - 0.6, label, ha="center", va="top", fontsize=7.5, color="#2b6cb0")
        ax.add_patch(Circle(p0, 0.3, fc="white", ec="#2b6cb0", lw=1.5, zorder=6))
        # pickup: on the bracket, ARM from the pivot, toward +y (robot's left) for both legs
        pick = (p0[0] + ARM * cth, Z_BR + ARM * sth)
        if pick_dir == "inboard":
            ax.plot([p0[0], pick[0]], [p0[1], pick[1]], color="#2f855a", lw=3.5, solid_capstyle="round", zorder=3)
        ax.add_patch(Circle(pick, 0.28, fc="#2f855a", ec="k", lw=0.8, zorder=7))
        return pick
    pL = leg(+PIV, thL, "outboard", "left leg")
    pR = leg(-PIV, thR, "inboard", "right leg")
    # rocker: ends straight above the pickups (pushrods vertical), centre over the crank tip
    eL, eR = (pL[0], Z_BEAM + (pL[1] - Z_BR)), (pR[0], Z_BEAM + (pR[1] - Z_BR))
    for p, e in ((pL, eL), (pR, eR)):
        ax.plot([p[0], e[0]], [p[1], e[1]], color="#555", lw=1.6, zorder=2)
    ax.plot([eL[0], eR[0]], [eL[1], eR[1]], color="#d69e2e", lw=6, solid_capstyle="round", zorder=4)
    mid = ((eL[0] + eR[0]) / 2, (eL[1] + eR[1]) / 2)
    ax.add_patch(Circle(mid, 0.33, fc="white", ec="#b7791f", lw=1.6, zorder=8))
    # centre link from the rocker's middle down to the core crank tip
    ax.plot([mid[0], tip_c[0]], [mid[1], tip_c[1]], color="#555", lw=1.6, zorder=2)
    for q in (eL, eR, tip_c):
        ax.add_patch(Circle(q, 0.2, fc="#333", ec="k", zorder=8))
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-9, 10.5); ax.set_ylim(-9, 19); ax.set_aspect("equal"); ax.axis("off")
    if show_dims:
        ax.annotate("rocker (whippletree):\nfloats on its 3 pins,\nits centre moves the\nmean of its two ends", mid, (6.0, 12.5), fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color="#b7791f"))
        ax.annotate("pushrods (vertical)", ((pL[0] + eL[0]) / 2, (pL[1] + eL[1]) / 2), (6.3, 3.6), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#555"))
        ax.annotate("left bracket end:\npickup 3.8 cm\nfrom its pivot", pL, (6.0, -1.0), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#2f855a"))
        ax.annotate("right pickup arm 3.8 cm,\npointing INBOARD (so it\nmoves like the left one)", pR, (-8.8, -6.2), fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color="#2f855a"))
        ax.annotate("core crank 3.8 cm\n(= pickup radius -> 1:1)", tip_c, (5.8, 7.6), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#b55"))
        ax.annotate("leg roll pivots,\n1.2 cm either side\nof the hip centre", (PIV, Z_BR), (-8.8, -1.6), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#2b6cb0"))
    else:
        ax.text(0.7, -8.0, f"left leg {thL:+.0f} deg, right leg {thR:+.0f} deg  ->  core turns {th_c:+.0f} deg on the hip box", ha="center", fontsize=8.5)

fig, axs = plt.subplots(1, 2, figsize=(14, 8.4))
draw(axs[0], 0, 0, "Waist whippletree, front view (robot's left to the right of the page), neutral", True)
draw(axs[1], 16, 0, "Left leg rolled 16 deg relative to the hip box, right leg 0", False)
fig.text(0.5, 0.035, "core roll on hip box = (left leg roll + right leg roll) / 2      (each leg's roll measured on its own pivot, same sense; ratio 1:1 = the round-1j walker)\n"
         "Every pickup and the crank are 3.8 cm (the roll bracket's arm), so the rocker's centre sits over the crank tip and every link is vertical. Rocker length 2.4 cm "
         "between its end pins.\nNo spring, no damper: the core's angle is fixed by the two leg servos at all times (fan torque, flight). The two pickup arms sit in different fore-aft planes "
         "so the right arm clears the left pivot.", ha="center", fontsize=8.3)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.12, wspace=0.02)
out = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "waist_whippletree.png")
fig.savefig(out, dpi=130)
print("wrote", out)
