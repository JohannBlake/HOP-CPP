"""
poc_net_sim.py  ─  Net Deformation POC
=======================================
69×69 flexible net with n ball(s) dropped at user-specified (x, y) ∈ [0,1]².

Physical heuristic
------------------
Each ball exerts a Gaussian attraction on surrounding net nodes.
  • heaviness  – ball weight  (larger → deeper / wider deformation)
  • tension    – net stiffness (larger → stiffer, smaller deformation)
The four corner nodes are always pinned (zero displacement) via a bilinear
correction applied after the free-field solve.

Key API
-------
    Xd, Yd = compute_deformation(ball_positions, heaviness, tension)
    visualize_single(ball_positions, heaviness, tension)
    visualize_grid(heaviness, tension, n=10)

ball_positions : (x, y) or list-of-(x, y) in [0, 1]
Returns Xd, Yd : (69, 69) arrays of deformed coordinates in [0, 1]
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection
from scipy.ndimage import map_coordinates

# ─────────────────────────────────── constants ────────────────────────────────
GRID_SIZE  = 130
DARK_BG    = "#0d1117"
NET_COLOR  = "#3a9ad9"
BALL_COLOR = "#ff4040"


# ─────────────────────────────────── core ─────────────────────────────────────

def make_grid(N: int = GRID_SIZE):
    """Uniform (X, Y) meshgrid in [0, 1]².  X[i,j] = x[j],  Y[i,j] = y[i]."""
    c = np.linspace(0.0, 1.0, N)
    return np.meshgrid(c, c)


def compute_deformation(
    ball_positions,
    heaviness: float = 1.0,
    tension:   float = 1.0,
    N:         int   = GRID_SIZE,
):
    """
    Deformed mesh node coordinates for n balls dropped on the net.

    Parameters
    ----------
    ball_positions : (x, y) or array-like of (x, y)  — positions in [0,1]²
    heaviness      : ball weight   (higher → larger deformation)
    tension        : net stiffness (higher → smaller / tighter deformation)
    N              : grid resolution (default GRID_SIZE=69)

    Returns
    -------
    Xd, Yd : (N, N) float64 arrays – deformed node coordinates in [0, 1]²
    """
    # ── normalise ball input ─────────────────────────────────────────────────
    balls = np.asarray(ball_positions, dtype=np.float64)
    if balls.ndim == 1:                  # single ball: (2,) → (1, 2)
        balls = balls[np.newaxis, :]

    X, Y = make_grid(N)                  # (N, N) each

    scale     = float(heaviness) / float(tension)
    sigma     = 0.12 * (1.0 + scale)    # Gaussian influence radius
    amplitude = 0.10 * scale            # peak displacement magnitude

    # ── vectorised over all balls  (n, N, N)  ────────────────────────────────
    bx = balls[:, 0, None, None]        # (n, 1, 1)
    by = balls[:, 1, None, None]
    dist2  = (X[None] - bx) ** 2 + (Y[None] - by) ** 2   # (n, N, N)
    r      = np.sqrt(dist2)
    kernel = np.exp(-dist2 / (2.0 * sigma ** 2))
    inv_d  = 1.0 / (r + 1e-12)

    # Unit direction from each node toward each ball
    ux = (bx - X[None]) * inv_d         # (n, N, N)
    uy = (by - Y[None]) * inv_d

    # Clamp displacement so no node overshoots the ball (prevents mesh folding)
    mag = np.minimum(amplitude * kernel, 0.9 * r)

    dx = (mag * ux).sum(axis=0)         # (N, N)
    dy = (mag * uy).sum(axis=0)

    # ── corner-pinning via bilinear correction ────────────────────────────────
    # Build a bilinear field that equals dx/dy at the four corners, then subtract
    # it — leaving all corner displacements exactly at zero.
    cx = np.array([dx[0, 0], dx[0, -1], dx[-1, 0], dx[-1, -1]])
    cy = np.array([dy[0, 0], dy[0, -1], dy[-1, 0], dy[-1, -1]])

    # Bilinear basis evaluated at every node
    #   w = [(1-X)(1-Y),  X(1-Y),  (1-X)Y,  X·Y]
    w = np.stack([(1-X)*(1-Y), X*(1-Y), (1-X)*Y, X*Y])   # (4, N, N)
    flat = w.reshape(4, -1)                                 # (4, N²)

    dx -= (cx @ flat).reshape(N, N)
    dy -= (cy @ flat).reshape(N, N)

    return X + dx, Y + dy


def warp_image(img, Xd, Yd):
    """
    Sample *img* at each deformed mesh node position.

    For every node (i, j) the output pixel gets the colour of the original
    image at (Xd[i,j], Yd[i,j]).  This gives an (N, N[, C]) array —
    N×N equally-spaced pixels, each representing one mesh node.

    Coordinate convention
    ---------------------
    Mesh  : X in [0,1] = left→right,  Y in [0,1] = bottom→top
    Image : column 0 = left,  row 0 = top  → need Y-flip when indexing.
    """
    H, W = img.shape[:2]
    img_f = img.astype(np.float32)
    if img_f.max() > 1.0:
        img_f /= 255.0

    # Deformed coords → pixel indices; flip Y (mesh Y=1 = top = image row 0)
    xi = np.clip(Xd          * (W - 1), 0, W - 1)
    yi = np.clip((1.0 - Yd)  * (H - 1), 0, H - 1)

    coords = [yi, xi]
    if img_f.ndim == 2:
        return map_coordinates(img_f, coords, order=1, mode='nearest')
    return np.stack(
        [map_coordinates(img_f[:, :, c], coords, order=1, mode='nearest')
         for c in range(img_f.shape[2])],
        axis=2,
    )


# ─────────────────────────────────── visualisation ────────────────────────────

def _line_segments(Xd, Yd, stride: int):
    """Polyline segments for horizontal + vertical grid lines (stride subsampling)."""
    N   = Xd.shape[0]
    idx = np.arange(0, N, stride)
    segs = []
    for i in idx:
        segs.append(np.column_stack([Xd[i, :],  Yd[i, :]]))
    for j in idx:
        segs.append(np.column_stack([Xd[:, j],  Yd[:, j]]))
    return segs


def draw_mesh(ax, Xd, Yd, *, stride: int = 4,
              color=NET_COLOR, lw: float = 0.6, alpha: float = 0.85,
              bg: str = DARK_BG):
    """Render deformed (or original) mesh onto ax via LineCollection."""
    ax.set_facecolor(bg)
    lc = LineCollection(_line_segments(Xd, Yd, stride),
                        colors=color, linewidths=lw, alpha=alpha,
                        antialiaseds=True)
    ax.add_collection(lc)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


def visualize_single(ball_positions, heaviness: float = 1.0,
                     tension: float = 1.0, stride: int = 3):
    """
    Side-by-side comparison: original mesh | deformed mesh.

    Returns
    -------
    Xd, Yd : deformed coordinate arrays (use these for downstream processing)
    """
    balls = np.asarray(ball_positions, dtype=np.float64)
    if balls.ndim == 1:
        balls = balls[np.newaxis, :]

    Xd, Yd = compute_deformation(balls, heaviness, tension)
    X0, Y0 = make_grid()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor(DARK_BG)

    draw_mesh(ax1, X0, Y0, stride=stride, color="#404858")
    ax1.set_title("Original", color="white", fontsize=11, pad=6)

    draw_mesh(ax2, Xd, Yd, stride=stride)
    n_label = f"{len(balls)} ball{'s' if len(balls) > 1 else ''}"
    ax2.set_title(
        f"Deformed  ({n_label}  |  heaviness={heaviness}  tension={tension})",
        color="white", fontsize=11, pad=6,
    )

    plt.tight_layout(pad=1.5)
    plt.show()
    return Xd, Yd


def visualize_grid(
    heaviness_range: tuple = (0.5, 3.0),
    tension_range:   tuple = (0.3, 2.0),
    n:               int   = 5,
    stride:          int   = 5,
    save:            bool  = True,
    bg_image:        str   = None,
):
    """
    n×n panel (n² cells) exploring all three parameters simultaneously.

    Layout
    ------
    Rows    : heaviness  increases top → bottom  (heaviness_range[0] → [1])
    Columns : tension    increases left → right  (tension_range[0]   → [1])
    Ball xy : moves diagonally — bx ∝ column index, by ∝ row index
              both mapped to [0.15, 0.85] so the ball stays inside the net.

    Each cell carries a unique (heaviness, tension, ball_position) triple,
    giving n² = 100 distinct combinations for the default n=10.
    """
    heaviness_vals = np.linspace(*heaviness_range, n)
    tension_vals   = np.linspace(*tension_range,   n)

    bg_img = mpimg.imread(bg_image) if bg_image is not None else None

    fig, axes = plt.subplots(n, n, figsize=(20, 20),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.04})
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        f"Net deformation — {n*n} combinations  ({GRID_SIZE}×{GRID_SIZE} net)  |  "
        f"rows: heaviness {heaviness_range[0]}→{heaviness_range[1]}  "
        f"cols: tension {tension_range[0]}→{tension_range[1]}  "
        f"ball: diagonal sweep (● marks drop position)",
        color="white", fontsize=10, y=0.999,
    )

    for r, h in enumerate(heaviness_vals):
        for c, t in enumerate(tension_vals):
            # Ball position follows the grid index diagonally
            bx = 0.15 + 0.70 * c / max(n - 1, 1)
            by = 0.15 + 0.70 * r / max(n - 1, 1)
            ax = axes[r, c]
            Xd, Yd = compute_deformation((bx, by), h, t)
            if bg_img is not None:
                warped = warp_image(bg_img, Xd, Yd)
                cmap = 'gray' if warped.ndim == 2 else None
                ax.imshow(warped, extent=[0, 1, 0, 1], origin='lower',
                          aspect='auto', cmap=cmap)
                ax.set_facecolor(DARK_BG)
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.set_aspect('equal')
                ax.axis('off')
            else:
                draw_mesh(ax, Xd, Yd, stride=stride)
            # Parameter labels on outer edges only
            if c == 0:
                ax.text(-0.06, 0.5, f"h={h:.1f}",
                        transform=ax.transAxes, color="#aaaaaa",
                        fontsize=5.5, ha="right", va="center")
            if r == n - 1:
                ax.text(0.5, -0.06, f"t={t:.2f}",
                        transform=ax.transAxes, color="#aaaaaa",
                        fontsize=5.5, ha="center", va="top")

    if save:
        path = "net_deformation_grid.png"
        fig.savefig(path, dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {path}")

    plt.show()


# ─────────────────────────────────── main demo ────────────────────────────────

if __name__ == "__main__":
#Demo 4: 10×10 exploration grid (100 unique combinations) ───────────────
    print("\nDemo 4: 10×10 grid — 100 combinations of position, heaviness, tension…")
    t0 = time.perf_counter()
    visualize_grid(heaviness_range=(0.5, 3.0), tension_range=(0.3, 2.0), n=5,
                   bg_image=r"misc/radiation_data/bw_jon_images_from_paper/eval_mowing_10.png")
    print(f"  Grid rendered in {time.perf_counter() - t0:.2f}s")
