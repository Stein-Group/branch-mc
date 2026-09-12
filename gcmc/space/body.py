"""Oriented body geometry: rotations, chords, and membership for cuboids and rods.

WHY THIS IS ITS OWN MODULE AT LAYER 2. These functions answer purely geometric
questions -- where is the body, and which particles are inside it -- and they
knew nothing about sampling even while they lived in `samplers/channel.py`. That
placement forced `channel_move` (layer 4, proposals) to import from layer 5, one
of the four upward imports the ladder test found.

What belongs here is anything a body needs to exist and be queried:

    random_rotation_matrix   a uniform orientation
    local_from_box           box coordinates -> body-local coordinates
    ray_box_chord            where a line through the box meets the walls
    axial_coord_full         a particle's coordinate along a wall-to-wall rod
    cuboid_members           which of an array of points lie inside an oriented
                             cuboid; `cuboid_members_cl` is the same test over a
                             cell list, which is cheaper when one is to hand
    make_cuboid_cloud        a uniform trial cloud inside an axis-aligned cuboid
    oriented_cuboid_trials   the same, rotated to a random orientation
    rod_members_full_cl      which slots lie inside a wall-to-wall rod

What does NOT belong here is any notion of a weight, a trial, an acceptance or a
transfer matrix. This module can tell you a particle is inside the body; it has
no opinion about what that costs.

MEMBERSHIP RETURNS SLOT IDS, not positions or dense indices. The cell list owns
the slot table, so a membership query that returned coordinates would force
every caller to re-derive which particle it was looking at -- and the two
answers could then disagree. Slots keep one identity for a particle across the
query, the sweep and the commit.

UNITS: reduced throughout. Every length -- box sides, half-extents, chord
parameters -- is on the same reduced scale.
"""

from __future__ import annotations

import math

import numpy as np

from .._types import RNG, FloatArray, IntArray
from .neighbors import CellList

#: The 27 periodic image offsets of a cubic cell, precomputed. Built once at
#: import rather than per call: `axial_coord_full` needs every image of a
#: candidate to find the one nearest the rod, and rebuilding a 27x3 array inside
#: a per-move loop was measurable.
_IMAGES = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=float,
)

__all__ = [
    "axial_coord_full",
    "cuboid_members",
    "cuboid_members_cl",
    "cuboid_packing_bound",
    "local_from_box",
    "make_cuboid_cloud",
    "oriented_cuboid_trials",
    "random_rotation_matrix",
    "ray_box_chord",
    "rod_members_full",
    "rod_members_full_cl",
]


def random_rotation_matrix(rng: RNG, dim: int = 3) -> FloatArray:
    """Uniform (Haar) rotation matrix from a numpy Generator.

    dim == 3 returns a 3x3 Haar rotation. dim == 2 returns the 2x2 rotation for
    an angle drawn uniformly on [0, 2*pi); it orients the flat channel rectangle
    (one long axis plus one perpendicular width) exactly as the 3x3 orients the
    cuboid, so column 0 is the sweep axis and the remaining column(s) span the
    cross-section in either dimension.
    """
    if dim == 2:
        theta = rng.uniform(0.0, 2.0 * math.pi)
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s], [s, c]])
    from scipy.spatial.transform import Rotation

    return Rotation.random(random_state=rng).as_matrix()


def local_from_box(
    P: FloatArray, center: FloatArray, R: FloatArray, side: float | FloatArray
) -> FloatArray:
    """Recover cuboid-local coordinates from periodic box coordinates:
    q = minimage(P - center); U = q @ R."""
    q = np.asarray(P, float) - np.asarray(center, float)
    q -= np.round(q / side) * side
    return q @ R


def _unit_images(dim: int) -> FloatArray:
    """The 3**dim nearest periodic images as offsets in {-1, 0, 1}**dim, in the
    same lexicographic (-1, 0, 1) order the membership scans walk (outermost axis
    varies slowest), so the first-hit seam rule is order-preserving in any
    dimension. dim == 3 returns the frozen `_IMAGES` constant unchanged."""
    if dim == 3:
        return _IMAGES
    axis = (-1.0, 0.0, 1.0)
    grids = np.meshgrid(*([np.array(axis)] * dim), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def ray_box_chord(
    center: FloatArray, axis: FloatArray, box: FloatArray
) -> tuple[float, float]:
    """Signed extents (s_min, s_max) of the infinite line {center + s*axis}
    clipped to the box [0, box)^3 (the slab method). center lies inside the box,
    so s_min <= 0 <= s_max and the chord length is s_max - s_min. For an axis
    line the chord is one full traversal (length box) whose ends coincide under
    the PBC; for an oblique line it is the wall-to-wall segment.
    """
    center = np.asarray(center, float)
    axis = np.asarray(axis, float)
    dim = len(axis)
    box = np.broadcast_to(np.asarray(box, float), (dim,))
    s_lo, s_hi = -np.inf, np.inf
    for i in range(dim):
        a = axis[i]
        if abs(a) < 1e-15:
            # line is constant on this axis; center must sit inside the slab
            if center[i] < 0.0 or center[i] > box[i]:
                return 0.0, 0.0
            continue
        t0 = (0.0 - center[i]) / a
        t1 = (box[i] - center[i]) / a
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        s_lo = max(s_lo, lo)
        s_hi = min(s_hi, hi)
    if not (s_lo <= s_hi):
        return 0.0, 0.0
    return float(s_lo), float(s_hi)


def axial_coord_full(
    coords: FloatArray,
    center: FloatArray,
    axis: FloatArray,
    s_min: float,
    s_max: float,
    box: FloatArray,
) -> FloatArray:
    """Axial coordinate s of each of `coords` along the full-span rod, i.e. the
    value in [s_min, s_max] of the periodic image that lies on the rod. Every
    coordinate handed here is already known to belong to the rod (a trial or a
    membership hit), so exactly one image qualifies (up to the measure-zero seam)
    and its s is returned. Used to order the sweep; the physics uses box coords.

    An image g shifts every axial coordinate by the same scalar g.axis, so the
    projection is done once and each image is a scalar subtraction. Images whose
    shifted range misses [s_min, s_max] cannot hold a hit and are skipped;
    survivors are visited in the fixed (-1, 0, 1) order, so the first-hit rule
    that decides s for a seam atom is unchanged. Generic orientations leave only
    g = 0, which is the common case and now costs one pass.
    """
    coords = np.asarray(coords, float)
    center = np.asarray(center, float)
    axis = np.asarray(axis, float)
    dim = len(axis)
    box = np.broadcast_to(np.asarray(box, float), (dim,))
    n = len(coords)
    if n == 0:
        return np.zeros(0)
    pad = 1e-9 * (abs(s_max) + abs(s_min) + 1.0)
    s0 = (coords - center) @ axis
    lo, hi = s0.min(), s0.max()
    ga_all = (_unit_images(dim) * box) @ axis  # (3**dim,) axial shift per image
    out = np.zeros(n)
    found = np.zeros(n, dtype=bool)
    for ga in ga_all:
        if (lo - ga) > s_max + pad or (hi - ga) < s_min - pad:
            continue
        s = s0 - ga
        ok = (~found) & (s >= s_min - pad) & (s <= s_max + pad)
        out[ok] = s[ok]
        found |= ok
        if found.all():
            break
    return out


def rod_members_full(
    coords: FloatArray,
    center: FloatArray,
    axis: FloatArray,
    e1: FloatArray,
    e2: FloatArray | None,
    s_min: float,
    s_max: float,
    half_w: float,
    box: FloatArray,
) -> tuple[IntArray, FloatArray]:
    """Indices of `coords` inside the full-span rod (axial range [s_min, s_max],
    cross-section 2*half_w per perpendicular axis) and their axial coordinates.
    Removal-membership counterpart to the trial region: an atom is a member iff a
    trial could have been generated at its position, so insertion and removal act
    on one body. In 3D the cross-section is a square e1 x e2; in 2D there is one
    perpendicular axis e1 (pass e2=None) and the body is a wall-to-wall strip.

    Three facts keep this an O(N) numpy scan rather than a spatial query:

    * The projections are affine in the image. (p - c - g).axis is
      (p - c).axis - g.axis, so ONE dim x (dim, N) product against the frame
      [axis|e1|(e2)] serves every image; each image is a shift of dim scalars.
    * The chord is clipped to the box, so the axis line lies in [0, L] and the
      rod reaches at most d_a = half_w*(|e1_a| + |e2_a|) past a face on axis a.
      A wrapped atom can only reach the rod through a nonzero image if it sits
      within d_a of face a. That shell is thin, so the bulk of the system is
      settled by g = 0 alone and the other images touch the shell only.
    * The projections are held as contiguous rows, not as columns of an
      (image, atom, dim) block. The strided slices of that block, not the
      arithmetic, dominated the cost.

    Images are enumerated in the fixed (-1, 0, 1) order and the first hit wins,
    so a seam atom (measure zero, and only reachable when the chord's ends meet
    through the boundary) keeps its former axial coordinate.
    """
    coords = np.asarray(coords, float)
    n = len(coords)
    if n == 0:
        return np.empty(0, dtype=int), np.empty(0)
    center = np.asarray(center, float)
    axis = np.asarray(axis, float)
    dim = len(axis)
    box = np.broadcast_to(np.asarray(box, float), (dim,))
    e1 = np.asarray(e1, float)
    cross = [e1] if e2 is None else [e1, np.asarray(e2, float)]
    M = np.column_stack([axis, *cross])  # (dim, dim)

    P = coords
    if (P.min() < 0.0 or P.max() >= box.min()) and (
        (P < 0.0).any() or (box <= P).any()
    ):
        P = np.mod(P, box)
    Q = M.T @ (P - center).T  # (dim, n): row 0 = s, rows 1.. = cross coords
    s0 = Q[0]
    cc = Q[1:]  # (dim-1, n)

    # per-axis reach of the cross-section past a face
    reach = half_w * sum(np.abs(e) for e in cross) + 1e-12
    shell = ((reach >= P) | (box - reach <= P)).any(axis=1)

    # g = 0, over every atom: narrow on the tightest test first, no gathers
    inside = s0 >= s_min
    inside &= s0 <= s_max
    for row in cc:
        inside &= np.abs(row) <= half_w
    s_out = np.where(inside, s0, 0.0)
    inside &= ~shell  # shell atoms are redone in order

    # shell: all 3**dim images at once, first hit in image order
    edge = np.nonzero(shell)[0]
    if len(edge):
        shifts = (_unit_images(dim) * box) @ M  # (3**dim, dim)
        s = s0[edge] - shifts[:, 0][:, None]  # (3**dim, m)
        hit = (s >= s_min) & (s <= s_max)
        for a in range(1, dim):
            ca = cc[a - 1][edge] - shifts[:, a][:, None]
            hit &= np.abs(ca) <= half_w
        any_hit = hit.any(axis=0)
        if any_hit.any():
            col = np.nonzero(any_hit)[0]
            first = hit.argmax(axis=0)[col]  # earliest image that hits
            sel = edge[col]
            s_out[sel] = s[first, col]
            inside[sel] = True

    loc = np.nonzero(inside)[0]
    return loc, s_out[loc]


def cuboid_members_cl(
    cl: CellList,
    center: FloatArray,
    R: FloatArray,
    half_long: float,
    half_w: float,
    side: float | FloatArray,
) -> IntArray:
    """`cuboid_members` restricted to the cells the rotated cuboid touches.

    The box-frame AABB of a rotated box has half-extents |R| @ half_extents:
    each box axis picks up |R[a, j]| of every local half-extent. That is the
    smallest axis-aligned superset, which is all the gather needs. In 2D the
    body is a rectangle, so the cross-section contributes one half_w, not two.
    """
    ext = np.array([half_long] + [half_w] * (cl.dim - 1))
    h = np.abs(np.asarray(R, float)) @ ext
    center = np.asarray(center, float)
    lo, hi = center - h, center + h
    # unsorted: the exact test below cuts the candidates down by an order of
    # magnitude, so sorting the survivors is far cheaper than sorting the gather
    cand = cl.gather_aabb(lo, hi, sort=False)
    if cand.size and cl.n_matrix:
        cand = cand[~cl.is_matrix[cand]]
    if cand.size == 0:
        return np.empty(0, dtype=np.int64)
    q = cl.coords[cand] - center
    _sides = np.broadcast_to(np.asarray(side, float), (cl.dim,))
    if bool((lo < 0.0).any()) or bool((hi >= _sides).any()):
        # The correction is needed only when the cuboid pokes through a face.
        # When the AABB [lo, hi] lies strictly inside [0, side) the skip is
        # exact, but NOT because every |q| is small: gather_aabb returns
        # candidates up to a cell width outside the AABB, so |q| can exceed
        # side/2 in the sliver. The real argument: the solid cuboid is then a
        # subset of [0, side), so no wrapped image of any in-box point can lie
        # inside it, and for a point whose raw image is inside, |q| <= h <=
        # side/2 makes the correction the identity. Raw and minimum-image
        # membership therefore agree on every candidate, sliver included.
        q -= np.round(q / side) * side
    # local coordinates as ROWS: `(U <= ext).all(axis=1)` on an (n, 3) array is a
    # strided reduction and costs more than the rotation it follows
    U = np.ascontiguousarray(R.T) @ q.T
    np.abs(U, out=U)
    inside = U[0] <= ext[0]
    for a in range(1, cl.dim):
        inside &= U[a] <= ext[a]
    return np.sort(cand[inside])


def rod_members_full_cl(
    cl: CellList,
    center: FloatArray,
    axis: FloatArray,
    e1: FloatArray,
    e2: FloatArray | None,
    s_min: float,
    s_max: float,
    half_w: float,
    box: FloatArray,
) -> tuple[IntArray, FloatArray]:
    """`rod_members_full` restricted to the cells the wall-to-wall rod touches.

    An AABB is useless here: a rod along the body diagonal has the whole box for
    its bounding box. So the rod is covered by sampling its axis and taking the
    union of the samples' cell blocks -- a tube, not a box.

    The sampling pitch and the margin are chosen together. With samples every
    `delta` along the axis, every rod point is within `delta/2` of a sample along
    the axis and within `half_w * sqrt(2)` of the axis (the cross-section's
    corner), so a margin of `delta/2 + half_w*sqrt(2)` covers the rod, and
    `delta = cw/2` keeps that margin inside one cell of reach.
    """
    center = np.asarray(center, float)
    axis = np.asarray(axis, float)
    delta = 0.5 * float(cl.cw.min())
    nseg = max(1, int(np.ceil((s_max - s_min) / delta)))
    s = np.linspace(s_min, s_max, nseg + 1)
    pts = np.mod(center + s[:, None] * axis, cl.box)
    margin = 0.5 * (s[1] - s[0] if nseg > 1 else delta) + half_w * math.sqrt(2.0) + 1e-9
    cand = cl.gather_points(pts, margin)
    if cand.size and cl.n_matrix:
        cand = cand[~cl.is_matrix[cand]]
    if cand.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0)
    idx, ax = rod_members_full(
        cl.coords[cand], center, axis, e1, e2, s_min, s_max, half_w, box
    )
    members = cand[idx]
    order = np.argsort(members, kind="stable")  # ascending, ax kept alongside
    return members[order], ax[order]


# ---------------------------------------------------------------------------
# THE ORIENTED CUBOID: drawing a trial cloud inside one, and asking which
# particles are already in it.
#
# These sat in `sampling/channel.py` because the channel sweep is what uses
# them. They are geometry -- a rotation, a half-extent test, a uniform draw in a
# box -- and they know nothing about sampling, which is why `cuboid_members_cl`
# (the same predicate over a cell list) was already here and its array twin was
# not. Layer 2 is where a distance test belongs.
# ---------------------------------------------------------------------------


def make_cuboid_cloud(
    rng: RNG, n_trials: int, L_long: float, w: float, box_L: float
) -> tuple[FloatArray, FloatArray]:
    """Thin cuboid trial cloud at a uniform centre and uniform (Haar) orientation.
    Returns (U, P): U local coordinates, P periodic box coordinates."""
    from scipy.spatial.transform import Rotation

    a, b = 0.5 * L_long, 0.5 * w
    U = np.column_stack(
        [
            rng.uniform(-a, a, n_trials),
            rng.uniform(-b, b, n_trials),
            rng.uniform(-b, b, n_trials),
        ]
    )
    c = rng.uniform(0.0, box_L, 3)
    R = Rotation.random(random_state=rng).as_matrix()
    P = np.mod(c + U @ R.T, box_L)
    return U, P


def oriented_cuboid_trials(
    rng: RNG,
    center: FloatArray,
    n_trials: int,
    L_long: float,
    w: float,
    side: float | FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Trial centres uniform in a thin cuboid (length L_long, cross-section w x w)
    at `center` with a uniform random orientation. Returns (P, R)."""
    a, b = 0.5 * L_long, 0.5 * w
    U = np.column_stack(
        [
            rng.uniform(-a, a, n_trials),
            rng.uniform(-b, b, n_trials),
            rng.uniform(-b, b, n_trials),
        ]
    )
    R = random_rotation_matrix(rng)
    P = np.mod(np.asarray(center, float) + U @ R.T, side)
    return P, R


def cuboid_members(
    coords: FloatArray,
    center: FloatArray,
    R: FloatArray,
    half_long: float,
    half_w: float,
    side: float | FloatArray,
) -> IntArray:
    """Indices of `coords` inside the oriented cuboid (half-extents half_long x
    half_w [x half_w]) centred at `center`. Removal-membership counterpart to
    the trial region, so insertion and removal share one body. In 2D the body is
    a rectangle (half_long x half_w); the cross-section drops one axis."""
    coords = np.asarray(coords, float)
    if len(coords) == 0:
        return np.empty(0, dtype=int)
    U = local_from_box(coords, center, R, side)
    half = np.array([half_long] + [half_w] * (U.shape[1] - 1))
    inside = (np.abs(U) <= half).all(axis=1)
    return np.where(inside)[0]


#: Kepler's sphere-packing density, pi / sqrt(18). The proven maximum fraction
#: of space that non-overlapping equal spheres can fill; its 2D twin is the
#: hexagonal density pi / sqrt(12).
_KEPLER_3D = math.pi / math.sqrt(18.0)
_HEX_2D = math.pi / math.sqrt(12.0)


def cuboid_packing_bound(
    channel_length: float, channel_width: float, dim: int, sigma: float
) -> float:
    """Upper bound on how many hard spheres of diameter `sigma` can have their
    CENTRES inside a channel_length x channel_width^(dim-1) body.

    Section 1.3 and 3.7 of the design plan. `kmax` is derived from this
    plus a margin, and 3.7 is what makes the bound's DIRECTION load-bearing: the
    window kernel is pi-invariant on {k_in <= min(kmax, M)} and the identity
    elsewhere, so for hard cores "set kmax and M above the geometric packing
    bound and the region is everything" -- exact, with both truncation counters
    identically zero. A bound that is too LOW silently truncates the state space
    and the chain reports a plausible wrong isotherm; a bound that is too HIGH
    only widens the DP's C array, which section 2B.6 measured at 2.2x in the
    worst case (M = 64: 20 ms at kmax = 12 against 44 ms at kmax = M). So this
    is deliberately an over-estimate, and it is RIGOROUS rather than fitted.

    THE ARGUMENT. Centres lie in the body; each carries a ball of radius
    sigma/2, and those balls are pairwise disjoint (centres are at least sigma
    apart) and contained in the body DILATED by sigma/2 -- whose measure is the
    Steiner formula for a box. So

        k * v_ball(sigma/2)  <=  eta_max * measure(dilated body)

    with eta_max the proven maximum packing density in that dimension: Kepler's
    pi/sqrt(18) in 3D, the hexagonal pi/sqrt(12) in 2D. No lattice, no column
    structure and no aspect ratio is assumed, which is what makes it safe for
    the obliquely-oriented thin channels this engine runs in.

    IT IS LOOSE FOR A THIN BODY, AND SAYING SO IS THE POINT. For L = 6, w = 1.5,
    sigma = 1 it returns about 58 where the true capacity is nearer 20: the
    dilation term dominates when two of the three extents are of order sigma.
    `sim_setup` therefore caps `kmax` at the pool size M, and under the
    lambda <= 4 scoping rule (section 3.5) M is the binding constraint for every
    realistic thin channel -- so `kmax = M` in practice and the bound's job is
    only to stop kmax exceeding it on a long or wide body. `k <= M` is not a
    truncation: it is a fact about subsets of an M-element pool.

    WHY NOT A TIGHTER BOUND. Slab arguments (partition the axis, bound each
    slab by the maximum number of cross-section points at the induced minimum
    separation) were tried and are WORSE here -- they double-count across slab
    boundaries, and for this geometry every choice of slab width returns
    something above the Steiner bound. A numerical maximum-independent-set on a
    grid would be tighter and would not be rigorous, which is the wrong trade
    for a quantity whose error in one direction is a silent bias.
    """
    L = float(channel_length)
    w = float(channel_width)
    s = float(sigma)
    if L <= 0.0 or w <= 0.0 or s <= 0.0:
        raise ValueError(
            f"packing bound needs a positive body and diameter, got "
            f"L={L}, w={w}, sigma={s}"
        )
    r = 0.5 * s
    if dim == 2:
        # Steiner in the plane: area + perimeter*r + pi r^2
        dilated = L * w + 2.0 * (L + w) * r + math.pi * r * r
        return float(_HEX_2D * dilated / (math.pi * r * r))
    # Steiner in space for a box a x b x c:
    #   V + 2r(ab+bc+ca) + pi r^2 (a+b+c) + (4/3) pi r^3
    a, b, c = L, w, w
    dilated = (
        a * b * c
        + 2.0 * r * (a * b + b * c + c * a)
        + math.pi * r * r * (a + b + c)
        + (4.0 / 3.0) * math.pi * r**3
    )
    return float(_KEPLER_3D * dilated / ((4.0 / 3.0) * math.pi * r**3))
