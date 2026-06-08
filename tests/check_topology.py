"""Topology checks: the single-atom case is the bit-identical degeneracy the
engine relies on today, and the multi-site path places a rigid body correctly.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gcmc import topology as T  # noqa: E402


def ok(m):
    print("  ok ", m)


# ---- single_atom is the identity --------------------------------------------
for dim in (2, 3):
    sa = T.single_atom(dim=dim)
    assert sa.n_sites == 1 and sa.dim == dim and sa.is_single_atom
    assert sa.site_types.tolist() == [0]
    rng = np.random.default_rng(0)
    centers = rng.uniform(0, 50, size=(200, dim))
    # place() returns exactly [center]
    for c in centers[:20]:
        p = T.place(sa, c)
        assert p.shape == (1, dim) and np.array_equal(p[0], c)
    # expand() returns the centres array unchanged, value-for-value
    topo = T.Topology(sa)
    exp = topo.expand(centers)
    assert np.array_equal(exp, centers), "single_atom expand must be the identity"
    assert exp.dtype == centers.dtype or np.allclose(exp, centers)
    assert topo.mol_ids(5).tolist() == [0, 1, 2, 3, 4]
    assert topo.site_types(5).tolist() == [0, 0, 0, 0, 0]
ok("single_atom places and expands as the identity (2D and 3D)")

# ---- water is a rigid three-site body ---------------------------------------
w = T.water(dim=3)
assert w.n_sites == 3 and w.site_types.tolist() == [0, 1, 1]
assert not w.is_single_atom
# place at a centre, identity orientation
c = np.array([10.0, 20.0, 30.0])
sites = T.place(w, c)
assert sites.shape == (3, 3)
assert np.allclose(sites[0], c), "O site sits on the centre"
# internal geometry: the two O-H distances are equal and equal the bond length
d_oh1 = np.linalg.norm(sites[1] - sites[0])
d_oh2 = np.linalg.norm(sites[2] - sites[0])
assert np.isclose(d_oh1, 0.9572) and np.isclose(d_oh2, 0.9572)
ok("water places three sites with the right bond lengths")

# rotation preserves the rigid geometry (all pairwise site distances invariant)
theta = 0.7
Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta),  np.cos(theta), 0],
               [0, 0, 1.0]])
rot = T.place(w, c, orientation=Rz)
def pdist(a):
    return np.array([np.linalg.norm(a[i] - a[j]) for i in range(3) for j in range(i+1, 3)])
assert np.allclose(pdist(sites), pdist(rot)), "rotation must preserve internal geometry"
assert np.allclose(rot[0], c), "centre is the rotation fixed point"
ok("rotation preserves water's internal distances")

# ---- Topology of several waters ---------------------------------------------
topo = T.Topology(w)
centers = np.array([[0., 0., 0.], [5., 0., 0.], [0., 5., 0.]])
flat = topo.expand(centers)
assert flat.shape == (9, 3)
assert topo.mol_ids(3).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
assert topo.site_types(3).tolist() == [0, 1, 1, 0, 1, 1, 0, 1, 1]
# each molecule's O lands on its centre
for i, ctr in enumerate(centers):
    assert np.allclose(flat[i * 3], ctr)
ok("Topology expands centres into typed, molecule-tagged site arrays")

print("\nALL TOPOLOGY TESTS PASSED")
