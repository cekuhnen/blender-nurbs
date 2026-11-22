"""Mirror tool: Mirror Copy (a pattern copy across a plane) and Make Symmetric
(CVs paired across a plane, twins kept in Edit Mode).
    blender -b --factory-startup --python tests/test_mirror_symmetry.py"""
import json, os, sys
import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import cvs, features, geom, pattern, store, symmetry
from blender_nurbs.kernel import load as load_kernel

ext = load_kernel().blender_ext
ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

def W(o):
    return np.array(geom.world_matrix(o), float)

def select(*objs):
    if ctx.object is not None and ctx.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[0]

def nurbs(name, pts, order=4, cyclic=False, kind='NURBS'):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new(kind)
    if kind == 'BEZIER':
        sp.bezier_points.add(len(pts) - 1)
        for b, c in zip(sp.bezier_points, pts):
            b.co = c
            b.handle_left_type = b.handle_right_type = 'FREE'
            b.handle_left = (c[0] - 0.3, c[1] + 0.2, c[2] + 0.1)
            b.handle_right = (c[0] + 0.25, c[1] - 0.1, c[2] + 0.3)
        sp.use_cyclic_u = cyclic
    else:
        sp.points.add(len(pts) - 1)
        for p, c in zip(sp.points, pts):
            p.co = (*c, 1.0)
        sp.order_u = order; sp.use_endpoint_u = not cyclic; sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    return o

def cvs_world(o):
    M = W(o)
    sp = o.data.splines[0]
    P = np.array([tuple(q.co[:3]) for q in sp.points], float)
    return P @ M[:3, :3].T + M[:3, 3]

def mirror_world(P, O, n):
    return pattern.mirror_matrix(O, n)[:3, :3] @ np.asarray(P, float).T + pattern.mirror_matrix(O, n)[:3, 3:4]

def run_mirror(**kw):
    return bpy.ops.nproto.mirror(**kw)

def surf_grid(o):
    d = ext.surface_cvs(store.get_brep(o).faces()[0].surface())
    return np.array(d["points"], float).reshape(d["count_u"], d["count_v"], 3), d

def surf_samples(o, n=7):
    """World points and unit normals (Su x Sv) on each face's surface."""
    M = W(o)
    P, N = [], []
    for s in store.get_surfaces(o):
        (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
        for i in range(n):
            for j in range(n):
                u = u0 + (u1 - u0) * (0.1 + 0.8 * i / (n - 1))
                v = v0 + (v1 - v0) * (0.1 + 0.8 * j / (n - 1))
                h = 1e-5 * max(u1 - u0, v1 - v0)
                p = np.array(s.evaluate(u, v), float)
                su = (np.array(s.evaluate(u + h, v), float) - np.array(s.evaluate(u - h, v), float)) / (2 * h)
                sv = (np.array(s.evaluate(u, v + h), float) - np.array(s.evaluate(u, v - h), float)) / (2 * h)
                nn = np.cross(M[:3, :3] @ su, M[:3, :3] @ sv)
                P.append(M[:3, :3] @ p + M[:3, 3]); N.append(nn / np.linalg.norm(nn))
    return np.array(P), np.array(N)

def nearest(A, B):
    """For each point of A: (distance to the nearest point of B, its index)."""
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    j = D.argmin(axis=1)
    return D[np.arange(len(A)), j], j


# --- the plane in local space (pure) ----------------------------------------
M = np.eye(4)
M[:3, :3] = np.diag([2.0, 0.5, 1.5]) @ pattern._rotation((0.3, 0.2, 1.0), 40.0)
M[:3, 3] = (1.0, -2.0, 0.5)
O, n = np.array([0.3, 0.4, -0.2]), pattern._unit((0.2, 1.0, 0.1))
sym = symmetry.plane_from_world(M, O, n)
X = np.random.default_rng(3).normal(size=(6, 3))
Xw = X @ M[:3, :3].T + M[:3, 3]
Rl = symmetry.reflect(sym, X)
Rw = Rl @ M[:3, :3].T + M[:3, 3]
check("local mirror = world mirror (object scaled unevenly and turned)",
      np.abs(Rw - mirror_world(Xw, O, n).T).max() < 1e-12, np.abs(Rw - mirror_world(Xw, O, n).T).max())
Pl = symmetry.project(sym, X) @ M[:3, :3].T + M[:3, 3]
check("projected points lie on the world plane", np.abs((Pl - O) @ n).max() < 1e-12)
O2, n2 = symmetry.world_plane(M, sym)
check("world_plane gives the plane back", abs(float((O2 - O) @ n)) < 1e-12 and np.allclose(n2, n))
check("height = world signed distance", np.allclose(symmetry.height(sym, X), (Xw - O) @ n))

# enforce: a moved CV puts its twin at its mirror image
sym0 = {"o": np.zeros(3), "d": np.array([0, 1.0, 0]), "c": np.array([0, 1.0, 0])}
P0 = np.array([[0, -2, 0], [1, -1, 0], [2, 0.3, 0], [1, 1, 0], [0, 2, 0]], float)
tw = symmetry.chain_twins(5)
lead = symmetry.master_mask(tw, 0)
S0 = symmetry.symmetrize(P0, tw, lead, sym0)
check("symmetrize: master half kept, the other mirrored, the middle on the plane",
      np.allclose(S0[:2], P0[:2]) and np.allclose(S0[3:], [[1, 1, 0], [0, 2, 0]]) and abs(S0[2, 1]) < 1e-15)
P1 = S0.copy(); P1[1] = (1.5, -1.2, 0.4)
Q, ch = symmetry.enforce(P1, S0, tw, np.zeros(5, bool), lead, sym0, 1e-9)
check("enforce: CV 1 moved -> CV 3 its mirror", ch and np.allclose(Q[3], (1.5, 1.2, 0.4)))
P1 = S0.copy(); P1[3] = (0.5, 0.7, -0.2)
Q, ch = symmetry.enforce(P1, S0, tw, np.zeros(5, bool), lead, sym0, 1e-9)
check("enforce: a follower moved (proportional editing) -> its master follows",
      np.allclose(Q[1], (0.5, -0.7, -0.2)))
P1 = S0.copy(); P1[1] = (1.0, -1.0, 1.0); P1[3] = (1.0, 1.0, -1.0)
sel = np.array([0, 0, 0, 1, 0], bool)
Q, _ = symmetry.enforce(P1, S0, tw, sel, lead, sym0, 1e-9)
check("enforce: both moved -> the selected one leads", np.allclose(Q[1], (1.0, -1.0, -1.0)))
Q, _ = symmetry.enforce(P1, S0, tw, np.ones(5, bool), lead, sym0, 1e-9)
check("enforce: both moved and selected -> the master side leads", np.allclose(Q[3], (1.0, 1.0, 1.0)))
P1 = S0.copy(); P1[2] = (2.2, 0.5, 0.1)
Q, _ = symmetry.enforce(P1, S0, tw, np.zeros(5, bool), lead, sym0, 1e-9)
check("enforce: the middle CV stays on the plane", abs(Q[2, 1]) < 1e-15 and np.allclose(Q[2, [0, 2]], (2.2, 0.1)))
P1 = S0.copy(); P1[3] = (0.4, 0.9, 0.3)              # an undo step that holds a move without its twin
Q = symmetry.resymmetrize(P1, tw, np.array([0, 0, 0, 1, 0], bool), lead, sym0)
check("resymmetrize (after undo): the selected CV leads", np.allclose(Q[1], (0.4, -0.9, 0.3)) and np.allclose(Q[3], P1[3]))
Q = symmetry.resymmetrize(P1, tw, np.zeros(5, bool), lead, sym0)
check("resymmetrize: nothing selected -> the master side", np.allclose(Q[3], (1, 1, 0)))
check("bezier twins: handles swap", list(symmetry.bezier_twins(3)) == [8, 7, 6, 5, 4, 3, 2, 1, 0])
check("grid twins reversed along V", list(symmetry.grid_twins(2, 3, "V")) == [2, 1, 0, 5, 4, 3])
check("cyclic twins with a shift", list(symmetry.chain_twins(6, True, 2)) == [2, 1, 0, 5, 4, 3])


# --- Make Symmetric on a curve ------------------------------------------------
pts = [(-2.0, -3.0, 0.0), (-1.0, -2.2, 0.8), (0.5, -1.0, 1.2), (1.0, 0.4, 1.0), (0.2, 1.3, 0.4),
       (-0.5, 2.0, 0.2), (-1.5, 3.4, 0.0)]
C = nurbs("SymCurve", pts)
C.location = (0.5, 0.0, 0.2)
C.rotation_euler = (0.0, 0.0, 0.3)
ctx.view_layer.update()
before = cvs_world(C)
select(C)
r = run_mirror(mirror_output='SYMMETRIC', center_point=(0, 0, 0), frame_z=(0, 1, 0), frame_x=(1, 0, 0))
check("Make Symmetric runs on a curve", r == {'FINISHED'})
P = cvs_world(C)
Mi = mirror_world(P, (0, 0, 0), (0, 1, 0)).T
check("curve: CV n-1-i is CV i's mirror image (world, object turned)", np.abs(P[::-1] - Mi).max() < 1e-5,
      np.abs(P[::-1] - Mi).max())
check("curve: the master half (the start) is kept", np.abs(P[:3] - before[:3]).max() < 1e-6)
check("curve: the middle CV on the plane", abs(P[3, 1]) < 1e-6)
check("curve: the plane is kept on the object", symmetry.sym_of(C) is not None)
O3, n3 = symmetry.world_plane(W(C), symmetry.sym_of(C))
check("curve: the kept plane is the world plane y = 0", abs(O3[1]) < 1e-9 and abs(abs(n3[1]) - 1) < 1e-9)

C2 = nurbs("SymCurveFlip", pts)
select(C2)
run_mirror(mirror_output='SYMMETRIC', center_point=(0, 0, 0), frame_z=(0, 1, 0), frame_x=(1, 0, 0),
           flip_master=True)
P = cvs_world(C2)
check("curve, Flip Master Side: the end is kept", np.abs(P[4:] - np.array(pts[4:])).max() < 1e-6)
check("curve, Flip Master Side: the start mirrors it", np.abs(P[::-1] - mirror_world(P, (0, 0, 0), (0, 1, 0)).T).max() < 1e-5)

# Bezier: point i twins n-1-i, its left handle the twin's right one
B = nurbs("SymBezier", [(-1.0, -1.5, 0.0), (0.2, -0.4, 0.5), (0.6, 0.6, 0.3), (-0.8, 1.8, 0.1)], kind='BEZIER')
select(B)
run_mirror(mirror_output='SYMMETRIC', center_point=(0, 0, 0), frame_z=(0, 1, 0), frame_x=(1, 0, 0))
bp = B.data.splines[0].bezier_points
mir = lambda v: np.array((v[0], -v[1], v[2]))
check("bezier: points mirrored", np.allclose(np.array(bp[3].co), mir(bp[0].co), atol=1e-6)
      and np.allclose(np.array(bp[2].co), mir(bp[1].co), atol=1e-6))
check("bezier: a left handle mirrors its twin's right handle",
      np.allclose(np.array(bp[3].handle_left), mir(bp[0].handle_right), atol=1e-6)
      and np.allclose(np.array(bp[2].handle_right), mir(bp[1].handle_left), atol=1e-6))

# a closed curve: the twin shift that fits
ring = [(np.cos(a) * 2 + 0.3 * np.sin(3 * a), np.sin(a) * 1.5 + 0.1, 0.2 * np.cos(2 * a))
        for a in np.linspace(0, 2 * np.pi, 9)[:-1] + 0.35]
R = nurbs("SymRing", ring, cyclic=True)
select(R)
run_mirror(mirror_output='SYMMETRIC', center_point=(0, 0, 0), frame_z=(0, 1, 0), frame_x=(1, 0, 0))
P = cvs_world(R)
s = symmetry.sym_of(R)
tw = symmetry.chain_twins(len(P), True, s["shift"]["0"])
check("closed curve: CV s-i is CV i's mirror", np.abs(P[tw] - mirror_world(P, (0, 0, 0), (0, 1, 0)).T).max() < 1e-5)


# --- Edit Mode keeps the pairs (the depsgraph handler) -----------------------
select(C)
bpy.ops.object.mode_set(mode='EDIT')
ctx.view_layer.update()
sp = C.data.splines[0]
for q in sp.points:
    q.select = False
sp.points[1].select = True
sp.points[1].co = (sp.points[1].co[0] + 0.3, sp.points[1].co[1] - 0.2, sp.points[1].co[2] + 0.5, 1.0)
C.data.update_tag()
ctx.view_layer.update()
ctx.view_layer.update()
sp = C.data.splines[0]
L = np.array([tuple(q.co[:3]) for q in sp.points])
Ms = symmetry.sym_of(C)
check("Edit Mode: moving CV 1 moves CV 5 to its mirror (handler)",
      np.abs(L[5] - symmetry.reflect(Ms, L[1:2])[0]).max() < 1e-5, L[5])
sp.points[3].co = (sp.points[3].co[0] + 0.4, sp.points[3].co[1] + 0.7, sp.points[3].co[2], 1.0)
C.data.update_tag()
ctx.view_layer.update()
ctx.view_layer.update()
L = np.array([tuple(q.co[:3]) for q in C.data.splines[0].points])
check("Edit Mode: the middle CV is held on the plane", abs(float(symmetry.height(Ms, L[3:4])[0])) < 1e-5)
twins = symmetry.twins_world(C)
check("Edit Mode: the selected CV's twin is drawn", len(twins) == 1
      and np.abs(np.array(twins[0]) - (W(C)[:3, :3] @ L[5] + W(C)[:3, 3])).max() < 1e-5)
bpy.ops.object.mode_set(mode='OBJECT')
P = cvs_world(C)
check("after Edit Mode the curve is still symmetric", np.abs(P[::-1] - mirror_world(P, (0, 0, 0), (0, 1, 0)).T).max() < 1e-5)

select(C)
bpy.ops.nproto.remove_symmetry()
check("Remove Symmetry drops the plane", symmetry.sym_of(C) is None)


# --- Make Symmetric on a surface, and its CV cage -----------------------------
nu, nv = 6, 5
rng = np.random.default_rng(7)
G = np.array([[(-2.5 + iu, -2.0 + iv, 0.3 * np.sin(iu + 0.7 * iv) + 0.15 * rng.normal())
               for iv in range(nv)] for iu in range(nu)], float)
G[:, :, 0] += 0.2                                   # off centre: not symmetric to start with
srf = ext.surface_from_cvs(3, 3, [tuple(p) for p in G.reshape(-1, 3)], geom.clamped_uniform_knots(nu, 3),
                           geom.clamped_uniform_knots(nv, 3), None)
S = store.new_object(ctx, "SymSurf", srf)
S.location = (0.0, 0.5, 0.0)
ctx.view_layer.update()
select(S)
r = run_mirror(mirror_output='SYMMETRIC', center_point=(0, 0, 0), frame_z=(1, 0, 0), frame_x=(0, 1, 0))
check("Make Symmetric runs on a surface", r == {'FINISHED'})
Gs, d = surf_grid(S)
sy = symmetry.sym_of(S)
check("surface: the rows are paired along U", sy.get("dir") == "U")
Gw = Gs.reshape(-1, 3) @ W(S)[:3, :3].T + W(S)[:3, 3]
tw = symmetry.grid_twins(nu, nv, "U")
err = np.abs(Gw[tw] - mirror_world(Gw, (0, 0, 0), (1, 0, 0)).T).max()
check("surface: CV row nu-1-i mirrors row i (float32 cage)", err < 1e-5, err)
check("surface: the master rows kept", np.abs(Gs[:3] - G[:3]).max() < 1e-5)
Pw, Nw = surf_samples(S)
dist, _ = nearest(mirror_world(Pw, (0, 0, 0), (1, 0, 0)).T, Pw)
check("surface: its mirror image lies on itself", dist.max() < 1e-4, dist.max())

bpy.ops.nproto.edit_cvs()
cage = ctx.active_object
check("cage made for the symmetric surface", cvs.is_cage(cage) and cage.mode == 'EDIT')
ctx.view_layer.update()
bm = bmesh.from_edit_mesh(cage.data)
bm.verts.ensure_lookup_table()
for v in bm.verts:
    v.select = False
k = 1 * nv + 2
bm.verts[k].select = True
bm.verts[k].co = bm.verts[k].co + Vector((0.2, 0.1, 0.6))
bmesh.update_edit_mesh(cage.data)
ctx.view_layer.update()
ctx.view_layer.update()
bm = bmesh.from_edit_mesh(cage.data)
bm.verts.ensure_lookup_table()
a = np.array(bm.verts[k].co); b = np.array(bm.verts[int(tw[k])].co)
Ac = np.linalg.inv(W(S)) @ W(cage)
a_s = Ac[:3, :3] @ a + Ac[:3, 3]; b_s = Ac[:3, :3] @ b + Ac[:3, 3]
check("cage: moving a CV moves its twin row's CV", np.abs(b_s - symmetry.reflect(sy, a_s[None])[0]).max() < 1e-5)
check("cage: the selected CV's twin is drawn", len(symmetry.twins_world(cage)) == 1)
bpy.ops.nproto.cvs_done()
Gs, _ = surf_grid(S)
Gw = Gs.reshape(-1, 3) @ W(S)[:3, :3].T + W(S)[:3, 3]
err = np.abs(Gw[tw] - mirror_world(Gw, (0, 0, 0), (1, 0, 0)).T).max()
check("cage Done: the surface stays symmetric", err < 1e-5, err)


# --- Mirror Copy: a surface, a solid, a curve --------------------------------
bpy.ops.nproto.add_primitive(kind="PLANE", width=1.0, depth=0.6)
PL = ctx.active_object
PL.location = (1.0, 2.0, 0.5)
PL.rotation_euler = (0.3, 0.0, 0.7)
ctx.view_layer.update()
select(PL)
r = run_mirror(mirror_output='COPY', center_point=(0, 0, 0), frame_z=(0, 1, 0), frame_x=(1, 0, 0))
cps = [o for o in bpy.data.objects if o.get("nproto_in_0") is PL]
check("Mirror Copy makes one copy", r == {'FINISHED'} and len(cps) == 1)
cp = cps[0]
check("the copy's matrix is proper (no negative scale)", np.linalg.det(W(cp)[:3, :3]) > 0)
Ps, Ns = surf_samples(PL)
Pc, Nc = surf_samples(cp)
T = pattern.mirror_matrix((0, 0, 0), (0, 1, 0))
dist, j = nearest(Ps @ T[:3, :3].T + T[:3, 3], Pc)
check("the copy is the mirror image", dist.max() < 1e-6, dist.max())
check("the copy faces out as the source does (normal = mirrored normal)",
      np.abs(Nc[j] - Ns @ T[:3, :3].T).max() < 1e-4, np.abs(Nc[j] - Ns @ T[:3, :3].T).max())
check("a mirror copy is a pattern copy (history)", pattern.pattern_of(cp) is not None
      and pattern.pattern_of(cp)[1]["mode"] == "MIRROR")
# it follows the source
PL.location = (1.5, 2.5, 0.0)
ctx.view_layer.update()
features.rebuild(cp)
Ps, Ns = surf_samples(PL)
Pc, Nc = surf_samples(cp)
dist, j = nearest(Ps @ T[:3, :3].T + T[:3, 3], Pc)
check("the mirror copy follows the moved source", dist.max() < 1e-6, dist.max())
check("... still facing out", np.abs(Nc[j] - Ns @ T[:3, :3].T).max() < 1e-4)
# Edit Mirror: the plane moved along its normal
grp = pattern.pattern_of(cp)[1]["group"]
bpy.ops.nproto.mirror(group=grp, mirror_offset=0.5, center_point=(0, 0, 0), frame_z=(0, 1, 0), frame_x=(1, 0, 0))
cps = [o for o in bpy.data.objects if o.get("nproto_in_0") is PL]
T5 = pattern.mirror_matrix((0, 0.5, 0), (0, 1, 0))
Pc, _ = surf_samples(cps[0])
dist, _ = nearest(Ps @ T5[:3, :3].T + T5[:3, 3], Pc)
check("Edit Mirror moves the plane: still one copy, mirrored across y = 0.5", len(cps) == 1 and dist.max() < 1e-6,
      dist.max())
check("summary names it a mirror", pattern.summary(pattern.pattern_of(cps[0])[1]).startswith("Mirror"))

# a solid (trimmed faces): the copy's faces point out of it
bpy.ops.nproto.add_primitive(kind="BOX")
BX = ctx.active_object
BX.location = (0.4, 2.0, 0.3)
BX.rotation_euler = (0.2, 0.1, 0.4)
ctx.view_layer.update()
select(BX)
run_mirror(mirror_output='COPY', center_point=(0, 0, 0), frame_z=(0, 1, 0), frame_x=(1, 0, 0))
bc = [o for o in bpy.data.objects if o.get("nproto_in_0") is BX][0]
def outward(o):
    """Per face: a world point on it and the kernel's outward normal there."""
    M = W(o)
    P, N = [], []
    for f in store.get_brep(o).faces():
        s = f.surface()
        (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
        u, v = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        p = np.array(s.evaluate(u, v), float)
        nn = M[:3, :3] @ np.array(f.outward_normal_at_uv(u, v), float)
        P.append(M[:3, :3] @ p + M[:3, 3]); N.append(nn / np.linalg.norm(nn))
    return np.array(P), np.array(N)
Ps, Ns = outward(BX)
Pc, Nc = outward(bc)
dist, j = nearest(Ps @ T[:3, :3].T + T[:3, 3], Pc)
check("solid: the copy is the mirror image", dist.max() < 1e-6, dist.max())
out_s = np.sum((Ps - Ps.mean(axis=0)) * Ns, axis=1)
out_c = np.sum((Pc - Pc.mean(axis=0)) * Nc, axis=1)
check("solid: the source's faces point out (the test's own check)", (out_s > 0).all())
check("solid: the copy's faces point out", (out_c > 0).all(), out_c)
check("solid: the copy's outward normals are the mirrored ones", np.abs(Nc[j] - Ns @ T[:3, :3].T).max() < 1e-6)
check("solid: the copy is still one solid body", store.get_brep(bc).face_count() == store.get_brep(BX).face_count())

# a trimmed open sheet (turned over in the worker)
bpy.ops.nproto.add_primitive(kind="DISK")
DK = ctx.active_object
DK.location = (0.5, 1.5, 0.2)
DK.rotation_euler = (0.4, 0.2, 0.0)
ctx.view_layer.update()
check("disk: a trimmed sheet", not store.is_natural(DK) and not store.get_brep(DK).is_manifold_solid())
select(DK)
run_mirror(mirror_output='COPY', center_point=(0, 0, 0), frame_z=(0, 1, 0), frame_x=(1, 0, 0))
dc = [o for o in bpy.data.objects if o.get("nproto_in_0") is DK][0]
Ps, Ns = outward(DK)
Pc, Nc = outward(dc)
dist, j = nearest(Ps @ T[:3, :3].T + T[:3, 3], Pc)
check("disk: the copy is the mirror image", dist.max() < 1e-6, dist.max())
check("disk: the copy faces out as the source does", np.abs(Nc[j] - Ns @ T[:3, :3].T).max() < 1e-6,
      np.abs(Nc[j] - Ns @ T[:3, :3].T).max())

# a curve
CC = nurbs("MirCurve", [(0.2, 0.5, 0), (1.0, 1.2, 0.4), (2.0, 0.8, 0.9), (2.5, 2.0, 0.0)])
CC.location = (0.3, 0.2, 0.0)
ctx.view_layer.update()
select(CC)
run_mirror(mirror_output='COPY', center_point=(0, 0, 0), frame_z=(0, 1, 0), frame_x=(1, 0, 0))
cc = [o for o in bpy.data.objects if o.get("nproto_in_0") is CC][0]
check("curve: Mirror Copy mirrors its CVs", np.abs(cvs_world(cc) - mirror_world(cvs_world(CC), (0, 0, 0), (0, 1, 0)).T).max() < 1e-6)
check("curve copy: matrix proper", np.linalg.det(W(cc)[:3, :3]) > 0)


# --- the widget's default plane --------------------------------------------
class _Op:
    mirror_output = "COPY"
half = nurbs("Half", [(0.0, 0.0, 0.0), (1.0, 0.5, 0.0), (2.0, 0.4, 0.0), (3.0, 0.0, 0.0)])
half.location = (0.0, 3.0, 0.0)          # touches nothing but x = 0 ... and lies above y = 3
ctx.view_layer.update()
op = _Op()
pattern.NPROTO_OT_pattern._mirror_widget(op, None, [half])
check("Mirror Copy starts on the plane the object touches (x = 0)", np.allclose(op.frame_z, (1, 0, 0)), op.frame_z)
op.mirror_output = "SYMMETRIC"
cen = nurbs("Centred", [(-1.0, 0.2, 0.0), (0.0, 0.8, 0.5), (1.0, 0.2, 0.0)])
cen.location = (0.0, 4.0, 2.0)
ctx.view_layer.update()
pattern.NPROTO_OT_pattern._mirror_widget(op, None, [cen])
check("Make Symmetric starts on the plane the object is centred on (x = 0)", np.allclose(op.frame_z, (1, 0, 0)),
      op.frame_z)


# --- the tool's stage, run without a window (its widget, arrow, preview, hints, keys)
from blender_nurbs import draw as _draw
class Stage(pattern._PatternTool):
    pass
src = nurbs("StageCurve", [(-1.0, 0.5, 0.3), (0.0, 1.5, 0.9), (1.2, 2.5, 0.5), (2.0, 3.0, 0.3)])   # off z = 0
src.location = (0.0, 0.2, 0.0)
ctx.view_layer.update()
st = Stage()
st.mode, st.mirror_output, st.flip_master, st.mirror_offset, st.turn = "MIRROR", "COPY", False, 0.0, 0.0
st.space, st.objects, st.group, st.path_obj, st.path_ref, st.surfaces = "SPACING", "", "", "", "", ""
st.center_point, st.frame_x, st.frame_z = (0, 0, 0), (1, 0, 0), (0, 0, 1)
st._ctx, st._cdrag, st._field, st._typed, st._value_mode, st._cbuf = ctx, None, 0, "", False, ""
st._moved_center, st._picking_axis, st._ptmp = False, False, {}
M_ = pattern.world_matrix(src)
st._pv = [(src, pattern._preview_lines(src, M_), pattern._preview_tris(src, M_), pattern._box(src, M_))]
st._default_widget(ctx, [src])
check("stage: the widget starts on y = 0 (the curve's box comes nearest it)", np.allclose(st.frame_z, (0, 1, 0)),
      st.frame_z)
check("stage: one field, the plane offset", st.drag_fields == ("mirror_offset",))
h = st.drag_handles()
check("stage: the arrow starts at the plane, along its normal", len(h) == 1 and np.allclose(h[0]["dir"], st.frame_z)
      and abs(h[0]["to_value"](h[0]["param"]) - st.mirror_offset) < 1e-12)
st.mirror_offset = 0.7
h = st.drag_handles()
check("stage: dragging the arrow tip moves the offset 1:1", abs(h[0]["to_value"](h[0]["param"] + 0.3) - 1.0) < 1e-12)
st.mirror_offset = 0.0
pv = st.drag_preview()
nl = len(st._pv[0][1])
check("stage, copy: the preview is the mirrored curve", len(pv["lines"]) == nl and
      np.abs(pv["lines"][0] - (st._pv[0][1][0] * np.array([1, -1, 1]))).max() < 1e-9)
st._widget_overlay(ctx)
refs = _draw.sketch_overlay["refs"]
check("stage: the overlay draws the plane and the arrow", len(refs) == 3 and len(refs[0][0]) == 5)
check("stage: hint rows and state words", st._rows()[0][0] == "Output" and st._state(ctx)[0] == "Mirror Copy")
check("stage: S switches to Make Symmetric", st._mirror_key(ctx, "s") and st.mirror_output == "SYMMETRIC")
pv = st.drag_preview()
check("stage, symmetric: the preview has the master half and its mirror", len(pv["lines"]) >= 2
      and len(pv["lines"]) % 2 == 0)
check("stage: F flips the master side", st._mirror_key(ctx, "f") and st.flip_master)
check("stage: X puts the normal along X", st._mirror_key(ctx, "x") and np.allclose(st.frame_z, (1, 0, 0)))
check("stage: W puts the plane through the world origin", st._mirror_key(ctx, "w") and np.allclose(st.center_point, 0))
check("stage: the pattern's own keys do nothing here", st._mirror_key(ctx, "m") and st.mode == "MIRROR")
check("stage: nothing to refuse", st.drag_check() == "")
check("stage: state says the master side", "master: the end" in st._state(ctx))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
