"""Tween: copies of an object stepped from where it is to where a second
object (the same shape, moved / turned / scaled) is, and on past it.

Select the two objects, the target (B) last so it is active: the copies
are the first (A) stepped evenly towards B -- each step the same
transform, so they run along the line, arc or helix that the move from A
to B makes (a scaled one spirals), whatever the objects' origins.
Continue adds copies past B with the same step; Long Way turns the other
way round (a 90° turn goes as 270°).

NURBS objects and curves come out as a Pattern (pattern.py, mode TWEEN)
with B as an input of the history: move, turn or scale B (or A) and the
copies follow; Edit Pattern changes the counts. Other objects are
duplicated in place (they do not follow).

The step from A to B is a fractional power of the transform B·A⁻¹: its
rotation by axis and angle, its scale by power, and the move in the
screw / spiral form that makes every step the same. A scale that differs
from A's along some axes (a stretched B) is stepped too, though a copy of
a turned object may then need a shear its transform can't hold (it is
reported).
"""

import math

import bpy
import numpy as np
from bpy.props import BoolProperty, IntProperty, StringProperty

from . import pattern

MAX_COUNT = 500


# --------------------------------------------------------------------------
# The maths (tested headless)
# --------------------------------------------------------------------------

def _translation(v):
    T = np.eye(4)
    T[:3, 3] = v
    return T


def _rodrigues(u, a):
    """3x3: a radians about the unit axis u."""
    u = np.asarray(u, float)
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def axis_angle(R):
    """(unit axis, angle in [0, pi]) of the rotation R."""
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < math.pi - 1e-4:
        return w / np.linalg.norm(w), th
    B = 0.5 * (R + np.eye(3))                   # near a half turn: B = u uᵀ
    i = int(np.argmax(np.diag(B)))
    u = B[:, i] / math.sqrt(max(B[i, i], 1e-300))
    u = u / np.linalg.norm(u)
    if float(w @ u) < 0.0:
        u = -u
    return u, th


def _g(alpha, L):
    """(λ^α - 1) / (λ - 1) for λ = e^L (real or complex): how far the move
    goes in α of a step that also turns / scales by λ (α when λ = 1)."""
    if abs(L) < 1e-12:
        return alpha
    return np.expm1(alpha * L) / np.expm1(L)


def step_transform(Wa, Wb, alpha, c, long_way=False):
    """World 4x4 that takes A (world matrix Wa) α of the way to B (Wb):
    α = 1 gives Wb·Wa⁻¹ exactly, α = 2 one step further. c: the point the
    stretch (a scale other than uniform) is stepped about -- A's middle."""
    T = np.asarray(Wb, float) @ np.linalg.inv(np.asarray(Wa, float))
    c = np.asarray(c, float)
    Tc = _translation(-c) @ T @ _translation(c)
    M, t = Tc[:3, :3], Tc[:3, 3]
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        raise ValueError("the second object is a mirror image of the first: it can't be stepped to")
    u, th = axis_angle(R)
    if long_way and th > 1e-9:
        th -= 2.0 * math.pi
    Ma = _rodrigues(u, alpha * th) @ (Vt.T @ np.diag(S ** alpha) @ Vt)
    ls = math.log(float(np.prod(S)) ** (1.0 / 3.0))
    t_par = float(t @ u) * u
    t_perp = t - t_par
    ta = _g(alpha, ls) * t_par
    r = float(np.linalg.norm(t_perp))
    if r > 1e-300:
        e1 = t_perp / r
        e2 = np.cross(u, e1)
        f = complex(_g(alpha, complex(ls, th)))
        ta = ta + r * (f.real * e1 + f.imag * e2)
    Ta = np.eye(4)
    Ta[:3, :3] = Ma
    Ta[:3, 3] = ta
    return _translation(c) @ Ta @ _translation(-c)


def alphas(count, extra):
    """How far along (1 = on B) each copy is: count between A and B, extra past B."""
    n = int(count) + 1
    return [k / n for k in range(1, n)] + [(n + j) / n for j in range(1, int(extra) + 1)]


def alpha_of(p, index):
    """The Pattern copy `index`'s α (None: no such copy)."""
    a = alphas(p.get("count", 3), p.get("extra", 0))
    return a[index - 1] if 0 < index <= len(a) else None


def copy_transform(p, src, target, index):
    """Pattern's TWEEN copy: the world 4x4 that moves src to copy index."""
    a = alpha_of(p, index)
    if a is None or target is None:
        return None
    return step_transform(pattern.world_matrix(src), pattern.world_matrix(target), a, pattern.anchor(src),
                          bool(p.get("long_way", False)))


def sheared(F, tol=1e-6):
    """A world matrix whose axes are not square to each other (an object can't hold it)."""
    A = np.asarray(F, float)[:3, :3]
    n = np.linalg.norm(A, axis=0)
    if np.any(n <= 1e-300):
        return True
    G = (A / n).T @ (A / n)
    return float(np.abs(G - np.eye(3)).max()) > tol


def same_shape(a, b, tol=1e-6):
    """Do a and b hold the same shape in their own spaces (their boxes match)?"""
    pa, pb = pattern._local_points(a), pattern._local_points(b)
    if pa is None or pb is None or not len(pa) or not len(pb):
        return a.type == b.type
    ba = np.array([pa.min(axis=0), pa.max(axis=0)])
    bb = np.array([pb.min(axis=0), pb.max(axis=0)])
    return float(np.abs(ba - bb).max()) <= tol * max(1.0, float(np.abs(ba).max()))


# --------------------------------------------------------------------------
# The operator
# --------------------------------------------------------------------------

class NPROTO_OT_tween(bpy.types.Operator):
    """Copies of the first selected object stepped evenly to the second
    (the active one: the same shape moved, turned or scaled), and on past
    it -- along the line, arc or spiral between them. NURBS copies follow
    both objects through history"""
    bl_idname = "nproto.tween"
    bl_label = "Tween"
    bl_options = {'REGISTER', 'UNDO'}

    count: IntProperty(name="Between", default=3, min=0, soft_max=50, max=MAX_COUNT,
                       description="Copies between the two objects")
    extra: IntProperty(name="Continue", default=0, min=0, soft_max=50, max=MAX_COUNT,
                       description="Copies past the second object, each one more step on")
    long_way: BoolProperty(name="Long Way Round", default=False,
                           description="Turn the other way round from the first to the second "
                                       "(a 90° turn goes as 270°)")
    swap: BoolProperty(name="Swap", default=False,
                       description="Step from the active object to the other one instead")
    group: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="Edit Pattern: the tween's pattern group")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        lay.prop(self, "count")
        lay.prop(self, "extra")
        lay.prop(self, "long_way")
        if not self.group:
            lay.prop(self, "swap")

    def _pair(self, context):
        """(A, B) or a reason why not."""
        if self.group:
            copies = pattern.group_copies(self.group)
            if not copies:
                return None, "that tween has no copies left"
            o = copies[0][0]
            a, b = o.get("nproto_in_0"), o.get("nproto_in_1")
            if a is None or b is None:
                return None, "that tween's objects are gone"
            return (a, b), ""
        sel = list(context.selected_objects)
        if len(sel) != 2:
            return None, f"select two objects, the target last ({len(sel)} selected)"
        act = context.view_layer.objects.active
        b = act if act in sel else sel[1]
        a = sel[0] if sel[1] is b else sel[1]
        return ((b, a) if self.swap else (a, b)), ""

    def invoke(self, context, event):
        if self.group:
            copies = pattern.group_copies(self.group)
            pg = pattern.pattern_of(copies[0][0]) if copies else None
            if pg is not None:
                p = pg[1]
                self.count = int(p.get("count", 3))
                self.extra = int(p.get("extra", 0))
                self.long_way = bool(p.get("long_way", False))
        return self.execute(context)

    def execute(self, context):
        pair, why = self._pair(context)
        if pair is None:
            self.report({'WARNING'}, "Tween: " + why)
            return {'CANCELLED'}
        a, b = pair
        al = alphas(self.count, self.extra)
        if not al:
            self.report({'WARNING'}, "Tween: no copies (Between and Continue are 0)")
            return {'CANCELLED'}
        Wa, Wb = pattern.world_matrix(a), pattern.world_matrix(b)
        c = pattern.anchor(a)
        try:
            mats = [step_transform(Wa, Wb, x, c, self.long_way) for x in al]
        except ValueError as exc:
            self.report({'WARNING'}, f"Tween: {exc}")
            return {'CANCELLED'}
        if np.abs(Wb - Wa).max() <= 1e-9 * max(1.0, float(np.abs(Wa).max())):
            self.report({'WARNING'}, "Tween: the two objects sit in the same place")
            return {'CANCELLED'}
        made, errors = [], []
        if pattern.usable(a):
            if store_needed(a):
                from .ops import _kernel_ok
                if not _kernel_ok(self):
                    return {'CANCELLED'}
            p = {"v": 2, "mode": "TWEEN", "count": int(self.count), "extra": int(self.extra),
                 "long_way": bool(self.long_way)}
            made, errors = pattern.make_pattern(context, [a], p, self.group or None, path=(b, "TWEEN"))
        else:
            from .movetool import _duplicate
            for T in mats:
                try:
                    made.append(_duplicate(context, a, T))
                except Exception as exc:
                    errors.append(str(exc))
        for e in errors:
            self.report({'ERROR'}, "Tween: " + e)
        if not made:
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        for o in [a, b] + made:
            try:
                o.select_set(True)
            except RuntimeError:
                pass
        context.view_layer.objects.active = b
        notes = []
        if any(sheared(T @ Wa) for T in mats):
            notes.append("a copy needs a shear its transform can't hold (B is stretched against A's turn): "
                         "placed as near as it allows")
        if not same_shape(a, b):
            notes.append(f"{b.name} is not shaped like {a.name}: the copies take {a.name}'s shape")
        n_between = min(int(self.count), len(made))
        txt = f"Tween: {len(made)} cop{'y' if len(made) == 1 else 'ies'} from {a.name} to {b.name}" + \
              (f" ({n_between} between, {len(made) - n_between} past)" if self.extra else "")
        self.report({'WARNING'} if notes else {'INFO'}, txt + ("; " + "; ".join(notes) if notes else ""))
        return {'FINISHED'}


def store_needed(obj):
    """A NURBS object (not a curve): its copies need the kernel."""
    return obj.type != 'CURVE'


def menu_tween(self, context):
    self.layout.operator(NPROTO_OT_tween.bl_idname, icon='IPO_EASE_IN_OUT')


classes = (NPROTO_OT_tween,)


def register():
    bpy.types.VIEW3D_MT_object.append(menu_tween)


def unregister():
    try:
        bpy.types.VIEW3D_MT_object.remove(menu_tween)
    except Exception:
        pass
