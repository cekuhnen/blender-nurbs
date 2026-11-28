"""Remove Fillets: a solid's rounded edges taken away, the faces round them
extended until they meet in sharp edges again (defeaturing, as Fusion's
Simplify, NX's Delete Face of blends or Shapr3D's face delete do).

One click: Remove All Fillets on the selected solids (no larger than Max
Radius, when set). Or start Remove Fillets: every fillet the tool finds
lights up red (all to be removed); click a fillet to keep it (it dims) or to
take it back, A for all / none, Enter to remove the red ones. Faces picked in
Face mode first: those fillets are removed at once.

A chain (fillets joined at rounded corners) goes as a whole; the fillets of
it that were not picked are put back on the new sharp edges with the radius
the kernel read off them (Keep Others). The full round of a slot's end and
the faces of a free-form skin are no fillets and stay (kernel/defeature.py
says why for each).

With history the removal is a step on the body (remove_fillets: all, or the
fillets found again by a point on each); without it the body changes in
place. Kernel side: kernel/defeature.py, worker job remove_fillets."""

import json

import bpy
import numpy as np
from bpy.props import BoolProperty, FloatProperty, StringProperty
from mathutils import Vector

from . import draw, features, geom, hints, jobs, store, subsel
from .ops import _is_solid, _kernel_ok, _select_only
from .pushpull import _FacePicking, _faces_json, face_frame, face_highlight
from .shell import _segments


def _defeature():
    from .kernel import defeature
    return defeature


def kernel_has_it():
    try:
        sm, _ = store._k()
    except Exception:
        return False
    return _defeature().has_kernel(sm)


NO_KERNEL = ("This kernel has no fillet removal yet: restart Blender after updating the add-on "
             "(a running Blender keeps the kernel it loaded)")


def _scale(o):
    return float(max(geom.world_matrix(o).to_scale())) or 1.0


def find(o):
    """defeature.find on body o (its local space), radii in world units."""
    sm, _ = store._k()
    found = _defeature().find(sm, store.get_brep(o))
    s = _scale(o)
    for f in found["fillets"]:
        for k in ("radius", "radius_min", "radius_max"):
            f[k] *= s
    return found


def _fmt_radius(f):
    a, b = f["radius_min"], f["radius_max"]
    if f["solver"] == 2 and abs(b - a) > 1e-9 * max(b, 1.0):
        return f"r {a:.4g}–{b:.4g}"
    return f"r {f['radius']:.4g}"


class NPROTO_OT_remove_fillets(_FacePicking, bpy.types.Operator):
    """Remove fillets from a solid: the faces round each rounded edge are
    extended until they meet in a sharp edge again. Start it: every fillet
    found lights up; click the ones to keep, Enter. Faces picked in Face
    mode: those fillets go at once. Remove All (one click): every fillet of
    the selected solids"""
    bl_idname = "nproto.remove_fillets"
    bl_label = "Remove Fillets"
    bl_options = {'REGISTER', 'UNDO'}

    faces: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="The fillets to remove: [[body name, a point on the face (local)], ...]")
    all_fillets: BoolProperty(name="All Fillets", default=False, options={'SKIP_SAVE'},
                              description="Every fillet of the selected solids (no larger than Max Radius)")
    max_radius: FloatProperty(name="Max Radius", default=0.0, min=0.0, unit='LENGTH',
                              description="With All Fillets: only fillets no larger than this (0: any size)")
    keep_others: BoolProperty(name="Keep Others", default=True,
                              description="A fillet that shares a rounded corner with removed ones is put back "
                                          "on the new sharp edge (off: the whole chain goes)")

    # -- the inputs -----------------------------------------------------------
    @staticmethod
    def _check(o):
        if not store.is_nurbs(o) or not _is_solid(o):
            return f"{o.name} is not a closed solid"
        return ""

    def _items(self):
        """{body name: (body, [local points])}"""
        out = {}
        for name, p in json.loads(self.faces or "[]"):
            o = bpy.data.objects.get(name)
            if o is None:
                raise ValueError(f"{name} not found")
            out.setdefault(name, (o, []))[1].append(np.array(p, float))
        return out

    def _solids(self, context):
        return [o for o in context.selected_objects if store.is_nurbs(o) and _is_solid(o)]

    def _take_picks(self, context):
        if self.faces:
            return
        picked = []
        for o in context.selected_objects:
            if store.is_nurbs(o):
                picked += [(o, face_frame(o, i)[0]) for i in subsel.picks(o, "FACE")]
        if picked:
            self.faces = _faces_json(picked)

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if not kernel_has_it():
            self.report({'ERROR'}, NO_KERNEL)
            return {'CANCELLED'}
        if self.all_fillets:
            return self.execute(context)
        self._take_picks(context)
        if self.faces or bpy.app.background:
            return self.execute(context)
        self._hover = None
        self._found = {}                # body name: find() result
        self._on = {}                   # body name: {fillet face index} to remove
        for o in self._solids(context):
            self._add_body(o, all_on=True)
        context.window_manager.modal_handler_add(self)
        self._show()
        self._pick_hints(context)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    # -- picking ----------------------------------------------------------------
    def _add_body(self, o, all_on=False):
        if o.name in self._found:
            return self._found[o.name]
        try:
            found = find(o)
        except Exception as exc:
            print("NURBS remove fillets:", exc)
            found = {"fillets": [], "corners": [], "chains": [], "skipped": []}
        self._found[o.name] = found
        self._on[o.name] = {f["face"] for f in found["fillets"]} if all_on else set()
        return found

    def _counts(self):
        n_on = sum(len(v) for v in self._on.values())
        n_all = sum(len(f["fillets"]) for f in self._found.values())
        return n_on, n_all

    def _pick_hints(self, context, msg=""):
        n_on, n_all = self._counts()
        if not self._found:
            state = ["click a face of a solid"]
        elif not n_all:
            state = ["no fillets found on " + ", ".join(self._found)]
        else:
            radii = [f["radius_max"] for x in self._found.values() for f in x["fillets"]]
            state = [f"{n_on} of {n_all} fillet{'s' if n_all != 1 else ''} to remove",
                     f"r {min(radii):.4g}" + (f"–{max(radii):.4g}" if max(radii) > min(radii) * (1 + 1e-6) else "")]
        hints.show(context, self.bl_label, state + ([msg] if msg else []),
                   [("", [hints.note("red: removed · dim: kept. A chain (fillets meeting at a rounded corner) "
                                     "is removed whole; the kept fillets of it are put back"),
                          hints.key("A", "all / none"),
                          hints.key("K", "Keep Others: the kept fillets of a chain are put back", self.keep_others)])],
                   [("MOUSE_LMB", "Keep / remove a fillet"), ("EVENT_RETURN", "Remove"), ("EVENT_ESC", "Cancel")])

    def _show(self):
        regs = []
        for name, found in self._found.items():
            o = bpy.data.objects.get(name)
            if o is None:
                continue
            on = self._on.get(name, set())
            chains_on = {f["chain"] for f in found["fillets"] if f["face"] in on}
            items = [(f["face"], f["face"] in on) for f in found["fillets"]]
            items += [(c["face"], c["chain"] in chains_on) for c in found["corners"]]
            for i, removed in items:
                try:
                    h = face_highlight(o, i, self.MAX_FILL)
                except Exception as exc:
                    print("NURBS remove fillets highlight:", exc)
                    continue
                # float32: the GPU vertex buffer misreads float64 rows of a curved face's fill (the
                # triangles came out huge and flooded the view red; a flat face's few happened to pass)
                regs.append((np.ascontiguousarray(np.asarray(h["tris"], float).reshape(-1, 3, 3), np.float32),
                             np.ascontiguousarray(_segments(h["lines"]), np.float32), not removed))
        draw.pick_overlay["regions"] = regs

    def _end_pick(self, context):
        draw.pick_overlay["regions"] = []
        super()._end_pick(context)

    def _describe(self, o, i):
        found = self._found.get(o.name)
        if found is None:
            return ""
        for f in found["fillets"]:
            if f["face"] == i:
                n = len(found["chains"][f["chain"]])
                return f"{f['kind']} fillet, {_fmt_radius(f)}" + (f" · chain of {n} faces" if n > 1 else "")
        for c in found["corners"]:
            if c["face"] == i:
                return "a rounded corner (goes with its fillets)"
        for s in found["skipped"]:
            if s["face"] == i:
                return "not removed: " + s["why"]
        return "not a fillet"

    def modal(self, context, event):
        if self._view_event(context, event):
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE':
            hit = self._hit(context, event)
            if hit != self._hover:
                self._show_face(hit)
                msg = ""
                if hit is not None and not self._check(hit[0]):
                    self._add_body(hit[0], all_on=False)
                    msg = self._describe(*hit)
                self._pick_hints(context, msg)
            context.area.tag_redraw()
            return {'PASS_THROUGH'}
        if event.type in ('ESC', 'RIGHTMOUSE') and event.value == 'PRESS':
            self._end_pick(context)
            return {'CANCELLED'}
        if event.type == 'A' and event.value == 'PRESS':
            n_on, n_all = self._counts()
            for name, found in self._found.items():
                self._on[name] = set() if n_on == n_all else {f["face"] for f in found["fillets"]}
            self._show()
            self._pick_hints(context)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'K' and event.value == 'PRESS':
            self.keep_others = not self.keep_others
            self._pick_hints(context)
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = self._hit(context, event)
            if hit is None:
                self._pick_hints(context, "no face there")
                return {'RUNNING_MODAL'}
            o, i = hit
            why = self._check(o)
            if why:
                self._pick_hints(context, why)
                return {'RUNNING_MODAL'}
            found = self._add_body(o)
            on = self._on[o.name]
            fil = {f["face"]: f for f in found["fillets"]}
            corner = next((c for c in found["corners"] if c["face"] == i), None)
            if i in fil and not event.shift:
                on.symmetric_difference_update({i})
            elif i in fil or corner is not None:           # a corner (or Shift): the whole chain
                ch = fil[i]["chain"] if i in fil else corner["chain"]
                faces = {j for j, f in fil.items() if f["chain"] == ch}
                if faces <= on:
                    on -= faces
                else:
                    on |= faces
            else:
                self._pick_hints(context, self._describe(o, i))
                return {'RUNNING_MODAL'}
            self._show()
            self._pick_hints(context, self._describe(o, i))
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type in ('RET', 'NUMPAD_ENTER', 'SPACE') and event.value == 'PRESS':
            picked = []
            for name, on in self._on.items():
                o = bpy.data.objects.get(name)
                fs = {f["face"]: f for f in self._found[name]["fillets"]}
                picked += [(o, np.array(fs[i]["point"], float)) for i in sorted(on)]
            if not picked:
                self._pick_hints(context, "nothing to remove: click a fillet")
                return {'RUNNING_MODAL'}
            self._end_pick(context)
            self.faces = _faces_json(picked)
            res = self.execute(context)
            return res
        return {'RUNNING_MODAL'}

    # -- making it --------------------------------------------------------------
    def execute(self, context):
        draw.pick_overlay["regions"] = []
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if not kernel_has_it():
            self.report({'ERROR'}, NO_KERNEL)
            return {'CANCELLED'}
        if not self.all_fillets:
            self._take_picks(context)
        try:
            items = self._items() if not self.all_fillets else {}
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, f"Remove Fillets: {exc}")
            return {'CANCELLED'}
        if self.all_fillets or not items:
            if not self.all_fillets and not bpy.app.background:
                self.report({'WARNING'}, "Remove Fillets: pick fillets (Face mode, or click them)")
                return {'CANCELLED'}
            items = {o.name: (o, None) for o in self._solids(context)}
        if not items:
            self.report({'WARNING'}, "Remove Fillets: select a solid")
            return {'CANCELLED'}
        hist = features.history_enabled(context)
        done = []
        for o, pts in items.values():
            why = self._check(o)
            if why:
                self.report({'ERROR'}, f"Remove Fillets: {why}")
                continue
            M = geom.world_matrix(o)
            params = {"keep_others": bool(self.keep_others), "max_radius": float(self.max_radius)}
            if pts is None:
                params["mode"] = "ALL"
                world = None
            else:
                params["mode"] = "PICKED"
                params["faces_local"] = [[float(x) for x in p] for p in pts]
                world = [[float(x) for x in M @ Vector(p)] for p in pts]
            try:
                info, breps = jobs.run("remove_fillets", {"target": o}, ["result"], face_pts=world,
                                       max_radius=params["max_radius"] if pts is None else 0.0,
                                       keep_others=params["keep_others"], into="target")
            except jobs.JobError as exc:
                self.report({'ERROR'}, f"Remove Fillets on {o.name}: {exc}")
                continue
            pre = o[store.PROP_BREP]
            if not hist and features.has_history(o):
                features.clear_history(o)
                self.report({'INFO'}, f"{o.name}: construction history removed")
            store.set_geometry_text(o, breps["result"], context, keep_untrimmed=False)
            if hist:
                features.push_step(o, "remove_fillets", [], params, base_text=pre)
            if subsel.picks(o, "FACE"):
                subsel.set_picks(o, "FACE", [])
            done.append(o)
            n = int(info.get("removed", 0))
            msg = f"{o.name}: {n} fillet{'s' if n != 1 else ''} removed"
            if info.get("restored"):
                msg += f", {info['restored']} put back"
            if info.get("failed"):
                msg += f"; {len(info['failed'])} could not be removed ({info['failed'][0]['why']})"
            for w in info.get("warnings", []):
                msg += f"; {w}"
            self.report({'WARNING'} if info.get("failed") or info.get("warnings") else {'INFO'}, msg)
        if not done:
            return {'CANCELLED'}
        _select_only(context, done)
        return {'FINISHED'}


def apply_remove_fillets(obj, text, inputs, p):
    """Remove Fillets on rebuild: every fillet again (mode ALL), or the ones
    at the stored points."""
    M = geom.world_matrix(obj)
    pts = None
    if p.get("mode") != "ALL":
        pts = [list(M @ Vector(q)) for q in p.get("faces_local", [])]
    try:
        _, breps = jobs.run("remove_fillets", {"target": (text, jobs.xform(obj))}, ["result"], face_pts=pts,
                            max_radius=float(p.get("max_radius", 0.0)) if pts is None else 0.0,
                            keep_others=bool(p.get("keep_others", True)), into="target")
    except jobs.JobError as exc:
        if "no removable fillet" in str(exc):     # (Max Radius below every fillet now): the body as it is
            return text
        raise
    return breps["result"]


def _label(p):
    from . import steps
    if p.get("mode") == "ALL":
        r = float(p.get("max_radius", 0.0))
        return "Remove Fillets · all" + (f" ≤ {steps.length_text(r)}" if r > 0 else "")
    n = len(p.get("faces_local", []))
    return f"Remove Fillets · {n} picked" + ("" if p.get("keep_others", True) else " · whole chains")


def _register_step():
    from . import steps
    steps.APPLY["remove_fillets"] = apply_remove_fillets
    steps.LABEL["remove_fillets"] = _label


_register_step()

classes = (NPROTO_OT_remove_fillets,)
