"""Twist / Bend / Taper with simulated input in a real window (the preview,
the twist handle dragged, typed values, keys, Enter):
    blender -b --factory-startup --python tests/test_deform_gui.py
The background run writes an empty .blend, starts a GUI Blender with
--enable-event-simulate on it (a window opens for ~20 s), and reads back what
that run found. Screenshots go next to the results (the path is printed)."""
import json, math, os, subprocess, sys, tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = "--deform-gui" in sys.argv


def background():
    tmp = tempfile.mkdtemp(prefix="nproto_deform_gui_")
    blend, out = os.path.join(tmp, "empty.blend"), os.path.join(tmp, "results.json")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    cmd = [bpy.app.binary_path, "--factory-startup", "--enable-event-simulate", "--window-geometry", "0", "0",
           "1400", "900", blend, "--python", os.path.abspath(__file__), "--", "--deform-gui", out]
    try:
        subprocess.run(cmd, timeout=240, capture_output=True)
    except subprocess.TimeoutExpired:
        print("FAIL the GUI run did not finish in 240 s")
        os._exit(1)
    if not os.path.isfile(out):
        print("FAIL the GUI run wrote no results")
        os._exit(1)
    res = json.load(open(out))
    ok = True
    for name, (good, extra) in res["checks"].items():
        print(("PASS " if good else "FAIL ") + name, extra if not good else extra)
        ok &= bool(good)
    if res.get("error"):
        print("FAIL the GUI run raised\n" + res["error"])
        ok = False
    print("screenshots in", tmp)
    print("ALL OK" if ok else "SOME FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


def gui():
    import traceback
    import numpy as np
    from mathutils import Euler, Vector
    from bpy_extras import view3d_utils
    sys.path.insert(0, ROOT)
    import blender_nurbs
    blender_nurbs.register()
    from blender_nurbs import deform, draw, features, store

    out = sys.argv[sys.argv.index("--deform-gui") + 1]
    shots = os.path.dirname(out)
    checks = {}
    res = {"checks": checks}

    def check(name, cond, extra=""):
        checks[name] = (bool(cond), str(extra))

    ctx = bpy.context
    win = ctx.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    over = lambda: ctx.temp_override(window=win, area=area, region=region)

    def wpx(P):
        q = view3d_utils.location_3d_to_region_2d(region, area.spaces.active.region_3d, Vector(P))
        return int(round(region.x + q[0])), int(round(region.y + q[1]))

    at = [0, 0]

    def move(p):
        for k in (3, 2, 1, 0):
            win.event_simulate('MOUSEMOVE', 'NOTHING', x=p[0] + k, y=p[1] + k)
        at[:] = p

    def key(t, uni=None):
        kw = {"unicode": uni} if uni else {}
        win.event_simulate(t, 'PRESS', x=at[0], y=at[1], **kw)
        win.event_simulate(t, 'RELEASE', x=at[0], y=at[1])

    def typed(text):
        names = {"0": 'ZERO', "1": 'ONE', "2": 'TWO', "3": 'THREE', "4": 'FOUR', "5": 'FIVE', "6": 'SIX',
                 "7": 'SEVEN', "8": 'EIGHT', "9": 'NINE', ".": 'PERIOD'}
        for ch in text:
            key(names[ch], ch)

    def shot(name):
        area.tag_redraw()
        with over():
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        with ctx.temp_override(window=win, screen=win.screen):
            bpy.ops.screen.screenshot(filepath=os.path.join(shots, name + ".png"))

    modal = lambda: [op.bl_idname for op in win.modal_operators]
    running = lambda: "NPROTO_OT_deform" in modal()
    vol = lambda o: abs(store.get_brep(o).volume(1e-6))
    preview = lambda: draw.pick_overlay.get("preview") or {}

    def last_step(o):
        st = features.stack(o)
        return st[-1]["params"] if st and st[-1]["op"] == "deform" else None

    def steps():
        ctx.scene.nproto_settings.construction_history = True
        with over():
            bpy.ops.nproto.add_primitive(kind='BOX', width=4.0, depth=1.0, height=0.5)
        bar = ctx.view_layer.objects.active
        bar.name = "Bar"
        lo, hi = (np.array(v, float) for v in store.get_brep(bar).bounding_box())
        c = 0.5 * (lo + hi)
        r3 = area.spaces.active.region_3d
        r3.view_perspective = 'PERSP'
        r3.view_rotation = Euler((1.1, 0.0, 0.5)).to_quaternion()
        r3.view_location = Vector(c)
        r3.view_distance = 9.0
        yield 0.5
        with over():
            bpy.ops.ed.undo_push(message="setup")
        v0 = vol(bar)

        # -- Twist: the preview, the handle dragged round the axis, Enter
        with over():
            bpy.ops.object.select_all(action='DESELECT')
            bar.select_set(True)
            ctx.view_layer.objects.active = bar
            bpy.ops.nproto.deform('INVOKE_DEFAULT', kind='TWIST', angle=90.0)
        yield 0.5
        check("Twist starts its value stage", running(), modal())
        pv = preview()
        T = np.asarray(pv.get("tris", np.zeros((0, 3))), float)
        check("the preview shows the twisted bar (filled and its edges)", len(T) > 30 and len(pv.get("lines", [])) >= 12,
              (len(T), len(pv.get("lines", []))))
        fm = deform.map_for(deform.points_of_brep(store.get_brep(bar)), {"deform_kind": "TWIST", "deform_angle": 90.0})
        far = T[T[:, 0] > c[0] + 1.9]
        check("the preview's far end is turned 90 degrees (taller than wide)",
              len(far) and np.ptp(far[:, 2]) > 0.9 and np.ptp(far[:, 1]) < 0.6,
              (np.ptp(far[:, 1]) if len(far) else None, np.ptp(far[:, 2]) if len(far) else None))
        shot("1_twist_preview")
        # the handle: round the axis end, r * 1.15 out, at 90 degrees from the bend direction (Z)
        E = fm.f.O + fm.f.L * fm.f.z
        s, a, b = fm.f.coords(deform.points_of_brep(store.get_brep(bar)))
        r = float(np.sqrt(a * a + b * b).max()) * 1.15
        from blender_nurbs.valuedrag import rotate_points
        H = lambda deg: rotate_points((E + r * fm.f.x)[None], E, fm.f.z, deg)[0]
        p0 = wpx(H(90.0))
        move(p0)
        yield 0.3
        win.event_simulate('LEFTMOUSE', 'PRESS', x=p0[0], y=p0[1])
        for k in range(1, 13):
            p = wpx(H(90.0 + 5.0 * k))
            win.event_simulate('MOUSEMOVE', 'NOTHING', x=p[0], y=p[1])
            at[:] = p
        yield 0.3
        win.event_simulate('LEFTMOUSE', 'RELEASE', x=at[0], y=at[1])
        yield 0.3
        check("dragging the handle keeps the stage open", running(), modal())
        shot("2_twist_dragged")
        key('RET')
        yield 1.0
        p = last_step(bar)
        check("Enter: a Twist step at the dragged angle (about 150 degrees)",
              not running() and p is not None and abs(p["deform_angle"] - 150.0) < 8.0,
              p and round(p["deform_angle"], 2))
        check("the twisted bar is one closed solid, its volume kept",
              store.get_brep(bar).is_manifold_solid() and abs(vol(bar) / v0 - 1) < 1e-4, vol(bar) / v0)
        shot("3_twisted")
        with over():
            bpy.ops.nproto.remove_step(index=len(features.stack(bar)) - 1)
        yield 0.3

        # -- Bend: S for smooth, typed 120, Enter
        with over():
            bpy.ops.nproto.deform('INVOKE_DEFAULT', kind='BEND', angle=45.0)
        yield 0.5
        check("Bend starts its value stage", running(), modal())
        move((region.x + region.width // 2, region.y + region.height // 2))
        key('S', "s")                    # (a letter after digits would be read as a unit)
        typed("120")
        yield 0.4
        shot("4_bend_preview")
        key('RET')
        yield 1.0
        p = last_step(bar)
        check("S + typed 120: a smooth 120 degree Bend step",
              not running() and p is not None and abs(p["deform_angle"] - 120.0) < 1e-9 and p["deform_smooth"],
              p)
        check("the bent bar is closed, its volume kept",
              store.get_brep(bar).is_manifold_solid() and abs(vol(bar) / v0 - 1) < 1e-4, vol(bar) / v0)
        shot("5_bent")
        with over():
            bpy.ops.nproto.remove_step(index=len(features.stack(bar)) - 1)
        yield 0.3

        # -- Taper: typed 0.5, Enter; then a fold: refused, the stage stays open
        with over():
            bpy.ops.nproto.deform('INVOKE_DEFAULT', kind='TAPER', factor=1.0, smooth=False)
        yield 0.5
        typed("0.5")
        yield 0.3
        shot("6_taper_preview")
        key('RET')
        yield 1.0
        p = last_step(bar)
        check("typed 0.5: a Taper step, the volume as a linear taper's",
              p is not None and abs(p["deform_factor"] - 0.5) < 1e-9 and abs(vol(bar) / v0 - 0.58333) < 1e-3,
              (p, vol(bar) / v0))

    def runner():
        it = steps()

        def tick():
            try:
                return next(it)
            except StopIteration:
                pass
            except Exception:
                res["error"] = traceback.format_exc()
            json.dump(res, open(out, "w"), default=str)
            bpy.ops.wm.quit_blender()
            return None
        return tick

    bpy.app.timers.register(runner(), first_interval=1.0)


if GUI:
    gui()
else:
    background()
