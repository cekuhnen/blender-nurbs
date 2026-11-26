"""Extrude's tilt knob with simulated input in a real window: a line drawn in
front view, seen from the side; the knob dragged swings the arrow up, Tab to
the tilt and a typed value:
    blender -b --factory-startup --python tests/test_extrude_tilt_gui.py
The background run writes an empty .blend, starts a GUI Blender with
--enable-event-simulate on it (a window opens for ~15 s), and reads back what
that run found. Screenshots go next to the results (the path is printed)."""
import json, math, os, subprocess, sys, tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = "--tilt-gui" in sys.argv


def background():
    tmp = tempfile.mkdtemp(prefix="nproto_tilt_gui_")
    blend, out = os.path.join(tmp, "empty.blend"), os.path.join(tmp, "results.json")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    cmd = [bpy.app.binary_path, "--factory-startup", "--enable-event-simulate", "--window-geometry", "0", "0",
           "1400", "900", blend, "--python", os.path.abspath(__file__), "--", "--tilt-gui", out]
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
        print(("PASS " if good else "FAIL ") + name, extra)
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
    from blender_nurbs import draw, features, store

    out = sys.argv[sys.argv.index("--tilt-gui") + 1]
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

    running = lambda: "NPROTO_OT_extrude" in [op.bl_idname for op in win.modal_operators]
    preview = lambda: draw.pick_overlay.get("preview") or {}
    knob = lambda: next((h for h in preview().get("handles") or [] if h["kind"] == "tilt"), None)

    def line():
        cu = bpy.data.curves.new("L", 'CURVE')
        cu.dimensions = '3D'
        sp = cu.splines.new('NURBS')
        sp.points.add(1)
        sp.points[0].co = (0.0, 0.0, 0.0, 1.0)
        sp.points[1].co = (1.0, 0.0, 0.0, 1.0)
        sp.order_u = 2
        sp.use_endpoint_u = True
        o = bpy.data.objects.new("L", cu)
        ctx.collection.objects.link(o)
        o[store.PROP_SKETCH_N] = [0.0, 1.0, 0.0]      # drawn in front view
        return o

    def far_row(o):
        from blender_nurbs import geom
        _, ext = features._k()
        d = ext.surface_cvs(store.get_surfaces(o)[0])
        P = np.array(d["points"], float).reshape(d["count_u"], d["count_v"], 3)
        return P[0], P[-1]

    def steps():
        ctx.scene.nproto_settings.construction_history = True
        L = line()
        r3 = area.spaces.active.region_3d
        r3.view_perspective = 'ORTHO'
        r3.view_rotation = Euler((math.pi / 2, 0.0, math.pi / 2)).to_quaternion()   # Right view: Y across, Z up
        r3.view_location = Vector((0.5, 0.6, 0.3))
        r3.view_distance = 3.0
        yield 0.5

        # -- typed distance, then the knob dragged 30 degrees up
        with over():
            bpy.ops.nproto.extrude('INVOKE_DEFAULT', inputs=json.dumps([["L", "S0"]]), tilt=0.0, draft=0.0,
                                   direction="NORMAL")
        yield 0.5
        check("Extrude starts its value stage", running())
        move((region.x + region.width // 2, region.y + 40))
        typed("1")
        yield 0.4
        k = knob()
        check("a tilt knob halfway up the arrow once there is a distance",
              k is not None and abs(np.asarray(k["pos"]) - [0.5, 0.55, 0.0]).max() < 1e-6, k and k["pos"])
        shot("1_knob")
        B = np.array([0.5, 0.0, 0.0])
        H = lambda deg: B + 0.55 * np.array([0.0, math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        p0 = wpx(H(0.0))
        move(p0)
        yield 0.3
        win.event_simulate('LEFTMOUSE', 'PRESS', x=p0[0], y=p0[1])
        for a in range(3, 31, 3):
            p = wpx(H(float(a)))
            win.event_simulate('MOUSEMOVE', 'NOTHING', x=p[0], y=p[1])
            at[:] = p
        yield 0.3
        win.event_simulate('LEFTMOUSE', 'RELEASE', x=at[0], y=at[1])
        yield 0.3
        k = knob()
        check("dragging the knob keeps the stage open, the knob on the arc", running() and k is not None,
              k and np.round(k["pos"], 4).tolist())
        shot("2_tilted")
        key('RET')
        yield 1.0
        made = [o for o in bpy.data.objects if o.name.startswith("Extrude")]
        if made:
            P0, P1 = far_row(made[-1])
            v = (P1 - P0).mean(axis=0)
            ang = math.degrees(math.atan2(v[2], v[1]))
            check("Enter: a wall swung about 30 degrees up from Y (seen from the side)",
                  not running() and abs(v[0]) < 1e-6 and abs(ang - 30.0) < 4.0 and abs(np.linalg.norm(v) - 1) < 1e-6,
                  np.round(v, 4).tolist())
            h = features.history(made[-1])
            p = h["params"] if h else {}
            check("the step keeps the tilt", abs(p.get("tilt", 0.0) - ang) < 1e-3, (p.get("tilt"), ang))
        else:
            check("Enter made an extrude", False)
        shot("3_made")

        # -- Tab to the tilt (after the draft), typed 20
        with over():
            bpy.ops.nproto.extrude('INVOKE_DEFAULT', inputs=json.dumps([["L", "S0"]]), tilt=0.0, draft=0.0,
                                   direction="NORMAL")
        yield 0.5
        move((region.x + region.width // 2, region.y + 40))
        typed("1")
        key('TAB')
        key('TAB')
        typed("20")
        yield 0.4
        shot("4_typed")
        key('RET')
        yield 1.0
        made = [o for o in bpy.data.objects if o.name.startswith("Extrude")]
        P0, P1 = far_row(made[-1])
        v = (P1 - P0).mean(axis=0)
        ang = math.degrees(math.atan2(v[2], v[1]))
        check("Tab, Tab, 20: a wall tilted 20 degrees up (a positive tilt leans up on screen)",
              abs(ang - 20.0) < 1e-6, np.round(v, 6).tolist())

    gen = steps()

    def tick():
        try:
            return next(gen)
        except StopIteration:
            pass
        except Exception:
            res["error"] = traceback.format_exc()
        json.dump(res, open(out, "w"))
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(tick, first_interval=1.0)


if GUI:
    gui()
else:
    background()
