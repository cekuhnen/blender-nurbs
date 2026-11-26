"""Trim Curves on curves that don't touch (apparent crossings) with simulated
input in a real window: an orthographic side view, then a perspective view
turned while the tool runs:
    blender -b --factory-startup --python tests/test_trim_curves_gui.py
The background run writes an empty .blend, starts a GUI Blender with
--enable-event-simulate on it (a window opens for ~10 s), and reads back what
that run found. Screenshots go next to the results (the path is printed)."""
import json, math, os, subprocess, sys, tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = "--trim-gui" in sys.argv


def background():
    tmp = tempfile.mkdtemp(prefix="nproto_trim_gui_")
    blend, out = os.path.join(tmp, "empty.blend"), os.path.join(tmp, "results.json")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    cmd = [bpy.app.binary_path, "--factory-startup", "--enable-event-simulate", "--window-geometry", "0", "0",
           "1400", "900", blend, "--python", os.path.abspath(__file__), "--", "--trim-gui", out]
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
    from blender_nurbs import curvejoin, curvetrim, draw, geom

    out = sys.argv[sys.argv.index("--trim-gui") + 1]
    shots = os.path.dirname(out)
    checks = {}
    res = {"checks": checks}

    def check(name, cond, extra=""):
        checks[name] = (bool(cond), str(extra))

    ctx = bpy.context
    win = ctx.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    r3 = area.spaces.active.region_3d
    over = lambda: ctx.temp_override(window=win, area=area, region=region)

    def wpx(P):
        q = view3d_utils.location_3d_to_region_2d(region, r3, Vector(P))
        return int(round(region.x + q[0])), int(round(region.y + q[1]))

    at = [0, 0]

    def move(p):
        for k in (3, 2, 1, 0):
            win.event_simulate('MOUSEMOVE', 'NOTHING', x=p[0] + k, y=p[1] + k)
        at[:] = p

    def key(t):
        win.event_simulate(t, 'PRESS', x=at[0], y=at[1])
        win.event_simulate(t, 'RELEASE', x=at[0], y=at[1])

    def click(p):
        move(p)
        win.event_simulate('LEFTMOUSE', 'PRESS', x=p[0], y=p[1])
        win.event_simulate('LEFTMOUSE', 'RELEASE', x=p[0], y=p[1])

    def shot(name):
        area.tag_redraw()
        with over():
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        with ctx.temp_override(window=win, screen=win.screen):
            bpy.ops.screen.screenshot(filepath=os.path.join(shots, name + ".png"))

    running = lambda: "NPROTO_OT_trim_curves" in [op.bl_idname for op in win.modal_operators]

    def line(name, a, b):
        cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
        sp = cu.splines.new('NURBS'); sp.points.add(1)
        for p, c in zip(sp.points, (a, b)):
            p.co = (*c, 1.0)
        sp.order_u = 2; sp.use_endpoint_u = True
        o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
        return o

    def world_spans(o):
        return [curvejoin.transform_spans(curvejoin.spline_spans(sp), geom.world_matrix(o)) for sp in o.data.splines]

    ZC = 0.2      # the cutting line's height; the circle (radius 0.5, plane x = -1) lies behind it

    def steps():
        with over():
            bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(-1.0, 0.0, 0.0), rotation=(0, math.pi / 2, 0))
        circ = ctx.view_layer.objects.active
        circ.name = "Ring"
        cut = line("Cutter", (0.0, -1.0, ZC), (0.0, 1.0, ZC))
        line("Other", (0.0, -1.0, -0.3), (0.0, 1.0, -0.3))
        r3.view_perspective = 'ORTHO'
        r3.view_rotation = Euler((math.pi / 2, 0.0, math.pi / 2)).to_quaternion()     # Right view: along -X
        r3.view_location = Vector((-0.5, 0.0, 0.0))
        r3.view_distance = 3.0
        yield 0.5
        v = curvetrim.view_spec(r3)
        check("the side view looks along -X, orthographic", np.allclose(v["dir"], (-1, 0, 0), atol=1e-6)
              and v["eye"] is None, v)
        with over():
            bpy.ops.object.select_all(action='DESELECT')
            cut.select_set(True)
            ctx.view_layer.objects.active = cut
            bpy.ops.ed.undo_push(message="setup")
            bpy.ops.nproto.trim_curves('INVOKE_DEFAULT')
        yield 0.4
        check("Trim Curves runs", running())
        top = wpx((-1.0, 0.0, 0.5))
        move(top)
        yield 0.3
        hv = draw.pick_overlay.get("hover") or []
        zs = [p.z for p in hv]
        check("hovering the ring's top: the piece above the line lights up (Auto: apparent crossings)",
              len(hv) > 4 and min(zs) > ZC - 1e-4 and max(zs) > 0.49, (len(hv), min(zs or [0]), max(zs or [0])))
        shot("1_ortho_hover")
        click(top)
        yield 0.4
        ws = world_spans(circ)
        P = np.vstack([curvetrim.sample_spans(w, 20)[0] for w in ws])
        ends = [curvejoin._ends(w) for w in ws]
        ez = [float(e[2]) for pair in ends for e in pair]
        check("click: the ring's top is trimmed off at the line's height as seen from the side",
              len(ws) == 1 and not circ.data.splines[0].use_cyclic_u and P[:, 2].max() < ZC + 1e-5
              and all(abs(z - ZC) < 1e-5 for z in ez), (len(ws), float(P[:, 2].max()), ez))
        shot("2_ortho_trimmed")
        key('BACK_SPACE')
        yield 0.3
        check("Backspace puts the ring back", circ.data.splines[0].use_cyclic_u and running())

        # perspective, turned while the tool runs: the crossings follow the new view
        r3.view_perspective = 'PERSP'
        r3.view_rotation = Euler((1.25, 0.0, 1.2)).to_quaternion()
        r3.view_location = Vector((-0.5, 0.0, 0.0))
        r3.view_distance = 4.0
        area.tag_redraw()
        yield 0.5
        v = curvetrim.view_spec(r3)
        E = np.array(v["eye"])
        # the ring's top as seen now: the point of the ring highest on screen
        ring = curvetrim.sample_spans(world_spans(circ)[0], 64)[0]
        scr = np.array([wpx(p) for p in ring])
        top = tuple(scr[np.argmax(scr[:, 1])])
        move(top)
        yield 0.3
        click(top)
        yield 0.4
        ws = world_spans(circ)
        ends = [e for w in ws for e in curvejoin._ends(w)]
        A, B = np.array([0.0, -1.0, ZC]), np.array([0.0, 1.0, ZC])

        def ray_line(P):
            """distance between the eye's ray through P and the cutting line"""
            r = (P - E) / np.linalg.norm(P - E)
            n = np.cross(r, B - A)
            return abs(float((A - E) @ n / np.linalg.norm(n)))
        miss = [ray_line(np.asarray(e)) for e in ends]
        zs = [float(e[2]) for e in ends]
        check("perspective: the cut ends lie on eye rays through the line (exact central projection)",
              len(ws) == 1 and len(ends) == 2 and max(miss) < 1e-5, (len(ws), miss))
        check("perspective: the cuts differ from the side view's (the view change was picked up)",
              any(abs(z - ZC) > 1e-3 for z in zs), zs)
        shot("3_persp_trimmed")
        key('RET')
        yield 0.4
        check("Enter finishes", not running())

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
