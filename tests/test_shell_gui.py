"""Shell with simulated input in a real window (hover, clicks to pick and
unpick faces, Enter, the arrow dragged, a typed thickness, F):
    blender -b --factory-startup --python tests/test_shell_gui.py
The background run writes an empty .blend, starts a GUI Blender with
--enable-event-simulate on it (a window opens for ~20 s), and reads back what
that run found. Screenshots go next to the results (the path is printed)."""
import json, math, os, subprocess, sys, tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = "--shell-gui" in sys.argv


def background():
    tmp = tempfile.mkdtemp(prefix="nproto_shell_gui_")
    blend, out = os.path.join(tmp, "empty.blend"), os.path.join(tmp, "results.json")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    cmd = [bpy.app.binary_path, "--factory-startup", "--enable-event-simulate", "--window-geometry", "0", "0",
           "1400", "900", blend, "--python", os.path.abspath(__file__), "--", "--shell-gui", out]
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
        print(("PASS " if good else "FAIL ") + name, extra if not good else "")
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

    out = sys.argv[sys.argv.index("--shell-gui") + 1]
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

    def click():
        win.event_simulate('LEFTMOUSE', 'PRESS', x=at[0], y=at[1])
        win.event_simulate('LEFTMOUSE', 'RELEASE', x=at[0], y=at[1])

    def shot(name):
        area.tag_redraw()
        with over():
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        with ctx.temp_override(window=win, screen=win.screen):
            bpy.ops.screen.screenshot(filepath=os.path.join(shots, name + ".png"))

    modal = lambda: [op.bl_idname for op in win.modal_operators]
    running = lambda: "NPROTO_OT_shell" in modal()
    vol = lambda o: abs(store.get_brep(o).volume(1e-6))

    def steps():
        # a bar 4 x 1 x 1 along X: both ends taken away makes a square tube
        with over():
            bpy.ops.nproto.add_primitive(kind='BOX', width=4.0, depth=1.0, height=1.0)
        bar = ctx.view_layer.objects.active
        bar.name = "Bar"
        lo, hi = (np.array(v, float) for v in store.get_brep(bar).bounding_box())
        c = 0.5 * (lo + hi)
        r3 = area.spaces.active.region_3d
        r3.view_perspective = 'PERSP'
        r3.view_rotation = Euler((1.1, 0.0, 0.82)).to_quaternion()      # from front right, above (+X, -Y, +Z)
        r3.view_location = Vector(c)
        r3.view_distance = 9.0
        yield 0.5
        with over():
            bpy.ops.ed.undo_push(message="setup")
        v0 = vol(bar)
        # pick points on the two end faces and the top (seen from this view)
        endR, endL, top = (hi[0], c[1], c[2]), (lo[0], c[1], c[2]), (c[0], c[1], hi[2])
        view_dir = np.array(r3.view_rotation @ Vector((0, 0, -1)))
        seen = {n: float(np.dot(view_dir, d)) < 0 for n, d in (("R", (1, 0, 0)), ("L", (-1, 0, 0)), ("T", (0, 0, 1)))}
        check("view: the right end and the top face the camera", seen["R"] and seen["T"], seen)

        with over():
            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.nproto.shell('INVOKE_DEFAULT')
        yield 0.3
        check("Shell starts in the picking stage", running(), modal())
        move(wpx(endR))
        yield 0.4
        check("hover lights the face", draw.pick_overlay.get("preview") is not None)
        click()
        yield 0.2
        # the view turns while picking: middle-mouse drag, the numpad, the navigation gizmo
        npick = lambda: len(draw.pick_overlay.get("regions") or [])
        q0 = r3.view_rotation.copy()
        mid = (region.x + region.width // 2, region.y + region.height // 3)
        move(mid)
        win.event_simulate('MIDDLEMOUSE', 'PRESS', x=mid[0], y=mid[1])
        for k in range(1, 11):
            win.event_simulate('MOUSEMOVE', 'NOTHING', x=mid[0] + 15 * k, y=mid[1] + 4 * k)
        at[:] = (mid[0] + 150, mid[1] + 40)
        yield 0.2
        win.event_simulate('MIDDLEMOUSE', 'RELEASE', x=at[0], y=at[1])
        yield 0.3
        turned = math.degrees(q0.rotation_difference(r3.view_rotation).angle)
        check("picking: a middle-mouse drag turns the view", turned > 5 and running() and npick() == 1,
              f"{turned:.1f} deg, running {running()}, picks {npick()}")
        key('NUMPAD_7')
        yield 0.4
        z = r3.view_rotation @ Vector((0, 0, 1))
        check("picking: numpad 7 gives the top view", z.z > 0.999 and running() and npick() == 1,
              f"{tuple(round(v, 3) for v in z)} running {running()}")
        r3.view_rotation = q0
        yield 0.3
        from blender_nurbs import subsel
        prefs = ctx.preferences
        size = prefs.view.gizmo_size_navigate_v3d * prefs.system.ui_scale
        hdr = sum(r.height for r in area.regions if r.type in ('HEADER', 'TOOL_HEADER') and r.height > 1)
        gz = (region.x + region.width - int(0.8 * size), region.y + region.height - hdr - int(0.7 * size))
        on = subsel.over_navigation(ctx, area, region, *gz)
        move(gz)
        yield 0.1
        click()
        yield 0.3
        check("picking: a click on the navigation gizmo picks nothing", on and running() and npick() == 1,
              f"on gizmo {on}, running {running()}, picks {npick()}")
        r3.view_rotation = q0            # (a click on an axis bubble turns the view to it)
        r3.view_perspective = 'PERSP'
        yield 0.3
        move(wpx(top))
        yield 0.2
        click()
        yield 0.2
        check("two faces picked (red)", len(draw.pick_overlay.get("regions") or []) == 2,
              len(draw.pick_overlay.get("regions") or []))
        area.tag_redraw()
        yield 0.4
        shot("1_two_picked")
        click()                          # the top again: taken back
        yield 0.2
        check("a click on a picked face takes it back", len(draw.pick_overlay.get("regions") or []) == 1)
        key('RET')
        yield 0.5
        check("Enter: the thickness stage (arrow on the face)", running() and draw.pick_overlay.get("preview"),
              modal())
        # type a thickness
        for ch, k in (("0", 'ZERO'), (".", 'PERIOD'), ("1", 'ONE')):
            key(k, ch)
        yield 0.3
        area.tag_redraw()
        yield 0.4
        shot("2_thickness_preview")
        pv = draw.pick_overlay.get("preview") or {}
        check("preview draws the inner edges", len(pv.get("lines", [])) > 10, len(pv.get("lines", [])))
        key('RET')
        yield 0.8
        want = v0 - (4.0 - 0.1) * 0.8 * 0.8
        check("Enter: shelled 0.1, one end open", not running() and abs(vol(bar) - want) < 1e-5,
              f"vol {vol(bar):.5f} want {want:.5f}")
        check("the red picks are gone", not draw.pick_overlay.get("regions"))
        st = features.stack(bar)
        check("history step shell", len(st) == 1 and st[0]["op"] == "shell", st)
        area.tag_redraw()
        yield 0.4
        shot("3_shelled")

        # undo, then F for walls outside, dragging the arrow
        with over():
            bpy.ops.ed.undo()
        yield 0.5
        bar = bpy.data.objects["Bar"]
        check("undo: the bar is whole again", abs(vol(bar) - v0) < 1e-6, vol(bar))
        with over():
            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.nproto.shell('INVOKE_DEFAULT')
        yield 0.3
        move(wpx(endR))
        yield 0.2
        click()
        yield 0.1
        key('RET')
        yield 0.5
        key('F', 'f')
        yield 0.1
        # drag the arrow: from the face's centre outwards along +X
        from blender_nurbs import shell
        B, n = shell.face_centre(bar, min(range(store.get_brep(bar).face_count()),
                                          key=lambda k: abs(shell.face_centre(bar, k)[0][0] - hi[0])))
        check("the arrow stands in the middle of the face", np.allclose(B, endR, atol=1e-6), B)
        p0 = wpx(B)
        p1 = wpx(B + 0.3 * n)
        move(p0)
        yield 0.1
        win.event_simulate('LEFTMOUSE', 'PRESS', x=p0[0], y=p0[1])
        for k in range(1, 9):
            q = (int(p0[0] + (p1[0] - p0[0]) * k / 8), int(p0[1] + (p1[1] - p0[1]) * k / 8))
            win.event_simulate('MOUSEMOVE', 'NOTHING', x=q[0], y=q[1])
            at[:] = q
        yield 0.1
        win.event_simulate('LEFTMOUSE', 'RELEASE', x=at[0], y=at[1])
        yield 0.3
        area.tag_redraw()
        yield 0.4
        shot("4_outside_drag")
        key('RET')
        yield 0.8
        check("F + drag: walls outside, volume grew", not running() and vol(bar) > 0.01, f"vol {vol(bar):.5f}")
        p = features.stack(bar)[-1]["params"] if features.stack(bar) else {}
        t = float(p.get("thickness", 0))
        want = (1 + 2 * t) ** 2 * (4 + t) - v0
        check("outside: the volume of walls round the bar", t > 0.05 and abs(vol(bar) - want) < 1e-4,
              f"t {t:.4f} vol {vol(bar):.5f} want {want:.5f} {p}")

    g = steps()

    def tick():
        try:
            return next(g)
        except StopIteration:
            pass
        except Exception:
            res["error"] = traceback.format_exc()
        json.dump(res, open(out, "w"), indent=1, default=str)
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(tick, first_interval=1.0)


if GUI:
    gui()
else:
    background()
