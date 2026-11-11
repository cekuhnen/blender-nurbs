"""Run crash-prone kernel jobs (trim, fillet) in kernel/worker.py, in a
separate Python process, and bring the results back into Blender."""

import json
import os
import shutil
import subprocess
import sys
import tempfile

from . import geom, store

WORKER = os.path.join(os.path.dirname(__file__), "kernel", "worker.py")


class JobError(RuntimeError):
    pass


def xform(obj):
    loc, rot, scale = geom.world_matrix(obj).decompose()
    axis, angle = rot.to_axis_angle()
    return {"loc": list(loc), "axis": list(axis), "angle": float(angle) * 57.29577951308232,
            "scale": list(scale)}


def curve_json(kcurve):
    from .kernel import ext
    d = ext.curve_cvs(kcurve)
    return {"degree": d["degree"], "points": [list(p) for p in d["points"]],
            "knots": list(d["knots"]), "weights": list(d["weights"])}


def run(op, objects, out_keys, timeout=120.0, **params):
    """objects: {key: blender object (NURBS)}. Returns (result dict, {key: Brep})."""
    tmp = tempfile.mkdtemp(prefix="nproto_job_")
    try:
        job = {"op": op, "breps": {}, "xforms": {}, "out": {}}
        for key, obj in objects.items():
            path = os.path.join(tmp, f"in_{key}.sm")
            if isinstance(obj, tuple):          # (brep text, xform dict): an intermediate result
                text, xf = obj
            else:
                text, xf = obj[store.PROP_BREP], xform(obj)
            with open(path, "w", newline="\n") as f:      # the same bytes on Windows (no \r added)
                f.write(text)
            job["breps"][key] = path
            job["xforms"][key] = xf
        for key in out_keys:
            job["out"][key] = os.path.join(tmp, f"out_{key}.sm")
        job["out_dir"] = tmp
        job.update(params)
        jpath = os.path.join(tmp, "job.json")
        with open(jpath, "w") as f:
            json.dump(job, f)
        try:
            proc = subprocess.run([sys.executable, "-I", WORKER, jpath], capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise JobError(f"kernel {op} timed out after {timeout:.0f}s")
        lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        if proc.returncode != 0 or not lines:
            raise JobError(f"kernel crashed during {op} (exit {proc.returncode}); "
                           "Blender is fine -- try other inputs or settings")
        res = json.loads(lines[-1])
        if not res.get("ok"):
            raise JobError(res.get("error", "kernel error"))
        breps = {}
        for name in sorted(os.listdir(tmp)):  # declared outputs + any out_*.sm a job wrote
            if name.startswith("out_") and name.endswith(".sm"):
                with open(os.path.join(tmp, name)) as f:
                    breps[name[4:-3]] = f.read()
        return res["result"], breps
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
