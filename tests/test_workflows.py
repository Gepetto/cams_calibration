"""End-to-end check of every calibration workflow, on the repo's own images.

Each case runs the real scripts as a user would, into a clean output directory,
and checks what came out. Expected values are the ones this rig has produced
consistently, so a silent change of behaviour shows up as a failure.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fixtures live with the tests, so a fresh clone can run them. They are the same
# captures the rig produced, stored greyscale: calibration converts to grey
# anyway, so the numbers are identical at a third of the size.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMAGES = ["--images-root", DATA]
PY = sys.executable
EXPECT = {
    "reproj_0": 0.2125, "reproj_2": 0.2016,
    "rmse": 0.6400, "baseline": 0.8269,
    "anchor": [-0.518, 1.917, 1.025],
}
results = []


def run(args, cwd=REPO, timeout=1200):
    p = subprocess.run([PY] + args, cwd=cwd, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def near(a, b, tol):
    return abs(a - b) <= tol


# --- 1. intrinsics + stereo, into a clean directory -------------------------
out = tempfile.mkdtemp()
code, log = run(["scripts/calibrate_cameras.py", "--cameras", "0", "2",
                 "--from-images", "--no-show", "--out", out] + IMAGES)
ok = code == 0
r0 = re.search(r"camera_0: reprojection ([\d.]+)", log)
r2 = re.search(r"camera_2: reprojection ([\d.]+)", log)
st = re.search(r"(\d+)/(\d+) shared views, rmse ([\d.]+), baseline ([\d.]+)", log)
check("calibrate_cameras runs into an empty directory", ok,
      "" if ok else log.strip().splitlines()[-1] if log.strip() else "")
if ok and r0 and r2 and st:
    check("intrinsics reproduce", near(float(r0.group(1)), EXPECT["reproj_0"], 1e-4)
          and near(float(r2.group(1)), EXPECT["reproj_2"], 1e-4),
          f"{r0.group(1)} / {r2.group(1)} px")
    check("stereo reproduces", near(float(st.group(3)), EXPECT["rmse"], 1e-4)
          and near(float(st.group(4)), EXPECT["baseline"], 1e-4),
          f"rmse {st.group(3)}, baseline {st.group(4)} m, {st.group(1)}/{st.group(2)} shared")
else:
    check("intrinsics reproduce", False, "no output parsed")
    check("stereo reproduces", False, "no output parsed")

expected_files = ["intrinsics/camera_0_intrinsics.yaml", "intrinsics/camera_2_intrinsics.yaml",
                  "extrinsics/cam_to_cam/camera_0_to_camera_2.yaml", "cameras.yaml"]
missing = [f for f in expected_files if not os.path.isfile(os.path.join(out, f))]
check("COMFI layout written", not missing, f"missing {missing}" if missing else "4 files")

# --- 2. world anchor, robot frame -------------------------------------------
code, log = run(["scripts/set_world_frame.py", "--cameras", "0", "2", "--robot",
                 "--from-images", "--no-show", "--out", out] + IMAGES)
anchor = re.search(r"camera_0: \[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\]", log)
ok = code == 0 and anchor is not None
check("set_world_frame --robot runs", ok,
      "" if ok else (log.strip().splitlines()[-1] if log.strip() else ""))
if anchor:
    got = [float(anchor.group(i)) for i in (1, 2, 3)]
    check("anchor reproduces", all(near(g, e, 1e-3) for g, e in zip(got, EXPECT["anchor"])),
          str(got))
check("world pose written", os.path.isfile(
    os.path.join(out, "extrinsics/cam_to_world/camera_0/camera_0_extrinsics.yaml")))
check("only the reference camera is anchored",
      os.listdir(os.path.join(out, "extrinsics/cam_to_world")) == ["camera_0"])

# --- 3. ground frame: never run on real data, so exercise the code path -----
# The repo only has 4-position robot pointings. Three of them do not describe a
# real ground frame, but they do prove the ground path runs rather than crashes.
ground = tempfile.mkdtemp()
for cam in (0, 2):
    src = os.path.join(DATA, f"images_world_cam_{cam}", "color")
    dst = os.path.join(ground, f"images_world_cam_{cam}", "color")
    os.makedirs(dst)
    for name in sorted(os.listdir(src))[:3]:
        shutil.copy(os.path.join(src, name), dst)
shutil.copytree(os.path.join(out, "intrinsics"), os.path.join(ground, "calib", "intrinsics"))
shutil.copytree(os.path.join(out, "extrinsics"), os.path.join(ground, "calib", "extrinsics"))
# point the script's image lookup at the trimmed copies
env_script = os.path.join(ground, "run_ground.py")
with open(env_script, "w") as fh:
    fh.write(f"""import sys, os
sys.path.insert(0, {REPO!r})
sys.argv = ["set_world_frame", "--cameras", "0", "2", "--from-images", "--no-show",
            "--out", {os.path.join(ground, 'calib')!r}]
import importlib.util
spec = importlib.util.spec_from_file_location(
    "swf", os.path.join({REPO!r}, "scripts", "set_world_frame.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.images_dir = lambda cam: os.path.join({ground!r}, f"images_world_cam_{{cam}}", "color")
m.main()
""")
code, log = run([env_script], cwd=ground)
ran = "Wand measurements" in log and "a frame marked on the ground" in log
check("set_world_frame ground path executes", code == 0 and ran,
      "code path only -- no real ground pointings exist to check correctness")

# --- 4. guards behave -------------------------------------------------------
code, log = run(["scripts/set_world_frame.py", "--cameras", "0", "2",
                 "--from-images", "--no-show", "--out", out] + IMAGES)
check("ground mode refuses robot images", code != 0 and "need 3" in log,
      "explains the mismatch" if "look like images for the robot frame" in log else "")

empty = tempfile.mkdtemp()
code, log = run(["scripts/set_world_frame.py", "--cameras", "0", "2", "--robot",
                 "--from-images", "--no-show", "--out", empty] + IMAGES)
check("missing intrinsics is a clear error",
      code != 0 and "missing intrinsics" in log and "calibrate_cameras.py first" in log)

code, log = run(["scripts/calibrate_cameras.py", "--cameras", "0",
                 "--from-images", "--no-show", "--out", tempfile.mkdtemp()] + IMAGES)
check("single camera is refused", code != 0 and "at least two cameras" in log)

# --- 5. install into RT-COSMIK ---------------------------------------------
dest = tempfile.mkdtemp()
code, log = run(["scripts/set_world_frame.py", "--cameras", "0", "2", "--robot",
                 "--from-images", "--no-show", "--out", out, "--install", dest] + IMAGES)
check("--install copies and verifies", code == 0 and "verified: RT-COSMIK loads 2" in log,
      re.search(r"installed (\d+) files", log).group(0) if "installed" in log else "")
check("manifest travels with the calibration",
      os.path.isfile(os.path.join(dest, "cameras.yaml")))

# --- 6. RT-COSMIK consumes the result --------------------------------------
probe = f"""
import sys, json, numpy as np
sys.path.insert(0, "/root/workspace/rt-cosmik/src")
from rtcosmik.camera.cam_utils import (load_camera_parameters, load_world_transformation,
                                       resolve_camera_ids, describe_camera_placement)
m, d, p, _, _ = load_camera_parameters({dest!r}, camera_ids=[0, 2])
wR, wT = load_world_transformation({dest!r}, ref_camera=0)
print(json.dumps({{"cams": len(m), "baseline": float(np.linalg.norm(p[1][:, 3])),
                  "anchor": np.asarray(wT).reshape(3).tolist(),
                  "resolved": {{str(k): v for k, v in resolve_camera_ids({dest!r}, [0, 2]).items()}},
                  "placements": {{str(k): v.tolist() for k, v in
                                 describe_camera_placement({dest!r}, camera_ids=[0, 2]).items()}}}}))
"""
code, log = run(["-c", probe])
data = None
for line in log.splitlines():
    if line.startswith("{"):
        data = json.loads(line)
check("RT-COSMIK loads the installed calibration", data is not None and data["cams"] == 2)
if data:
    check("RT-COSMIK uses the checkerboard chain",
          near(data["baseline"], EXPECT["baseline"], 1e-3), f"baseline {data['baseline']:.4f} m")
    check("RT-COSMIK reads the anchor",
          all(near(g, e, 1e-3) for g, e in zip(data["anchor"], EXPECT["anchor"])))
    check("camera ids resolve through the manifest", data["resolved"].get("0") == 0)

print()
failed = [n for n, ok, _ in results if not ok]
print(f"  {len(results) - len(failed)}/{len(results)} passed")
if failed:
    print("  FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
