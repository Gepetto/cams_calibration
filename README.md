# cams_calibration

Calibrates a multi-camera rig for [RT-COSMIK](https://github.com/Gepetto/rt-cosmik).
Output is written directly in the layout RT-COSMIK reads, so no conversion step.

Two steps: intrinsics and stereo pairs from a checkerboard, then one world anchor
from a pointed wand.

## Setup

Check `utils/settings.py` matches your hardware:

| setting | meaning |
|---|---|
| `checkerboard_rows`, `checkerboard_columns` | inner corners, i.e. squares − 1 |
| `checkerboard_scaling` | square size in metres |
| `wand_marker_size` | aruco marker size in metres |
| `wand_end_effector_local_pos` | wand tip offset from its marker |

## 1. Cameras

```bash
python3 scripts/calibrate_cameras.py --cameras 0 2 4 6 --labels 0=front_left 2=front_right
```

SPACE saves one image per camera, `q` finishes. Move the checkerboard around the
volume, keeping it visible to **both cameras of each adjacent pair** — cameras
are paired 0-2, 2-4, 4-6, and only shots seen by both count. The report shows how
many did:

```
camera_0 -> camera_2: 17/17 shared views, rmse 0.6400, baseline 0.8269 m
```

Fewer than 6 shared views is refused. If a pair is starved, capture more with the
board in the space those two cameras overlap.

`--labels` records human names for the cameras. Optional but recommended.

## 2. World frame

```bash
python3 scripts/set_world_frame.py --cameras 0 2 4 6            # frame on the ground, 3 points
python3 scripts/set_world_frame.py --cameras 0 2 4 6 --robot    # robot base, 4 points
```

Point the wand at each position in turn; SPACE saves one image per camera.

Only the **first camera** is anchored — the rest are placed by chaining the stereo
results, which is more accurate than pointing the wand at each of them. Every
camera that sees the wand is still measured: with 3+ cameras the measurements are
averaged, with 2 the first camera's is used and the disagreement only reported.

Check the printed position is where the camera physically is. If it sits near the
origin or at an implausible height, the transform is inverted.

## 3. Install into RT-COSMIK

Add `--install` to either script, or both:

```bash
python3 scripts/set_world_frame.py --cameras 0 2 --robot --install
```

It copies into RT-COSMIK's `settings.cam_calib_path` and loads the result back to
confirm it is usable.

## Output

```
config/cam_params/
├── cameras.yaml                                              which camera is which
├── intrinsics/camera_<i>_intrinsics.yaml                     K, D
└── extrinsics/
    ├── cam_to_cam/camera_<a>_to_camera_<b>.yaml              stereo pairs
    └── cam_to_world/camera_<ref>/camera_<ref>_extrinsics.yaml  the anchor
```

World poses store the **camera's pose in the world frame** (`p_world = R @ p_cam + T`),
so `T` is the camera's position in the room. Stereo pairs keep OpenCV's own
convention and are not inverted. See RT-COSMIK's README for the full convention.

## When the checkerboard cannot reach a pair

A side option, and a much less accurate one. If two cameras are too far apart or
too opposed to ever share a board view, their pose can be recovered from a person
walking in the scene instead:

```bash
python3 scripts/calibrate_from_human.py --cameras 0 2 4 6 \
    --videos recordings/walk --height 1.78
```

It writes the same `cam_to_cam` files, so steps 2 and 3 are unchanged. Intrinsics
must exist first.

The subject has to **walk a circuit covering the floor area**. Standing in one
place or walking a straight line leaves the geometry underdetermined; the script
measures this and refuses such a recording.

Expect the cameras to be placed **about 3% of their baseline out** — 25 mm on a
0.8 m pair, 150 mm on a 5 m one, against roughly 5 mm for the checkerboard. That
is the state of the art for this technique, not a shortfall of the
implementation. Prefer `calibrate_cameras.py` wherever a board can reach.

In RT-COSMIK that error becomes a ~0.5° rigid rotation of the reconstruction
rather than a distortion, so joint angles are unaffected and absolute placement
shifts by ~60 mm. A pipeline that triangulates 2D instead would see real shape
distortion and should not use this.

## Recalibrating without the rig

Both scripts take `--from-images` to reuse the images already on disk,
`--no-show` to run with no display, and `--images-root` to read them from
somewhere other than this repository:

```bash
python3 scripts/calibrate_cameras.py --cameras 0 2 --from-images --no-show
```

## Tests

```bash
python3 tests/run_all.py            # everything
python3 tests/run_all.py geometry   # just the fast maths checks
```

`test_geometry.py` checks the pose maths in milliseconds. `test_workflows.py`
runs both scripts end to end on the images in this repo and checks the numbers
they produce, so a change of behaviour fails rather than passing quietly. It
takes a few minutes.

The images they run on are in `tests/data/`, so a fresh clone can run them with
no rig. They are the rig's own captures stored greyscale — calibration converts
to grey anyway, so the numbers are identical at a third of the size (10 MB).

Not covered: live capture, and the ground world frame — only robot-base
pointings exist, so that path is checked for running, not for correctness.

## Camera identity

`camera_0`, `camera_2` … are v4l2 indices, which are enumeration order, not
identity: they can change on reboot or replug. `cameras.yaml` records the USB port
behind each id so RT-COSMIK can detect a recabled rig and remap instead of pairing
a camera with the wrong calibration. Identical cameras usually share a placeholder
serial, so the port is the only discriminator — keep each camera in its own socket.

Moving a camera to a **new place in the room** invalidates its extrinsics and
cannot be detected. Recalibrate.
