"""Unit checks for the pose maths, with no images or cameras involved.

These run in milliseconds, so they are the ones to run while editing. The
end-to-end behaviour is covered separately by test_workflows.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.calib_utils as cu

results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def rot(axis, angle):
    import cv2 as cv
    return cv.Rodrigues(np.asarray(axis, dtype=float) * angle)[0]


# --- invert_pose ------------------------------------------------------------
rng = np.random.default_rng(0)
R = rot([0.3, -0.7, 0.2], 1.1)
T = np.array([1.0, -2.0, 3.0])
Ri, Ti = cu.invert_pose(R, T)
Rb, Tb = cu.invert_pose(Ri, Ti)
check("invert_pose round-trips", np.allclose(R, Rb) and np.allclose(T, Tb))

# a point mapped forward then back must return to itself
p = rng.normal(size=3)
check("invert_pose really inverts the transform",
      np.allclose(Ri @ (R @ p + T) + Ti, p))

# --- frame_from_directions --------------------------------------------------
origin = np.array([1.0, 2.0, 3.0])
_, F = cu.frame_from_directions(origin, [2.0, 0.0, 0.0], [0.0, 5.0, 0.0])
check("frame axes are orthonormal", np.allclose(F.T @ F, np.eye(3), atol=1e-12))
check("frame is right-handed", np.isclose(np.linalg.det(F), 1.0))
check("x axis follows the direction given", np.allclose(F[:, 0], [1, 0, 0]))
check("third point only fixes rotation about x, not scale",
      np.allclose(cu.frame_from_directions(origin, [2, 0, 0], [0, 5, 0])[1],
                  cu.frame_from_directions(origin, [9, 0, 0], [0, 0.1, 0])[1]))

# --- the two world frames must keep their opposite x senses -----------------
# Regression guard: the robot frame's x runs from the second pair's midpoint
# back towards the origin, the ground frame's runs away from its origin. A
# refactor that unified them silently flipped the robot frame's x and z.
POINTS = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([0.5, 2.0, 0.0]), np.array([1.5, 2.0, 0.0])]
real = cu.wand_positions_in_camera
cu.wand_positions_in_camera = lambda *a, **k: POINTS[:k.get("expected", 4)]
_, R_robot = cu.robot_frame_in_camera(None, None, None, None, None)
cu.wand_positions_in_camera = lambda *a, **k: POINTS[:3]
_, R_ground = cu.ground_frame_in_camera(None, None, None, None, None)
cu.wand_positions_in_camera = real

# robot: origin is midpoint(P0,P1) = (0.5,0,0), x towards origin from
# midpoint(P2,P3) = (1,2,0), so x points along -y here.
check("robot frame x runs towards the origin",
      np.allclose(R_robot[:, 0], [-0.5, -2.0, 0.0] / np.linalg.norm([-0.5, -2.0, 0.0])),
      str(np.round(R_robot[:, 0], 3)))
# ground: origin P0, x towards P1 = +x
check("ground frame x runs away from the origin",
      np.allclose(R_ground[:, 0], [1.0, 0.0, 0.0]), str(np.round(R_ground[:, 0], 3)))
check("the two frames disagree on x, as they should",
      not np.allclose(R_robot[:, 0], R_ground[:, 0]))

# --- average_rotations ------------------------------------------------------
base = rot([0, 0, 1], 0.4)
spread = [rot([0, 0, 1], 0.4 + d) for d in (-0.1, 0.0, 0.1)]
check("averaging symmetric rotations returns the middle one",
      np.allclose(cu.average_rotations(spread), base, atol=1e-9))
check("averaging one rotation returns it unchanged",
      np.allclose(cu.average_rotations([base]), base))
check("the average is a rotation",
      np.isclose(np.linalg.det(cu.average_rotations(spread)), 1.0))

# --- fuse_world_anchor ------------------------------------------------------
# Two cameras, exact measurements: fusing must return exactly the truth.
R0, T0 = rot([0.1, 0.2, 0.9], 0.6), np.array([0.5, -1.0, 2.0])
R2, T2 = rot([0.4, 0.1, 0.3], 1.3), np.array([-1.0, 0.4, 2.2])
relative = {0: (np.eye(3), np.zeros(3)), 2: (R2.T @ R0, R2.T @ (T0 - T2))}
measured = {0: (R0, T0), 2: (R2, T2)}
Rf, Tf, spread = cu.fuse_world_anchor(measured, relative, reference=0)
check("exact measurements fuse to the truth",
      np.allclose(Rf, R0, atol=1e-9) and np.allclose(Tf, T0, atol=1e-9))
check("spread is zero when the cameras agree",
      max(a for a, _ in spread.values()) < 1e-9)

# A distant camera's rotation error becomes a large position error, so its
# position estimate must be down-weighted rather than averaged in evenly.
noisy = {0: (R0, T0), 2: (rot([0, 1, 0], np.radians(6)) @ R2, T2)}
_, Tw, _ = cu.fuse_world_anchor(noisy, relative, reference=0)
_, Teven, _ = cu.fuse_world_anchor(noisy, relative, reference=0,
                                   wand_rot_deg=1e-9, wand_pos_mm=1e9)
check("a distant camera's rotation error is kept out of the position",
      np.linalg.norm(Tw - T0) < np.linalg.norm(Teven - T0),
      f"weighted {np.linalg.norm(Tw-T0)*1000:.0f} mm vs even {np.linalg.norm(Teven-T0)*1000:.0f} mm")
check("only cameras with both a pose and a chain are used",
      set(cu.fuse_world_anchor(measured, {0: relative[0]}, reference=0)[2]) == {0})

print()
failed = [n for n, ok in results if not ok]
print(f"  {len(results) - len(failed)}/{len(results)} passed")
if failed:
    print("  FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
