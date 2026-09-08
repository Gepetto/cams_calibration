#!/usr/bin/env python3
"""Anchor the camera rig in the world by pointing a wand at known positions.

Two world frames are used in practice and they differ only in what the wand is
pointed at, so both live here:

    default    a frame marked out on the ground, from 3 pointed positions
    --robot    the robot base as the world frame, from 4 pointed positions

The anchor is written into the COMFI layout RT-COSMIK reads:

    <out>/extrinsics/cam_to_world/camera_<ref>/camera_<ref>_extrinsics.yaml

Only the *reference* camera is anchored. The others are placed relative to it by
chaining the stereo results from calibrate_cameras.py, which is more accurate
than pointing the wand at each of them; writing a world pose per camera would
also make RT-COSMIK prefer those poses over the chain.

Every camera that can see the wand is still measured. With three or more of them
the measurements are averaged into the anchor, cancelling part of the pointing
error. With two, a disagreement cannot be blamed on either camera, so the
reference camera's own measurement is kept and the disagreement is only
reported.

    # ground frame, capturing from the live cameras
    python3 scripts/set_world_frame.py --cameras 0 2 4 6

    # robot base as world frame, from images already on disk
    python3 scripts/set_world_frame.py --cameras 0 2 --robot --from-images --no-show
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_path)

from utils.calib_utils import (load_intrinsics, robot_frame_in_camera,
                               ground_frame_in_camera, invert_pose,
                               save_world_pose, intrinsics_path,
                               list_cameras_with_v4l2, install_to_rtcosmik,
                               chain_stereo_poses, fuse_world_anchor)
from utils.settings import Settings

settings = Settings()

# Averaging the wand measurements only pays off when they are of similar
# quality. Below three cameras a disagreement cannot be attributed to either
# camera, so the reference camera's own measurement is kept instead.
MIN_CAMERAS_TO_FUSE = 3

# Length of the world axes drawn on the verification image, in metres.
CHECK_AXIS_LENGTH = 0.1

# Above this the cameras disagree enough that one of them is likely mispointed.
DISAGREEMENT_WARN_DEG = 5.0

# What the wand is pointed at, and what it takes to define a frame from it.
FRAMES = {
    "ground": (ground_frame_in_camera, 3, "a frame marked on the ground"),
    "robot": (robot_frame_in_camera, 4, "the robot base"),
}


IMAGES_ROOT = repo_path


def images_dir(camera_id):
    """Where one camera's world-pointing images live."""
    return os.path.join(IMAGES_ROOT, f"images_world_cam_{camera_id}", "color")


def draw_wand_overlay(shown, corners, camera_matrix, dist_coeffs):
    """Draw the wand's own axes and its reprojected tip onto a preview frame.

    This mirrors wand_positions_in_camera exactly -- same corner model, same
    solver, same tip offset, and likewise only the first detected marker -- so
    the cross drawn here sits where SPACE would actually record the position.
    """
    half = settings.wand_marker_size / 2
    marker_points = np.array([[-half, half, 0], [half, half, 0],
                              [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(marker_points, corners[0].reshape(-1, 2),
                                  camera_matrix, dist_coeffs, False,
                                  cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None

    cv2.drawFrameAxes(shown, camera_matrix, dist_coeffs, rvec, tvec, half)

    rotation, _ = cv2.Rodrigues(rvec)
    tip = tvec + rotation @ settings.wand_end_effector_local_pos
    # The tip is already in camera coordinates, so project it with an identity pose.
    image_points, _ = cv2.projectPoints(tip.reshape(1, 3), np.zeros((3, 1)),
                                        np.zeros((3, 1)), camera_matrix, dist_coeffs)
    point = image_points.reshape(2)
    if not np.all(np.isfinite(point)):
        return None

    # Draw the shaft from the marker to the tip, so a wrong tip offset is obvious.
    tip_xy = tuple(np.round(point).astype(int))
    marker_xy = tuple(np.round(corners[0].reshape(-1, 2).mean(axis=0)).astype(int))
    cv2.line(shown, marker_xy, tip_xy, (0, 255, 255), 2)
    cv2.drawMarker(shown, tip_xy, (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    cv2.putText(shown, f"tip {tip.flatten()[2]:.2f} m",
                (tip_xy[0] + 24, tip_xy[1]), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3)
    return tip.flatten()


def capture(camera_ids, required_images, detector, intrinsics, show=True):
    """Store one image set per keypress, until the wand has been pointed everywhere."""
    available = list(list_cameras_with_v4l2().keys())
    missing = [c for c in camera_ids if c not in available]
    if missing:
        raise RuntimeError(f"cameras {missing} not found; v4l2 lists {available}")

    captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_ids]
    for cap in captures:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        cap.set(cv2.CAP_PROP_FPS, settings.fs)

    for camera_id in camera_ids:
        os.makedirs(images_dir(camera_id), exist_ok=True)

    print(f"Point the wand at each of the {required_images} positions.")
    print("SPACE saves one image per camera, ESC or q finishes.")
    img_idx = 0
    try:
        while True:
            frames = [cap.read()[1] for cap in captures]
            if not all(frame is not None for frame in frames):
                continue

            if show:
                tiles = []
                for frame, camera_id in zip(frames, camera_ids):
                    shown = frame.copy()
                    corners, ids, _ = detector.detectMarkers(
                        cv2.cvtColor(shown, cv2.COLOR_BGR2GRAY))
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(shown, corners, ids)
                        K, D = intrinsics[camera_id]
                        draw_wand_overlay(shown, corners, K, D)
                    else:
                        cv2.putText(shown, "no wand", (24, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
                    tile = cv2.resize(shown, (640, 480), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(tile, f"cam {camera_id}  {img_idx}/{required_images}",
                                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    tiles.append(tile)
                cv2.imshow("world frame", np.hstack(tiles))

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                for frame, camera_id in zip(frames, camera_ids):
                    cv2.imwrite(
                        os.path.join(images_dir(camera_id), f"img_{img_idx}.png"), frame)
                img_idx += 1
                print(f"  saved position {img_idx}/{required_images}")
                if img_idx == required_images:
                    print("  all positions captured")
            elif key in (27, ord('q')):
                break
    finally:
        for cap in captures:
            cap.release()
        cv2.destroyAllWindows()


def save_world_frame_check(camera_ids, reference, world_R_cam, world_T_cam,
                           relative, intrinsics, out, show=True):
    """Reproject the anchored world frame onto one image per camera.

    The printed position says where the camera thinks it is, but not whether the
    frame itself landed on the thing the wand was pointed at. This draws the
    anchor that was just saved -- after fusion, and after the stereo chain for
    the non-reference cameras -- back into each camera's own view, where a wrong
    anchor is obvious: the axes sit off the physical frame, or the origin floats.

    Because the non-reference cameras are drawn through the chain rather than
    from their own measurement, a bad stereo result shows up here too, as axes
    that are right in the reference view and drift in the others.
    """
    # The anchor is the reference camera expressed in the world; projecting
    # world points into an image needs the opposite direction.
    ref_R_world, ref_T_world = invert_pose(world_R_cam, world_T_cam)

    tiles = []
    for camera_id in camera_ids:
        folder = images_dir(camera_id)
        names = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not names:
            continue
        image = cv2.imread(names[0], 1)
        if image is None:
            continue

        # Carry the anchor into this camera along the chain: p_i = R_i p_ref + T_i.
        R_i, T_i = relative[camera_id]
        cam_R_world = R_i @ ref_R_world
        cam_T_world = R_i @ ref_T_world + T_i

        K, D = intrinsics[camera_id]
        cv2.drawFrameAxes(image, K, D, cv2.Rodrigues(cam_R_world)[0],
                          np.asarray(cam_T_world, dtype=float).reshape(3, 1),
                          CHECK_AXIS_LENGTH, 4)
        role = "reference" if camera_id == reference else "chained"
        cv2.putText(image, f"cam {camera_id} ({role})", (24, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
        cv2.putText(image, os.path.basename(names[0]), (24, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        tiles.append(cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA))

    if not tiles:
        return None

    check = np.hstack(tiles)
    path = os.path.join(out, "world_frame_check.png")
    cv2.imwrite(path, check)
    if show:
        cv2.imshow("world frame check", check)
        print("  press any key on the check window to finish")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", action="store_true",
                        help="use the robot base as the world frame (4 pointed "
                             "positions) instead of a frame on the ground (3)")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 2],
                        help="cameras that will see the wand; the first is the reference")
    parser.add_argument("--out", default=os.path.join(repo_path, "config", "cam_params"),
                        help="calibration root to write (COMFI layout)")
    parser.add_argument("--from-images", action="store_true",
                        help="skip capture and solve from images already on disk")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open windows; required with no display")
    parser.add_argument("--images-root", default=None, metavar="DIR",
                        help="where the images_world_cam_<i>/ folders live "
                             "(default: this repository)")
    parser.add_argument("--install", nargs="?", const=True, default=None, metavar="PATH",
                        help="also copy the result into RT-COSMIK "
                             "(default: its settings.cam_calib_path)")
    args = parser.parse_args()

    if args.images_root:
        global IMAGES_ROOT
        IMAGES_ROOT = args.images_root

    pose_in_camera, required_images, frame_name = FRAMES["robot" if args.robot
                                                         else "ground"]

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())

    intrinsics = {}
    for camera_id in args.cameras:
        path = intrinsics_path(args.out, camera_id)
        if not os.path.isfile(path):
            raise SystemExit(f"missing intrinsics for camera {camera_id}: {path}\n"
                             f"run scripts/calibrate_cameras.py first")
        intrinsics[camera_id] = load_intrinsics(path)

    if not args.from_images:
        capture(args.cameras, required_images, detector, intrinsics,
                show=not args.no_show)

    reference = args.cameras[0]

    print(f"\nWand measurements ({frame_name} as world frame)")
    measured = {}
    for camera_id in args.cameras:
        folder = images_dir(camera_id)
        count = len([n for n in os.listdir(folder) if n.endswith(".png")]) \
            if os.path.isdir(folder) else 0
        if count != required_images:
            # The two frames need different numbers of positions, so a mismatch
            # usually means the images were captured for the other one.
            hint = "" if count == 0 else \
                f" -- these look like images for the {'ground' if args.robot else 'robot'} frame"
            print(f"  camera_{camera_id}: {count} images "
                  f"(need {required_images}), skipped{hint}")
            continue

        K, D = intrinsics[camera_id]
        # Returns the world expressed in camera coordinates...
        cam_T_world, cam_R_world = pose_in_camera(
            os.path.join(folder, "*.png"), K, D, detector, settings.wand_marker_size)
        # ...so invert it: RT-COSMIK stores the camera expressed in the world.
        measured[camera_id] = invert_pose(cam_R_world, cam_T_world)
        print(f"  camera_{camera_id}: position {np.round(measured[camera_id][1], 3)} m")

    if reference not in measured:
        raise SystemExit(
            f"the reference camera {reference} has no usable wand measurement")

    relative = chain_stereo_poses(args.out, args.cameras)
    _, _, spread = fuse_world_anchor(measured, relative, reference)

    if len(measured) >= MIN_CAMERAS_TO_FUSE:
        world_R_cam, world_T_cam, spread = fuse_world_anchor(
            measured, relative, reference)
        print(f"\nFused {len(measured)} measurements into camera_{reference}'s anchor.")
    else:
        world_R_cam, world_T_cam = measured[reference]
        if len(measured) > 1:
            print(f"\nUsing camera_{reference}'s own measurement: fusing needs at "
                  f"least {MIN_CAMERAS_TO_FUSE} cameras to tell a bad pointing "
                  f"from a good one.")

    if len(measured) > 1:
        print("  how far each camera's estimate sits from the fused average:")
        for camera_id, (angle, distance) in sorted(spread.items()):
            print(f"    camera_{camera_id}: {angle:5.2f} deg, {distance*1000:6.1f} mm")
        worst = max(a for a, _ in spread.values())
        if worst > DISAGREEMENT_WARN_DEG:
            print(f"  warning: {worst:.1f} deg disagreement. At least one camera is "
                  f"pointed badly; check the wand positions before trusting the anchor.")

    path = save_world_pose(world_R_cam, world_T_cam, reference, args.out)
    print(f"\nAnchor -> {os.path.relpath(path, args.out)}")

    print("\nSanity check -- this is the camera's physical position in the room.")
    print("If it sits at the origin or at an implausible height, the pose is inverted.")
    print(f"  camera_{reference}: {np.round(world_T_cam, 3)} m, "
          f"height {world_T_cam[2]:.2f} m")
    others = [c for c in args.cameras if c != reference]
    if others:
        print(f"\nCameras {', '.join(str(c) for c in others)} are placed relative to "
              f"camera_{reference} by chaining the stereo results.")

    check = save_world_frame_check(args.cameras, reference, world_R_cam, world_T_cam,
                                   relative, intrinsics, args.out,
                                   show=not args.no_show)
    if check:
        print(f"\nWorld frame drawn on one image per camera -> "
              f"{os.path.relpath(check, args.out)}")
        print("  the axes must sit on the frame the wand was pointed at; x red, "
              "y green, z blue")

    print(f"\nWritten under {args.out}")
    if args.install is not None:
        install_to_rtcosmik(args.out, None if args.install is True else args.install)


if __name__ == "__main__":
    main()
