#!/usr/bin/env python3
"""Calibrate N cameras: intrinsics per camera, stereo pose per adjacent pair.

Writes straight into the COMFI layout, which is what RT-COSMIK reads, so the
output of this script is usable by the toolbox with no conversion step:

    <out>/intrinsics/camera_<i>_intrinsics.yaml
    <out>/extrinsics/cam_to_cam/camera_<a>_to_camera_<b>.yaml

Capture and calibration are separate, so a recorded session can be recalibrated
without the rig:

    # capture checkerboard images from the live cameras, then calibrate
    python3 scripts/calibrate_cameras.py --cameras 0 2 4 6

    # calibrate from images already on disk
    python3 scripts/calibrate_cameras.py --cameras 0 2 --from-images
"""
import argparse
import os
import sys

import cv2
import numpy as np

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_path)

from utils.calib_utils import (calibrate_intrinsics, load_intrinsics, calibrate_stereo_pair,
                               save_intrinsics, save_stereo_pose,
                               intrinsics_path, list_cameras_with_v4l2,
                               install_to_rtcosmik, save_camera_manifest,
                               camera_hardware_info, count_shared_checkerboard_views)
from utils.settings import Settings

settings = Settings()

# Below this many shared views a stereo pair is not worth solving; the
# result would be dominated by whichever few shots happened to overlap.
MIN_SHARED_VIEWS = 6


IMAGES_ROOT = repo_path


def images_dir(camera_id):
    """Where one camera's checkerboard images live."""
    return os.path.join(IMAGES_ROOT, f"images_calib_cam_{camera_id}", "color")


def capture(camera_ids):
    """Show the live views and store a checkerboard image set on each keypress."""
    camera_dict = list_cameras_with_v4l2()
    available = list(camera_dict.keys())
    missing = [c for c in camera_ids if c not in available]
    if missing:
        raise RuntimeError(f"cameras {missing} not found; v4l2 lists {available}")

    captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_ids]
    for cap in captures:
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        cap.set(cv2.CAP_PROP_FPS, settings.fs)

    for camera_id in camera_ids:
        os.makedirs(images_dir(camera_id), exist_ok=True)

    print("SPACE saves one image per camera, ESC or q finishes.")
    img_idx = 0
    try:
        while True:
            frames = [cap.read()[1] for cap in captures]
            if not all(frame is not None for frame in frames):
                continue

            # One window showing every camera keeps the operator's attention in
            # one place, and scales past two cameras.
            tiles = [cv2.resize(f, (640, 480), interpolation=cv2.INTER_NEAREST)
                     for f in frames]
            for tile, camera_id in zip(tiles, camera_ids):
                cv2.putText(tile, f"cam {camera_id}", (12, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("calibration", np.hstack(tiles))

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                for frame, camera_id in zip(frames, camera_ids):
                    cv2.imwrite(os.path.join(images_dir(camera_id), f"img_{img_idx}.png"), frame)
                img_idx += 1
                print(f"  saved image set {img_idx}")
            elif key in (27, ord('q')):
                break
    finally:
        for cap in captures:
            cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 2],
                        help="camera ids, in rig order; adjacent ones are paired")
    parser.add_argument("--out", default=os.path.join(repo_path, "config", "cam_params"),
                        help="calibration root to write (COMFI layout)")
    parser.add_argument("--from-images", action="store_true",
                        help="skip capture and calibrate from images already on disk")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open detection windows; required with no display")
    parser.add_argument("--install", nargs="?", const=True, default=None,
                        metavar="PATH",
                        help="also copy the result into RT-COSMIK "
                             "(default: its settings.cam_calib_path)")
    parser.add_argument("--images-root", default=None, metavar="DIR",
                        help="where the images_calib_cam_<i>/ folders live "
                             "(default: this repository)")
    parser.add_argument("--labels", nargs="+", default=None, metavar="ID=NAME",
                        help="human names for the cameras, e.g. 0=front_left "
                             "2=front_right. Recorded alongside the calibration.")
    args = parser.parse_args()

    if args.images_root:
        global IMAGES_ROOT
        IMAGES_ROOT = args.images_root

    cameras = args.cameras
    if len(cameras) < 2:
        raise SystemExit("at least two cameras are needed to relate them")

    if not args.from_images:
        capture(cameras)

    print("\nIntrinsics")
    for camera_id in cameras:
        folder = images_dir(camera_id)
        if not os.path.isdir(folder):
            raise SystemExit(f"no images for camera {camera_id}: {folder}")
        reproj, mtx, dist = calibrate_intrinsics(
            images_folder=os.path.join(folder, "*.png"), show=not args.no_show)
        path = save_intrinsics(mtx, dist, reproj, camera_id, args.out)
        print(f"  camera_{camera_id}: reprojection {float(reproj):.4f} px -> "
              f"{os.path.relpath(path, args.out)}")

    print("\nStereo pairs")
    for cam_a, cam_b in zip(cameras[:-1], cameras[1:]):
        # Adjacent cameras in a ring overlap less the more cameras there are, so
        # check the pair actually shares enough views before trying to solve it.
        shared, total = count_shared_checkerboard_views(
            os.path.join(images_dir(cam_a), "*.png"),
            os.path.join(images_dir(cam_b), "*.png"))
        if shared < MIN_SHARED_VIEWS:
            raise SystemExit(
                f"  camera_{cam_a} -> camera_{cam_b}: only {shared} of {total} shots "
                f"show the checkerboard to both cameras (need {MIN_SHARED_VIEWS}).\n"
                f"  Capture more shots with the board visible to both, in the space "
                f"they overlap.")

        mtx_a, dist_a = load_intrinsics(intrinsics_path(args.out, cam_a))
        mtx_b, dist_b = load_intrinsics(intrinsics_path(args.out, cam_b))
        rmse, R, T = calibrate_stereo_pair(
            mtx_a, dist_a, mtx_b, dist_b,
            os.path.join(images_dir(cam_a), "*.png"),
            os.path.join(images_dir(cam_b), "*.png"),
            show=not args.no_show)
        path = save_stereo_pose(mtx_a, dist_a, mtx_b, dist_b, R, T, rmse,
                                cam_a, cam_b, args.out)
        print(f"  camera_{cam_a} -> camera_{cam_b}: {shared}/{total} shared views, "
              f"rmse {float(rmse):.4f}, baseline {np.linalg.norm(T):.4f} m "
              f"-> {os.path.relpath(path, args.out)}")

    # Record which physical camera each id refers to, so a reshuffled rig is
    # caught before it silently pairs a camera with another one's calibration.
    labels = {}
    for item in (args.labels or []):
        if "=" not in item:
            raise SystemExit(f"--labels expects ID=NAME, got {item!r}")
        key, value = item.split("=", 1)
        labels[int(key)] = value
    manifest = save_camera_manifest(args.out, cameras, labels)
    print(f"\nRecorded the rig in {os.path.relpath(manifest, args.out)}:")
    for camera_id in cameras:
        info = camera_hardware_info(camera_id)
        name = labels.get(camera_id, f"camera_{camera_id}")
        print(f"  {name}: {info.get('model', 'unknown')} on port "
              f"{info.get('bus_info', 'unknown')}")

    print(f"\nWritten under {args.out}")
    if args.install is not None:
        install_to_rtcosmik(args.out, None if args.install is True else args.install)
    print("Anchor the world frame next: scripts/set_world_frame.py")


if __name__ == "__main__":
    main()
