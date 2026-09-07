#!/usr/bin/env python3
"""Rough camera-to-camera calibration from a person walking in the scene.

SIDE OPTION, AND AN INACCURATE ONE. It places the cameras about 3% of their
baseline apart from where they really are -- 25 mm on a 0.8 m pair, 150 mm on a
5 m one, against roughly 5 mm for the checkerboard. Use calibrate_cameras.py
whenever a checkerboard can reach both cameras of a pair. This exists for the
case where it cannot: wide or opposed pairs, several metres apart.

Intrinsics still come from the checkerboard and the world frame still comes from
set_world_frame.py; this only replaces extrinsics/cam_to_cam/.

What that error does downstream depends on the pipeline. RT-COSMIK takes each
camera's monocular 3D skeleton and fuses them, so the body's shape never comes
from the calibration: the error turns into a ~0.5 deg rigid rotation of the whole
reconstruction, leaving joint angles and segment lengths intact (measured: 0.01%
on segment lengths, identical joint angles against mocap) and costing ~60 mm of
absolute placement. A pipeline that triangulates 2D instead would see real
deformation -- 1.3% on segment lengths -- and should not use this.

    # 1. intrinsics first, as usual
    python3 scripts/calibrate_cameras.py --cameras 0 2 4 6

    # 2. record the subject walking a circuit, then
    python3 scripts/calibrate_from_human.py --cameras 0 2 4 6 \
        --videos recordings/walk --height 1.78

    # 3. anchor the world frame, as usual
    python3 scripts/set_world_frame.py --cameras 0 2 4 6 --robot

Recording protocol, which matters more than anything else here: the subject must
walk a circuit covering the floor area. Standing in one place -- however much
they move their arms, squat or jump -- leaves the joint cloud too flat to pin the
geometry down, and so does walking a straight line. The script measures this and
refuses a recording that cannot support a calibration.

Measured against motion capture on 8 COMFI participants, 24 camera pairs:
0.97 deg rotation, 2.8-3.1% of baseline in translation. That is the published
state of the art, not an implementation shortfall -- Lee et al. (RA-L 2022)
report 0.022 rad and 0.057-0.123 m for the same class of method. The ceiling is
the pose estimator's view-dependent keypoint bias, about 7 px, and no amount of
optimisation gets past it. See utils/human_calib.py for the measurements and for
what was tried and rejected.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_path)

from utils.calib_utils import (load_intrinsics, intrinsics_path, save_stereo_pose,
                               install_to_rtcosmik)
from utils.human_calib import (cloud_thickness, describe_conditioning, register_pair,
                               triangulate, stature_scale)
from utils.settings import Settings

settings = Settings()


def read_timestamps(path):
    """Frame timestamps, or None when the recording has none."""
    if not os.path.isfile(path):
        return None
    stamps = []
    with open(path) as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                stamps.append(np.datetime64(row[1].replace(" ", "T")))
    return np.asarray(stamps) if stamps else None


def align_frames(video_dir, cameras, step):
    """Frame index per camera, matched to the reference camera.

    Cameras that free-run drift apart by up to a frame, which at walking speed
    is a couple of centimetres of subject motion -- the same size as the error
    being measured. Timestamps are used when the recording has them; without
    them the frame index is taken at face value.
    """
    stamps = {c: read_timestamps(os.path.join(video_dir, f"camera_{c}_timestamps.csv"))
              for c in cameras}
    reference = stamps[cameras[0]]
    if reference is None:
        counts = []
        for cam in cameras:
            cap = cv2.VideoCapture(os.path.join(video_dir, f"camera_{cam}.mp4"))
            counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
        picks = list(range(0, min(counts), step))
        return picks, {c: picks for c in cameras}

    picks = list(range(0, len(reference), step))
    want = reference[picks].astype("datetime64[us]").astype(np.int64)
    matched = {cameras[0]: picks}
    for cam in cameras[1:]:
        if stamps[cam] is None:
            matched[cam] = picks
            continue
        have = stamps[cam].astype("datetime64[us]").astype(np.int64)
        idx = np.clip(np.searchsorted(have, want), 0, len(have) - 1)
        earlier = np.maximum(idx - 1, 0)
        take_earlier = np.abs(have[earlier] - want) < np.abs(have[idx] - want)
        matched[cam] = np.where(take_earlier, earlier, idx).tolist()
    return picks, matched


def extract(video_dir, cameras, intrinsics, distortions, step):
    """Run the pose estimator over the recording, one batch per aligned frame."""
    try:
        from rtcosmik.config_loader import settings as rt_settings
        from rtcosmik.model_weights import resolve_detector_engine
        from rtcosmik.nlf.nlf import NLFEstimator, extract_views
    except ImportError as exc:
        raise SystemExit(
            f"this script needs RT-COSMIK importable for the pose estimator: {exc}\n"
            "  Install it, or run with --cache to reuse an earlier extraction.") from exc

    width, height = rt_settings.width, rt_settings.height
    estimator = NLFEstimator(
        yolo_path=resolve_detector_engine(rt_settings.yolo_path, len(cameras)),
        nlf_path=rt_settings.nlf_path, cano_path=rt_settings.cano_path,
        image_size=(width, height), cam_Ks=[k.astype(np.float32) for k in intrinsics],
        indices=rt_settings.nlf_indices, conf=rt_settings.yolo_conf,
        imgsz=rt_settings.yolo_imgsz, device=rt_settings.device,
        warmup=True, warmup_iters=3)

    picks, matched = align_frames(video_dir, cameras, step)
    captures = [cv2.VideoCapture(os.path.join(video_dir, f"camera_{c}.mp4"))
                for c in cameras]
    for cap, cam in zip(captures, cameras):
        if not cap.isOpened():
            raise SystemExit(f"cannot open camera_{cam}.mp4 in {video_dir}")

    num_joints = len(rt_settings.nlf_indices)
    shape = (len(picks), len(cameras), num_joints)
    poses3d = np.full(shape + (3,), np.nan, np.float32)
    keypoints = np.full(shape + (2,), np.nan, np.float32)

    for slot in range(len(picks)):
        frames = []
        for i, cam in enumerate(cameras):
            captures[i].set(cv2.CAP_PROP_POS_FRAMES, matched[cam][slot])
            ok, frame = captures[i].read()
            if not ok:
                break
            frames.append(frame)
        if len(frames) != len(cameras):
            poses3d = poses3d[:slot]
            keypoints = keypoints[:slot]
            break
        output, _, _, _ = estimator.estimate_from_frames(frames)
        views = extract_views(output, len(cameras))
        for i in range(len(cameras)):
            if views.poses3d[i] is not None:
                poses3d[slot, i] = views.poses3d[i][:num_joints]
            if views.keypoints[i] is not None:
                keypoints[slot, i] = views.keypoints[i][:num_joints]
        if slot % 100 == 0:
            print(f"  {slot}/{len(picks)} frames", flush=True)

    for cap in captures:
        cap.release()
    return keypoints, poses3d, list(rt_settings.marker_names)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 2],
                        help="camera ids, in rig order; adjacent ones are paired")
    parser.add_argument("--videos", default=None, metavar="DIR",
                        help="recording holding camera_<i>.mp4 for every camera")
    parser.add_argument("--height", type=float, default=None, metavar="M",
                        help="subject stature in metres; sets the metric scale")
    parser.add_argument("--out", default=os.path.join(repo_path, "config", "cam_params"),
                        help="calibration root to write (COMFI layout)")
    parser.add_argument("--calib", default=None, metavar="DIR",
                        help="where the intrinsics live (default: --out)")
    parser.add_argument("--step", type=int, default=5, metavar="N",
                        help="use every Nth frame (default 5)")
    parser.add_argument("--cache", default=None, metavar="FILE",
                        help="keypoint cache: reused when it exists, written when not")
    parser.add_argument("--install", nargs="?", const=True, default=None, metavar="PATH",
                        help="also copy the result into RT-COSMIK "
                             "(default: its settings.cam_calib_path)")
    parser.add_argument("--force", action="store_true",
                        help="calibrate even if the recording is too flat to support it")
    args = parser.parse_args()

    cameras = args.cameras
    if len(cameras) < 2:
        raise SystemExit("at least two cameras are needed to relate them")
    if args.height is None:
        raise SystemExit(
            "--height is required: the reconstruction is only defined up to one\n"
            "  global scale, and the subject's stature is what fixes it.")

    calib_root = args.calib or args.out
    intrinsics, distortions = [], []
    for cam in cameras:
        path = intrinsics_path(calib_root, cam)
        if not os.path.isfile(path):
            raise SystemExit(
                f"missing intrinsics for camera {cam}: {path}\n"
                "  Run scripts/calibrate_cameras.py first; this script replaces only "
                "the stereo half.")
        mtx, dist = load_intrinsics(path)
        intrinsics.append(np.asarray(mtx, float))
        distortions.append(np.asarray(dist, float).reshape(-1))

    if args.cache and os.path.isfile(args.cache):
        cached = np.load(args.cache, allow_pickle=False)
        keypoints, poses3d = cached["keypoints"], cached["poses3d"]
        marker_names = [str(n) for n in cached["marker_names"]]
        print(f"Reusing {keypoints.shape[0]} frames from {args.cache}")
        if list(cached["cameras"]) != list(cameras):
            raise SystemExit(f"cache holds cameras {list(cached['cameras'])}, "
                             f"not {cameras}")
    else:
        if not args.videos:
            raise SystemExit("--videos is required unless --cache points at an "
                             "existing extraction")
        if not os.path.isdir(args.videos):
            raise SystemExit(f"no such recording: {args.videos}")
        print(f"Extracting poses from {args.videos}")
        keypoints, poses3d, marker_names = extract(
            args.videos, cameras, intrinsics, distortions, args.step)
        if args.cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
            np.savez_compressed(args.cache, keypoints=keypoints, poses3d=poses3d,
                                cameras=np.asarray(cameras),
                                marker_names=np.asarray(marker_names))
            print(f"  cached to {args.cache}")

    seen = np.isfinite(poses3d[:, :, 0, 0]).sum(axis=0)
    print(f"\n{keypoints.shape[0]} frames sampled; "
          f"subject detected in {seen.tolist()} of them per camera")
    if min(seen) < 30:
        raise SystemExit("too few frames saw the subject in every camera to calibrate")

    thickness = cloud_thickness(poses3d[:, 0])
    usable, message = describe_conditioning(thickness)
    print(f"\nRecording shape\n  {message}")
    if not usable and not args.force:
        raise SystemExit("  Refusing to calibrate on this recording (--force to "
                         "override).")

    print("\nCamera pairs")
    poses = [(np.eye(3), np.zeros(3))]
    for i, cam in enumerate(cameras[1:], start=1):
        result = register_pair(
            keypoints[:, 0], keypoints[:, i], poses3d[:, 0], poses3d[:, i],
            intrinsics[0], distortions[0], intrinsics[i], distortions[i])
        if result is None:
            raise SystemExit(f"  camera_{cameras[0]} -> camera_{cam}: too few frames "
                             "showed the subject to both cameras")
        poses.append((result["R"], result["T"]))
        print(f"  camera_{cameras[0]} -> camera_{cam}: {result['frames']} shared frames, "
              f"{100 * result['inlier_frac']:.0f}% inliers")

    points = triangulate(np.swapaxes(keypoints, 0, 1), poses, intrinsics, distortions)
    scale = stature_scale(points, marker_names, args.height)
    poses = [(R, T * scale) for R, T in poses]
    print(f"\nScale from a {args.height:.2f} m subject: x{scale:.4f}")

    print("\nWriting stereo pairs")
    for a, b in zip(range(len(cameras) - 1), range(1, len(cameras))):
        rotation = poses[b][0] @ poses[a][0].T
        translation = poses[b][1] - rotation @ poses[a][1]
        path = save_stereo_pose(intrinsics[a], distortions[a],
                                intrinsics[b], distortions[b],
                                rotation, translation, 0.0,
                                cameras[a], cameras[b], args.out)
        print(f"  camera_{cameras[a]} -> camera_{cameras[b]}: "
              f"baseline {np.linalg.norm(translation):.4f} m "
              f"-> {os.path.relpath(path, args.out)}")

    print(f"\nWritten under {args.out}")
    if args.install is not None:
        install_to_rtcosmik(args.out, None if args.install is True else args.install)
    print("Anchor the world frame next: scripts/set_world_frame.py")
    print("\nThis is a rough calibration: expect the cameras to be placed about 3% of\n"
          "their baseline out, roughly 5x worse than a checkerboard. Prefer\n"
          "calibrate_cameras.py for any pair a checkerboard can reach. Check the\n"
          "baselines above look like the room before trusting them.")


if __name__ == "__main__":
    main()
