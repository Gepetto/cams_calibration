# To run the code from repo root : python3 scripts/rgb/calibrate_cameras.py
# Modes:
#   (default)              live capture + checkerboard intrinsics
#                           adjacent-pair extrinsics, composed sequentially.
#   --offline               same as above but skips live capture, uses existing images.
#   --soder                 checkerboard intrinsics + world-frame extrinsics from soder{N}.txt
#                           + composition into camera_0_to_camera_N.yaml.
#   --num-cameras {2,4}     2-camera (c0,c2) or 4-camera (c0,c2,c4,c6) rig.

import os
import sys
import argparse

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.settings import Settings
settings = Settings()

import numpy as np
import cv2
from utils.calib_utils import (
    calibrate_camera,
    save_cam_params,
    load_cam_params,
    stereo_calibrate,
    save_cam_to_cam_params,
    list_cameras_with_v4l2,
    save_pose_to_yaml,
    load_camera_extrinsics,
    compose_via_world,
    run_soder_world_frame,
    camera_ids_for,
    intrinsics_path,
    extrinsics_path,
    cam_to_cam_path,
    calib_images_dir,
    has_images,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate 2 or 4 RGB cameras: intrinsics + extrinsics."
    )
    parser.add_argument("--offline", action="store_true",
                         help="Skip live camera capture and use images already saved on disk.")
    parser.add_argument("--soder", action="store_true",
                         help="Use soder{N}.txt (mocap-based) for extrinsics.")
    parser.add_argument("--num-cameras", type=int, choices=[2, 4], default=2,
                         help="Number of physical cameras: 2 (c0, c2) or 4 (c0, c2, c4, c6).")
    return parser.parse_args()


def run_intrinsics_calibration(cam_ids, imgs_dirs, cam_params_dir):
    intrinsics = {}
    calibrated_cam_ids = []

    for cam_id, img_dir in zip(cam_ids, imgs_dirs):
        if not has_images(img_dir):
            print(f"[SKIP] camera_{cam_id}: no images found in {img_dir}")
            continue

        reproj, mtx, dist = calibrate_camera(images_folder=os.path.join(img_dir, "*.png"))
        out_path = intrinsics_path(cam_params_dir, cam_id)
        save_cam_params(mtx, dist, reproj, out_path)
        print(f"rmse cam{cam_id} RGB = {reproj} (saved {out_path})")
        intrinsics[cam_id] = (mtx, dist)
        calibrated_cam_ids.append(cam_id)

    cv2.destroyAllWindows()
    return intrinsics, calibrated_cam_ids


def run_checkerboard_extrinsics(cam_ids, imgs_dirs, cam_params_dir):
    mtx, dist = {}, {}
    for cam_id in cam_ids:
        path = intrinsics_path(cam_params_dir, cam_id)
        if os.path.isfile(path):
            mtx[cam_id], dist[cam_id] = load_cam_params(path)

    if cam_ids[0] in mtx:
        ref_id = cam_ids[0]
        c0_ext = extrinsics_path(cam_params_dir, ref_id)
        save_pose_to_yaml(
            np.eye(3), np.zeros((3, 1)), c0_ext,
            frame_from=f"camera_{ref_id}", frame_to="world",
            scale_factor=1.0, rms_error=0.0, source_file="checkerboard_reference"
        )
        print(f"[CHECKERBOARD] Set camera_{ref_id} as world origin (saved {c0_ext})")

    pair_R, pair_T = {}, {}
    for a, b in zip(cam_ids[:-1], cam_ids[1:]):
        if a not in mtx or b not in mtx:
            continue

        dir_a, dir_b = imgs_dirs[cam_ids.index(a)], imgs_dirs[cam_ids.index(b)]
        if not has_images(dir_a) or not has_images(dir_b):
            continue

        rmse, R_ab, T_ab = stereo_calibrate(
            mtx[a], dist[a], mtx[b], dist[b],
            os.path.join(dir_a, "*.png"),
            os.path.join(dir_b, "*.png"),
        )
        pair_R[(a, b)], pair_T[(a, b)] = R_ab, T_ab

        out_path = cam_to_cam_path(cam_params_dir, a, b)
        save_cam_to_cam_params(mtx[a], dist[a], mtx[b], dist[b], R_ab, T_ab, rmse, out_path)
        print(f"rmse cam{a}->cam{b}: {rmse} (saved {out_path})")

    R_running, T_running = np.eye(3), np.zeros((3, 1))
    for a, b in zip(cam_ids[:-1], cam_ids[1:]):
        if (a, b) not in pair_R:
            break

        R_ab, T_ab = pair_R[(a, b)], pair_T[(a, b)]
        R_running = R_ab @ R_running
        T_running = R_ab @ T_running + T_ab

        ext_path = extrinsics_path(cam_params_dir, b)
        save_pose_to_yaml(
            R_running, T_running, ext_path,
            frame_from=f"camera_{b}", frame_to="world",
            scale_factor=1.0, rms_error=0.0, source_file="checkerboard"
        )
        print(f"[CHECKERBOARD] Saved {ext_path}")

        if len(cam_ids) > 2 and b != cam_ids[1]:
            out_path = cam_to_cam_path(cam_params_dir, cam_ids[0], b)
            save_cam_to_cam_params(mtx[cam_ids[0]], dist[cam_ids[0]], mtx[b], dist[b],
                                    R_running, T_running, 0.0, out_path)
            print(f"Saved {out_path}")


def run_soder_composition(cam_ids, cam_params_dir):
    K0_path = intrinsics_path(cam_params_dir, cam_ids[0])
    R0_path = extrinsics_path(cam_params_dir, cam_ids[0])
    if not os.path.isfile(K0_path) or not os.path.isfile(R0_path):
        print("[SKIP] Missing camera_0 intrinsics or extrinsics for Soder composition.")
        return

    K0, D0 = load_cam_params(K0_path)
    R_c0_w, d_c0_w = load_camera_extrinsics(R0_path)

    for cam_id in cam_ids[1:]:
        Kn_path = intrinsics_path(cam_params_dir, cam_id)
        Rn_path = extrinsics_path(cam_params_dir, cam_id)
        if not os.path.isfile(Kn_path) or not os.path.isfile(Rn_path):
            continue

        Kn, Dn = load_cam_params(Kn_path)
        R_cn_w, d_cn_w = load_camera_extrinsics(Rn_path)
        R_rel, d_rel = compose_via_world(R_c0_w, d_c0_w, R_cn_w, d_cn_w)

        out_path = cam_to_cam_path(cam_params_dir, cam_ids[0], cam_id)
        save_cam_to_cam_params(K0, D0, Kn, Dn, R_rel, d_rel, 0.0, out_path)
        print(f"Saved composed pose {out_path}")


def main():
    args = parse_args()
    cam_params_dir = os.path.join(repo_path, "config", "cam_params")
    os.makedirs(cam_params_dir, exist_ok=True)

    cam_ids = camera_ids_for(args.num_cameras)
    imgs_dirs = [calib_images_dir(repo_path, cid) for cid in cam_ids]
    for d in imgs_dirs:
        os.makedirs(d, exist_ok=True)

    if args.soder:
        run_intrinsics_calibration(cam_ids, imgs_dirs, cam_params_dir)
        run_soder_world_frame(cam_ids, cam_params_dir)
        run_soder_composition(cam_ids, cam_params_dir)
        return

    if not args.offline:
        camera_dict = list_cameras_with_v4l2()
        device_indices = list(camera_dict.keys())[:len(cam_ids)]

        if len(device_indices) < len(cam_ids):
            sys.exit(f"[ERROR] Found only {len(device_indices)} camera devices, but requested {len(cam_ids)}.")

        captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in device_indices]

        for cap in captures:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
                cap.set(cv2.CAP_PROP_FPS, settings.fs)

        cv2.namedWindow("RGB Camera Calibration Capture", cv2.WINDOW_NORMAL)
        img_idx = 0
        try:
            while True:
                frames = [cap.read()[1] for cap in captures]
                if not all(f is not None for f in frames):
                    continue

                resized = [cv2.resize(f, (640, 480), interpolation=cv2.INTER_NEAREST) for f in frames]
                cv2.imshow("RGB Camera Calibration Capture", np.hstack(resized))

                c = cv2.waitKey(10)
                if c == ord('s'):
                    print("Images taken")
                    for frame, d in zip(frames, imgs_dirs):
                        cv2.imwrite(os.path.join(d, f"img_{img_idx}.png"), frame)
                    img_idx += 1
                elif c == ord('q'):
                    break
                if cv2.getWindowProperty("RGB Camera Calibration Capture", cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            for cap in captures:
                cap.release()
            cv2.destroyAllWindows()

    run_intrinsics_calibration(cam_ids, imgs_dirs, cam_params_dir)
    run_checkerboard_extrinsics(cam_ids, imgs_dirs, cam_params_dir)


if __name__ == "__main__":
    main()