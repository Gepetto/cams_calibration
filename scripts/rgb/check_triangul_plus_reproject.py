#!/usr/bin/env python3
"""
Debug script: check reprojection RMSE using CHECKERBOARD extrinsics only.

- Loads intrinsics (cX_params_color.yaml)
- Loads checkerboard extrinsics (c{base}_to_c{cam}_params_color.yaml)
- Extracts RTMPose keypoints from given videos
- Triangulates using checkerboard extrinsics
- Reprojects and computes RMSE in pixels

Optionally, also evaluates RMSE with INVERTED extrinsics to detect
direction-convention mistakes.
"""

import os
import sys
import cv2
import numpy as np

# Adjust this path resolution if needed, same as in calibrate_extrinsics_rtmpose.py
script_path = os.path.abspath(__file__)
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from utils.settings import Settings
from utils.calib_utils import (
    load_cam_params,
    load_cam_to_cam_params,
    PoseTrackerEstimator,
    triangulate_points,
)

settings = Settings()


def extract_keypoints_from_video(video_path: str, estimator: PoseTrackerEstimator):
    """
    Same logic as in calibrate_extrinsics_rtmpose.py, slightly repackaged.
    Returns:
        keypoints_arr: (T, J, 2)
        conf_arr:      (T, J)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    all_kps = []
    all_conf = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = estimator.estimate(frame)
        keypoints, bboxes, _ = results

        if keypoints is None or len(keypoints) == 0:
            # No person detected: fill with NaNs and zeros (if we already know J)
            if all_kps:
                J = all_kps[-1].shape[0]
                all_kps.append(np.full((J, 2), np.nan, dtype=np.float32))
                all_conf.append(np.zeros((J,), dtype=np.float32))
            # If it's the very first frame and nothing is detected, we just skip
            continue

        # Choose person with largest bounding box (if multiple)
        if bboxes is not None and len(bboxes) > 1:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            person_idx = int(np.argmax(areas))
        else:
            person_idx = 0

        kps = keypoints[person_idx]  # (J, 3)
        xy = kps[:, :2].astype(np.float32)
        conf = kps[:, 2].astype(np.float32)

        all_kps.append(xy)
        all_conf.append(conf)

    cap.release()

    if not all_kps:
        raise RuntimeError(f"No keypoints extracted from {video_path}")

    keypoints_arr = np.stack(all_kps, axis=0)  # (T, J, 2)
    conf_arr = np.stack(all_conf, axis=0)      # (T, J)
    return keypoints_arr, conf_arr


def compute_reprojection_rmse_multicam(
    kps_by_cam,
    conf_by_cam,
    K_by_cam,
    D_by_cam,
    base_cam,
    extr_R,
    extr_T,
    conf_thresh=0.5,
    J3d=26,
):
    """
    Copy of the multi-camera RMSE logic from calibrate_extrinsics_rtmpose.py.
    extr_R[cam], extr_T[cam] are assumed to be R_base_to_cam, t_base_to_cam.
    """
    cams = sorted(kps_by_cam.keys())
    assert base_cam in cams

    # Align frame count
    T = min(kps_by_cam[cam].shape[0] for cam in cams)

    # Joint consistency
    J = kps_by_cam[base_cam].shape[1]
    if J < J3d:
        raise ValueError(f"Expected at least {J3d} joints, got J={J}")
    J_eff = J3d

    # Build lists for triangulation
    mtxs = []
    dists = []
    projections = []
    Rt_by_cam = {}

    for cam in cams:
        K = K_by_cam[cam]
        D = D_by_cam[cam]

        if cam == base_cam:
            R = np.eye(3)
            t = np.zeros(3)
        else:
            if cam not in extr_R or cam not in extr_T:
                raise ValueError(f"Missing extrinsics for camera {cam}")
            R = extr_R[cam]
            t = np.asarray(extr_T[cam]).reshape(3)

        Rt_by_cam[cam] = (R, t)
        mtxs.append(K)
        dists.append(D)
        P = np.hstack([R, t.reshape(3, 1)])
        projections.append(P)

    total_err = 0.0
    total_count = 0

    for t_idx in range(T):
        keypoints_list = []
        conf_all = np.ones((J_eff,), dtype=bool)

        for cam in cams:
            kps_frame = kps_by_cam[cam][t_idx, :J_eff, :]
            keypoints_list.append(kps_frame)
            conf_frame = conf_by_cam[cam][t_idx, :J_eff]
            conf_all &= (conf_frame > conf_thresh)

        if not np.any(conf_all):
            continue

        # 3D triangulation in base_cam frame
        p3d_frame = triangulate_points(
            keypoints_list,
            mtxs,
            dists,
            projections,
        )  # (J_eff, 3)

        # Reprojection
        for j in range(J_eff):
            if not conf_all[j]:
                continue

            X = p3d_frame[j]

            for cam in cams:
                R_cam, t_cam = Rt_by_cam[cam]
                K_cam = K_by_cam[cam]

                Xc = R_cam @ X + t_cam
                if Xc[2] <= 1e-6:
                    continue

                x_n = Xc[:2] / Xc[2]
                p_hat = (K_cam @ np.array([x_n[0], x_n[1], 1.0]))[:2]
                p_true = kps_by_cam[cam][t_idx, j]

                err = np.sum((p_hat - p_true) ** 2)
                total_err += err
                total_count += 1

    if total_count == 0:
        return np.nan

    rmse = float(np.sqrt(total_err / total_count))
    return rmse


def debug_checkerboard_rmse(
    recorded_session,
    config_dir,
    base_cam,
    conf_thresh=0.5,
):
    """
    recorded_session: dict[cam_idx -> video_path], cam_idx can be int or str.
    config_dir: path to config folder (same as in your repo).
    base_cam: base camera index (same type as keys in recorded_session).
    """
    # Normalize keys to a consistent type
    # Here we keep them as strings to match your existing config
    cam_indices = sorted(recorded_session.keys())
    if base_cam not in cam_indices:
        raise ValueError(f"base_cam {base_cam} not in recorded_session keys {cam_indices}")

    cam_params_dir = os.path.join(config_dir, "cam_params")

    # 1) Load intrinsics
    K_by_cam = {}
    D_by_cam = {}
    for cam in cam_indices:
        intr_path = os.path.join(cam_params_dir, f"c{cam}_params_color.yaml")
        if not os.path.exists(intr_path):
            raise FileNotFoundError(f"Missing intrinsics for cam {cam}: {intr_path}")
        K, D = load_cam_params(intr_path)
        K_by_cam[cam] = K
        D_by_cam[cam] = D

    # 2) Load checkerboard extrinsics (base_cam -> other cams)
    extr_R_cb = {}
    extr_T_cb = {}
    for cam in cam_indices:
        if cam == base_cam:
            continue
        stereo_cb_path = os.path.join(
            cam_params_dir,
            f"c{base_cam}_to_c{cam}_params_color.yaml",
        )
        if not os.path.exists(stereo_cb_path):
            raise FileNotFoundError(
                f"Missing checkerboard stereo file for pair (base={base_cam}, cam={cam}): {stereo_cb_path}"
            )
        R_cb, T_cb = load_cam_to_cam_params(stereo_cb_path)
        extr_R_cb[cam] = R_cb
        extr_T_cb[cam] = T_cb.reshape(3)

    # 3) Extract keypoints per camera
    estimators = {
        cam: PoseTrackerEstimator(
            det_model=settings.det_model_path,
            pose_model=settings.pose_model_path,
        )
        for cam in cam_indices
    }

    kps_by_cam = {}
    conf_by_cam = {}

    for cam, video_path in recorded_session.items():
        print(f"Extracting keypoints from cam {cam} video: {video_path}")
        kps, conf = extract_keypoints_from_video(video_path, estimators[cam])
        kps_by_cam[cam] = kps
        conf_by_cam[cam] = conf

    # 4) Align sequence lengths
    T_min = min(kps.shape[0] for kps in kps_by_cam.values())
    for cam in cam_indices:
        kps_by_cam[cam] = kps_by_cam[cam][:T_min]
        conf_by_cam[cam] = conf_by_cam[cam][:T_min]

    # 5) RMSE with checkerboard extrinsics (direct)
    print("\n=== RMSE with CHECKERBOARD extrinsics (as loaded) ===")
    rmse_cb = compute_reprojection_rmse_multicam(
        kps_by_cam,
        conf_by_cam,
        K_by_cam,
        D_by_cam,
        base_cam=base_cam,
        extr_R=extr_R_cb,
        extr_T=extr_T_cb,
        conf_thresh=conf_thresh,
        J3d=26,
    )
    print(f"Global RMSE (checkerboard, direct): {rmse_cb:.3f} px")

    # 6) RMSE with INVERTED checkerboard extrinsics (to check direction)
    print("\n=== RMSE with INVERTED checkerboard extrinsics ===")
    extr_R_inv = {}
    extr_T_inv = {}
    for cam in cam_indices:
        if cam == base_cam:
            continue
        R = extr_R_cb[cam]
        t = extr_T_cb[cam]
        R_inv = R.T
        t_inv = -R_inv @ t
        extr_R_inv[cam] = R_inv
        extr_T_inv[cam] = t_inv

    rmse_cb_inv = compute_reprojection_rmse_multicam(
        kps_by_cam,
        conf_by_cam,
        K_by_cam,
        D_by_cam,
        base_cam=base_cam,
        extr_R=extr_R_inv,
        extr_T=extr_T_inv,
        conf_thresh=conf_thresh,
        J3d=26,
    )
    print(f"Global RMSE (checkerboard, inverted): {rmse_cb_inv:.3f} px")


def main():
    # Adapt these paths/indices to your setup

    config_dir = os.path.join(repo_path, "config")

    # Example: same as in your current calibrate_extrinsics_rtmpose.py main()
    recorded_session = {
        "0": "/root/workspace/ros_ws/src/rt-cosmik/tests/full/data/camera_0.mp4",
        "2": "/root/workspace/ros_ws/src/rt-cosmik/tests/full/data/camera_2.mp4",
    }

    base_cam = "0"
    debug_checkerboard_rmse(
        recorded_session=recorded_session,
        config_dir=config_dir,
        base_cam=base_cam,
        conf_thresh=0.5,
    )


if __name__ == "__main__":
    main()
