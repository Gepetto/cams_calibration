# To run from repo root : python3 scripts/rgb/calibrate_extrinsics_rtmpose.py

import os
import sys
import cv2
import numpy as np

# Resolve repo root similarly to calibrate_cameras.py
script_path = os.path.abspath(__file__)
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add repo root to PYTHONPATH
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from utils.settings import Settings
from utils.calib_utils import (
    list_cameras_with_v4l2,
    load_cam_params,
    save_cam_to_cam_params,
    load_cam_to_cam_params,
    PoseTrackerEstimator,
    triangulate_points,
)

settings = Settings()

# -------------------------------------------------------------------
#  Helpers for video recording
# -------------------------------------------------------------------


def make_video_dirs(config_dir: str, camera_dict):
    """
    Create per-camera directories to store calibration videos.

    Returns:
        video_dirs: dict[int, str] mapping cam_index -> directory path
    """
    video_dirs = {}
    for cam_idx in sorted(camera_dict.keys()):
        cam_dir = os.path.join(config_dir, f"videos_calib_cam_{cam_idx}", "color")
        os.makedirs(cam_dir, exist_ok=True)
        video_dirs[cam_idx] = cam_dir
    return video_dirs


def build_mosaic(frames, cols=2):
    """
    Build a simple mosaic (grid) image from a list of frames (all same size).
    """
    if not frames:
        return None
    h, w, c = frames[0].shape
    n = len(frames)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))

    # Fill with black images if needed
    padded = frames + [np.zeros_like(frames[0]) for _ in range(rows * cols - n)]

    rows_imgs = []
    for r in range(rows):
        row = np.hstack(padded[r * cols : (r + 1) * cols])
        rows_imgs.append(row)
    mosaic = np.vstack(rows_imgs)
    return mosaic


def record_calibration_videos(config_dir: str):
    """
    Open all available cameras, show a live RTMPose preview, and let the user
    toggle recording of synchronized calibration videos for *all* cameras.

    Press:
        's' -> start/stop recording a clip for all cameras
        'q' -> quit

    Returns:
        recorded_sessions: List[Dict[int, str]]
            Each element is a dict mapping cam_index -> video_path for one session.
    """
    camera_dict = list_cameras_with_v4l2()
    if not camera_dict:
        raise RuntimeError("No cameras detected by v4l2-ctl.")

    cam_indices = sorted(camera_dict.keys())
    print("Detected cameras:", camera_dict)

    # Open captures
    captures = []
    for cam_idx in cam_indices:
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"WARNING: could not open camera {cam_idx}")
        else:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
            cap.set(cv2.CAP_PROP_FPS, settings.fs)
        captures.append(cap)

    if len(captures) < 2:
        print("WARNING: fewer than 2 cameras detected; recording still works,"
              " but extrinsics calibration will need at least 2.")
    # Prepare output dirs
    video_dirs = make_video_dirs(config_dir, camera_dict)
    print("Calibration videos will be stored under:")
    for cam_idx, d in video_dirs.items():
        print(f"  cam {cam_idx}: {d}")

    # Pose estimators (one per camera)
    pose_estimators = {
        cam_idx: PoseTrackerEstimator(
            det_model=settings.det_model_path,
            pose_model=settings.pose_model_path,
        )
        for cam_idx in cam_indices
    }

    # Recording state
    recording = False
    session_id = 0
    writers = {cam_idx: None for cam_idx in cam_indices}
    current_session_paths = None
    recorded_sessions = []

    # Writer codec (MP4)
    mp4_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = float(settings.fs)

    try:
        while True:
            # Grab frames
            frames = {}
            for cam_idx, cap in zip(cam_indices, captures):
                if not cap.isOpened():
                    frames[cam_idx] = None
                    continue
                ret, frame = cap.read()
                frames[cam_idx] = frame if ret else None

            # Require all cameras to succeed for a "synchronized" frame
            if any(frames[cam_idx] is None for cam_idx in cam_indices):
                # Just skip this loop if a frame is missing
                continue

            # RTMPose overlay + mosaic preview
            preview_tiles = []
            for cam_idx in cam_indices:
                frame = frames[cam_idx]
                # RTMPose tracking + visualization (one window per cam)
                results = pose_estimators[cam_idx].estimate(frame)
                pose_estimators[cam_idx].visualize(frame, results, idx=cam_idx)

                # Small tile for global mosaic
                tile = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                cv2.putText(
                    tile,
                    f"Cam {cam_idx}",
                    (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                preview_tiles.append(tile)

            mosaic = build_mosaic(preview_tiles, cols=2)
            if recording and mosaic is not None:
                cv2.putText(
                    mosaic,
                    f"REC #{session_id}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            if mosaic is not None:
                cv2.imshow("RGB calib mosaic", mosaic)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                # toggle recording
                if not recording:
                    # start new session
                    recording = True
                    current_session_paths = {}
                    print(f"\n=== Starting recording session #{session_id} ===")
                    # create writers for each camera
                    for cam_idx in cam_indices:
                        frame = frames[cam_idx]
                        h, w, _ = frame.shape
                        out_dir = video_dirs[cam_idx]
                        out_path = os.path.join(
                            out_dir, f"calib_video_{session_id:03d}.mp4"
                        )
                        writers[cam_idx] = cv2.VideoWriter(
                            out_path, mp4_fourcc, fps, (w, h)
                        )
                        current_session_paths[cam_idx] = out_path
                        print(f"  cam {cam_idx}: {out_path}")
                else:
                    # stop current session
                    print(f"=== Stopping recording session #{session_id} ===\n")
                    recording = False
                    for cam_idx in cam_indices:
                        if writers[cam_idx] is not None:
                            writers[cam_idx].release()
                            writers[cam_idx] = None
                    if current_session_paths is not None:
                        recorded_sessions.append(current_session_paths)
                        current_session_paths = None
                    session_id += 1

            if key == ord("q"):
                print("Quitting capture loop...")
                break

            # Write frames if recording
            if recording:
                for cam_idx in cam_indices:
                    if writers[cam_idx] is not None:
                        writers[cam_idx].write(frames[cam_idx])

    finally:
        if recording:
            for cam_idx in cam_indices:
                if writers[cam_idx] is not None:
                    writers[cam_idx].release()
        for cap in captures:
            cap.release()
        cv2.destroyAllWindows()

    print("\nRecorded sessions:")
    for i, sess in enumerate(recorded_sessions):
        print(f"  Session #{i}:")
        for cam_idx, path in sess.items():
            print(f"    cam {cam_idx}: {path}")

    return recorded_sessions


# -------------------------------------------------------------------
#  RTMPose-based keypoint extraction from videos
# -------------------------------------------------------------------


def extract_keypoints_from_video(
    video_path: str,
    det_model_path: str,
    pose_model_path: str
):
    """
    Run PoseTrackerEstimator on a video and return (keypoints, confidences).

    Returns:
        keypoints: (T, J, 2) in pixel coordinates
        conf:      (T, J)    confidence scores
    """
    estimator = PoseTrackerEstimator(
        det_model=det_model_path,
        pose_model=pose_model_path,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    all_kps = []
    all_conf = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = estimator.estimate(frame)
        # According to visualize(): results = (keypoints, bboxes, ...)
        keypoints, bboxes, _ = results

        if keypoints is None or len(keypoints) == 0:
            # no person detected
            if all_kps:
                J = all_kps[-1].shape[0]
            else:
                # can't infer J, skip this frame
                continue
            all_kps.append(np.full((J, 2), np.nan, dtype=np.float32))
            all_conf.append(np.zeros((J,), dtype=np.float32))
            continue

        # Choose the person with the largest bounding box (if multiple)
        if bboxes is not None and len(bboxes) > 1:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            person_idx = int(np.argmax(areas))
        else:
            person_idx = 0

        kps = keypoints[person_idx]  # (J, 3): x, y, score
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


# -------------------------------------------------------------------
#  Extrinsics from RTMPose keypoints (OpenCV-only version)
# -------------------------------------------------------------------


def build_correspondences(kps1, kps2, conf1, conf2, conf_thresh=0.8):
    """
    kps*: (T, J, 2), conf*: (T, J)
    Returns:
        pts1, pts2: (N, 2) in pixels
    """
    T, J, _ = kps1.shape
    pts1 = []
    pts2 = []
    for t in range(T):
        for j in range(J):
            if conf1[t, j] > conf_thresh and conf2[t, j] > conf_thresh:
                pts1.append(kps1[t, j])
                pts2.append(kps2[t, j])
    if not pts1:
        raise RuntimeError("No valid correspondences above confidence threshold.")
    return np.asarray(pts1, dtype=np.float32), np.asarray(pts2, dtype=np.float32)


def estimate_extrinsics_from_keypoints(
    kps1,
    kps2,
    conf1,
    conf2,
    K1,
    D1,
    K2,
    D2,
    conf_thresh=0.8,
):
    """
    Estimate (R, T) between *camera 2* and *camera 1* from RTMPose 2D keypoints.

    Pipeline:
        - build 2D-2D correspondences across all frames/joints
        - undistort to normalized coordinates
        - estimate essential matrix with RANSAC
        - recover pose (R, t)
        - compute reprojection RMSE (for info, using triangulate_points)
    """
    pts1_pix, pts2_pix = build_correspondences(kps1, kps2, conf1, conf2, conf_thresh)

    # Undistort to normalized coordinates
    pts1_norm = cv2.undistortPoints(
        pts1_pix.reshape(-1, 1, 2), K1, D1
    ).reshape(-1, 2)
    pts2_norm = cv2.undistortPoints(
        pts2_pix.reshape(-1, 1, 2), K2, D2
    ).reshape(-1, 2)

    # Essential matrix in normalized space (focal=1, pp=(0,0))
    E, mask = cv2.findEssentialMat(
        pts1_norm,
        pts2_norm,
        1.0,
        (0.0, 0.0),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1e-3,
    )
    if E is None:
        raise RuntimeError("findEssentialMat failed.")

    mask = mask.ravel().astype(bool)
    pts1_in = pts1_norm[mask]
    pts2_in = pts2_norm[mask]

    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in)

    # Compute reprojection RMSE using triangulate_points (pairwise case)
    rmse = compute_reprojection_rmse_pair(kps1, kps2, conf1, conf2, K1, D1, K2, D2, R, t, conf_thresh)
    return R, t, rmse


def compute_reprojection_rmse_pair(
    kps1,
    kps2,
    conf1,
    conf2,
    K1,
    D1,
    K2,
    D2,
    R,
    t,
    conf_thresh=0.8,
):
    """
    Use current (R, t) to triangulate each joint (via triangulate_points)
    and reproject back to both cameras; compute RMS error in pixels.

    - kps1, kps2: (T, J, 2)
    - conf1, conf2: (T, J)
    - R, t: transform from cam1 frame to cam2 frame (X2 = R * X1 + t)
    - Assumes RTMPose body26 => at least 26 joints.
    """
    T, J, _ = kps1.shape
    if J < 26:
        raise ValueError(
            f"triangulate_points currently assumes 26 joints (body26). "
            f"Got J = {J}."
        )

    # We will use only the first 26 joints for triangulation
    J3d = 26

    # good convention
    R = R.T
    t = -R @ t

    # Prepare camera matrices/distortions/projections in normalized space
    mtxs = [K1, K2]
    dists = [D1, D2]
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])        # cam1 at origin
    P2 = np.hstack([R, t.reshape(3, 1)])                 # cam2 in cam1 frame
    projections = [P1, P2]

    total_err = 0.0
    total_count = 0

    for t_idx in range(T):
        # For this frame, assemble 2D keypoints for each camera
        # Use only the first 26 joints (body26)
        kps_frame_cam1 = kps1[t_idx, :J3d, :]
        kps_frame_cam2 = kps2[t_idx, :J3d, :]

        keypoints_list_frame = [kps_frame_cam1, kps_frame_cam2]

        # 3D triangulation in cam1 frame
        p3d_frame = triangulate_points(
            keypoints_list_frame,
            mtxs,
            dists,
            projections,
        )  # shape (26, 3)

        # Reprojection + error accumulation
        for j in range(J3d):
            if conf1[t_idx, j] <= conf_thresh or conf2[t_idx, j] <= conf_thresh:
                continue

            X = p3d_frame[j]  # (3,)

            # Cam1: X already expressed in cam1 frame
            Xc1 = X
            x1 = Xc1[:2] / Xc1[2]
            p1_hat = (K1 @ np.array([x1[0], x1[1], 1.0]))[:2]
            p1_true = kps1[t_idx, j]

            # Cam2: transform to cam2 frame using R, t
            Xc2 = R @ X + t.reshape(3)
            x2 = Xc2[:2] / Xc2[2]
            p2_hat = (K2 @ np.array([x2[0], x2[1], 1.0]))[:2]
            p2_true = kps2[t_idx, j]

            err1 = np.sum((p1_hat - p1_true) ** 2)
            err2 = np.sum((p2_hat - p2_true) ** 2)

            total_err += err1 + err2
            total_count += 2

    if total_count == 0:
        return np.nan

    rmse = float(np.sqrt(total_err / total_count))
    return rmse

def compute_reprojection_rmse_multicam(
    kps_by_cam,
    conf_by_cam,
    K_by_cam,
    D_by_cam,
    base_cam,
    extr_R,
    extr_T,
    conf_thresh=0.8,
    J3d=26,
):
    """
    Compute global reprojection RMSE (in pixels) over ALL cameras, using ALL cameras
    for triangulation, given extrinsics expressed in the base camera frame.

    Args:
        kps_by_cam:  dict[int, np.ndarray], each (T, J, 2)
        conf_by_cam: dict[int, np.ndarray], each (T, J)
        K_by_cam:    dict[int, (3,3)]
        D_by_cam:    dict[int, (dist_len,)]
        base_cam:    int, index of the base camera (world frame)
        extr_R:      dict[int, (3,3)], extr_R[cam] is R_base_to_cam, cam != base_cam
        extr_T:      dict[int, (3,)], extr_T[cam] is t_base_to_cam
        conf_thresh: scalar confidence threshold
        J3d:         number of joints to triangulate (26 for body26)

    Returns:
        rmse_pixels: float
    """
    cams = sorted(kps_by_cam.keys())
    assert base_cam in cams

    # Align frame count across cameras
    T = min(kps_by_cam[cam].shape[0] for cam in cams)
    # Check joint count consistency
    J = kps_by_cam[base_cam].shape[1]
    if J < J3d:
        raise ValueError(f"Expected at least {J3d} joints, got J={J}")
    J_eff = J3d

    # Build lists for triangulate_points:
    # world frame = base_cam frame
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
            # good convention
            R = extr_R[cam].T
            t = np.asarray(-R @ extr_T[cam]).reshape(3)

        Rt_by_cam[cam] = (R, t)
        mtxs.append(K)
        dists.append(D)
        P = np.hstack([R, t.reshape(3, 1)])  # [3x4]
        projections.append(P)

    total_err = 0.0
    total_count = 0

    for t_idx in range(T):
        # Build the list of 2D keypoints for this frame for each camera
        keypoints_list = []
        # Check confidences: we only use joints where ALL cams are confident
        conf_all = np.ones((J_eff,), dtype=bool)

        for cam in cams:
            kps_frame = kps_by_cam[cam][t_idx, :J_eff, :]   # (J_eff, 2)
            keypoints_list.append(kps_frame)
            conf_frame = conf_by_cam[cam][t_idx, :J_eff]   # (J_eff,)
            conf_all &= (conf_frame > conf_thresh)

        if not np.any(conf_all):
            # No joint seen confidently by all cameras at this frame
            continue

        # Triangulate 3D joints in base_cam frame using ALL cameras
        p3d_frame = triangulate_points(
            keypoints_list,
            mtxs,
            dists,
            projections,
        )  # (J_eff, 3) in base_cam frame

        # Reproject into each camera and accumulate error
        for j in range(J_eff):
            if not conf_all[j]:
                continue

            X = p3d_frame[j]  # (3,)

            for cam in cams:
                R_cam, t_cam = Rt_by_cam[cam]
                K_cam = K_by_cam[cam]

                # Point in camera coordinates
                Xc = R_cam @ X + t_cam
                if Xc[2] <= 1e-6:
                    continue  # behind camera or invalid depth

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

# -------------------------------------------------------------------
#  High-level: extrinsics calibration for last session
# -------------------------------------------------------------------


def calibrate_from_last_session(
    recorded_sessions,
    config_dir: str,
    conf_thresh: float = 0.8,
):
    """
    Take the *last* recorded session (multi-cam), and:

      1. Extract RTMPose keypoints for all cameras.
      2. For each camera != base_cam:
         - estimate extrinsics base->cam from RTMPose (E + recoverPose).
         - load checkerboard extrinsics base->cam if available.
      3. Compute global multi-camera RMSE using:
         - RTMPose extrinsics set
         - checkerboard extrinsics set (if complete)
      4. Save RTMPose extrinsics per pair:
         config/cam_params/c<base>_to_c<other>_params_color_rtmpose.yaml
    """
    if not recorded_sessions:
        print("No recorded sessions, nothing to calibrate.")
        return

    last_session = recorded_sessions[-1]  # dict[cam_idx -> video_path]
    cam_indices = sorted(last_session.keys())
    if len(cam_indices) < 2:
        print("Need at least 2 cameras in session to calibrate extrinsics.")
        return

    base_cam = cam_indices[0]
    print(f"\nUsing last session, base camera = {base_cam}")
    print("Session videos:")
    for cam_idx, path in last_session.items():
        print(f"  cam {cam_idx}: {path}")

    # 1) Load intrinsics
    K_by_cam = {}
    D_by_cam = {}
    cam_params_dir = os.path.join(config_dir, "cam_params")

    for cam_idx in cam_indices:
        intr_path = os.path.join(cam_params_dir, f"c{cam_idx}_params_color.yaml")
        if not os.path.exists(intr_path):
            raise FileNotFoundError(
                f"Missing intrinsics for cam {cam_idx}: {intr_path}"
            )
        Ki, Di = load_cam_params(intr_path)
        K_by_cam[cam_idx] = Ki
        D_by_cam[cam_idx] = Di

    # 2) Extract RTMPose keypoints for all cameras
    kps_by_cam = {}
    conf_by_cam = {}

    print("\nExtracting RTMPose keypoints for all cameras...")
    for cam_idx in cam_indices:
        video_path = last_session[cam_idx]
        print(f"  cam {cam_idx}: {video_path}")
        kps, conf = extract_keypoints_from_video(
            video_path,
            det_model_path=settings.det_model_path,
            pose_model_path=settings.pose_model_path,
        )
        kps_by_cam[cam_idx] = kps
        conf_by_cam[cam_idx] = conf

    # 3) Align all cameras to same number of frames (truncate to minimum T)
    T_min = min(arr.shape[0] for arr in kps_by_cam.values())
    for cam_idx in cam_indices:
        kps_by_cam[cam_idx] = kps_by_cam[cam_idx][:T_min]
        conf_by_cam[cam_idx] = conf_by_cam[cam_idx][:T_min]

    # 4) Build extrinsic sets for RTMPose and checkerboard
    extr_R_rtmpose = {}
    extr_T_rtmpose = {}
    extr_R_cb = {}
    extr_T_cb = {}

    kps_base = kps_by_cam[base_cam]
    conf_base = conf_by_cam[base_cam]

    for cam_idx in cam_indices:
        if cam_idx == base_cam:
            continue

        kps_other = kps_by_cam[cam_idx]
        conf_other = conf_by_cam[cam_idx]

        print(f"\nEstimating extrinsics base={base_cam} -> cam {cam_idx} from RTMPose...")
        R_rt, t_rt, rmse_pair_raw = estimate_extrinsics_from_keypoints(
            kps_base,
            kps_other,
            conf_base,
            conf_other,
            K_by_cam[base_cam],
            D_by_cam[base_cam],
            K_by_cam[cam_idx],
            D_by_cam[cam_idx],
            conf_thresh=conf_thresh,
        )

        # Convert RTMPose extrinsics to checkerboard convention (base->other):
        # RTMPose gives R_rt, t_rt such that its convention differs; convert via
        # R_conv = R_rt.T
        # T_conv = - R_conv @ t_rt
        R_conv = R_rt.T
        T_conv = - R_conv @ t_rt.reshape(3)

        # Store converted extrinsics for global RMSE computation
        extr_R_rtmpose[cam_idx] = R_conv
        extr_T_rtmpose[cam_idx] = T_conv

        # Recompute pairwise RMSE using converted extrinsics (checkerboard convention)
        rmse_pair = compute_reprojection_rmse_pair(
            kps_base, kps_other, conf_base, conf_other,
            K_by_cam[base_cam], D_by_cam[base_cam],
            K_by_cam[cam_idx], D_by_cam[cam_idx],
            R_conv, T_conv, conf_thresh
        )

        print(f"  Pairwise RMSE (base={base_cam}, cam={cam_idx}) with RTMPose extrinsics (converted): {rmse_pair:.3f} px")

        # Try to load checkerboard stereo for this pair
        stereo_cb_path = os.path.join(
            cam_params_dir,
            f"c{base_cam}_to_c{cam_idx}_params_color.yaml",
        )
        if os.path.exists(stereo_cb_path):
            R_cb, T_cb = load_cam_to_cam_params(stereo_cb_path)
            extr_R_cb[cam_idx] = R_cb
            extr_T_cb[cam_idx] = T_cb.reshape(3)
        else:
            print(f"  No checkerboard stereo file for pair (base={base_cam}, cam={cam_idx}), path: {stereo_cb_path}")

        # Save RTMPose stereo parameters per pair
        out_path = os.path.join(
            cam_params_dir,
            f"c{base_cam}_to_c{cam_idx}_params_color_rtmpose.yaml",
        )
        save_cam_to_cam_params(
            K_by_cam[base_cam],
            D_by_cam[base_cam],
            K_by_cam[cam_idx],
            D_by_cam[cam_idx],
            R_conv,
            T_conv.reshape(3, 1),
            rmse_pair,
            out_path,
        )
        print(f"  Saved RTMPose extrinsics to: {out_path}")

    # 5) Global multi-camera RMSE for RTMPose extrinsics
    print("\n=== Global multi-camera RMSE with RTMPose extrinsics ===")
    rmse_global_rtmpose = compute_reprojection_rmse_multicam(
        kps_by_cam,
        conf_by_cam,
        K_by_cam,
        D_by_cam,
        base_cam=base_cam,
        extr_R=extr_R_rtmpose,
        extr_T=extr_T_rtmpose,
        conf_thresh=conf_thresh,
        J3d=26,
    )
    print(f"Global RMSE (all cameras, RTMPose extrinsics): {rmse_global_rtmpose:.3f} px")

    # 6) Global multi-camera RMSE for checkerboard extrinsics (if available)
    if len(extr_R_cb) == (len(cam_indices) - 1):
        print("\n=== Global multi-camera RMSE with checkerboard extrinsics ===")
        rmse_global_cb = compute_reprojection_rmse_multicam(
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
        print(f"Global RMSE (all cameras, checkerboard extrinsics): {rmse_global_cb:.3f} px")
    else:
        print("\nCheckerboard extrinsics not available for all cameras, skipping global CB RMSE.")


# -------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------


def main():
    # In your repo, config is usually under <repo_root>/config
    config_dir = os.path.join(repo_path, "config")

    # 1) Record videos for calibration (multi-cam, robust)
    # recorded_sessions = record_calibration_videos(config_dir)
    recorded_sessions = [{"0": "/root/workspace/ros_ws/src/rt-cosmik/tests/full/data/camera_0.mp4", "2": "/root/workspace/ros_ws/src/rt-cosmik/tests/full/data/camera_2.mp4"}]  # <-- For testing, skip recording step

    # 2) Run extrinsics calibration for the last session using RTMPose
    #    (You can pass base_cam, conf_thresh, max_frames as you like)
    calibrate_from_last_session(
        recorded_sessions,
        config_dir=config_dir
    )


if __name__ == "__main__":
    main()
