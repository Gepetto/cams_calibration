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
    PoseTrackerEstimator,
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


def build_correspondences(kps1, kps2, conf1, conf2, conf_thresh=0.5):
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
    conf_thresh=0.5,
):
    """
    Estimate (R, T) between cam2 and cam1 from RTMPose 2D keypoints.

    Pipeline:
        - build 2D-2D correspondences across all frames/joints
        - undistort to normalized coordinates
        - estimate essential matrix with RANSAC
        - recover pose (R, t)
        - compute reprojection RMSE (for info)
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

    # Compute reprojection RMSE over all valid measurements (not only inliers)
    rmse = compute_reprojection_rmse(kps1, kps2, conf1, conf2, K1, D1, K2, D2, R, t)
    return R, t, rmse


def compute_reprojection_rmse(kps1, kps2, conf1, conf2, K1, D1, K2, D2, R, t, conf_thresh=0.5):
    """
    Use current (R, t) to triangulate each joint and reproject back
    to both cameras; compute RMS error in pixels.
    """
    T, J, _ = kps1.shape
    total_err = 0.0
    total_count = 0

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])   # cam1 at origin
    P2 = np.hstack([R, t.reshape(3, 1)])            # cam2 in cam1 frame

    for t_idx in range(T):
        for j in range(J):
            if conf1[t_idx, j] <= conf_thresh or conf2[t_idx, j] <= conf_thresh:
                continue

            p1_pix = kps1[t_idx, j]
            p2_pix = kps2[t_idx, j]

            # undistort -> normalized coords
            p1n = cv2.undistortPoints(
                p1_pix.reshape(1, 1, 2), K1, D1
            ).reshape(2)
            p2n = cv2.undistortPoints(
                p2_pix.reshape(1, 1, 2), K2, D2
            ).reshape(2)

            X_h = cv2.triangulatePoints(
                P1, P2, p1n.reshape(2, 1), p2n.reshape(2, 1)
            )
            X = (X_h[:3] / X_h[3])[:, 0]  # (3,)

            # Reproject to cam1
            Xc1 = X
            p1n_hat = Xc1[:2] / Xc1[2]
            p1_pix_hat = (K1 @ np.array([p1n_hat[0], p1n_hat[1], 1.0]))[:2]

            # Reproject to cam2
            Xc2 = R @ X + t.reshape(3)
            p2n_hat = Xc2[:2] / Xc2[2]
            p2_pix_hat = (K2 @ np.array([p2n_hat[0], p2n_hat[1], 1.0]))[:2]

            err1 = np.sum((p1_pix_hat - p1_pix) ** 2)
            err2 = np.sum((p2_pix_hat - p2_pix) ** 2)

            total_err += err1 + err2
            total_count += 2

    if total_count == 0:
        return np.nan
    return float(np.sqrt(total_err / total_count))


# -------------------------------------------------------------------
#  High-level: extrinsics calibration for last session
# -------------------------------------------------------------------


def calibrate_from_last_session(
    recorded_sessions,
    config_dir: str,
):
    """
    Take the *last* recorded session, and for each camera (except base_cam),
    estimate extrinsics w.r.t. base_cam from RTMPose keypoints.

    Saves YAML files like:
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

    base_cam = cam_indices[0]  # smallest index
    if base_cam not in cam_indices:
        raise ValueError(f"Base camera {base_cam} not present in last session.")

    print(f"\nCalibrating extrinsics using last session, base camera = {base_cam}")
    print("Session videos:")
    for cam_idx, path in last_session.items():
        print(f"  cam {cam_idx}: {path}")

    # Load intrinsics for all cameras
    K = {}
    D = {}
    cam_params_dir = os.path.join(config_dir, "cam_params")
    for cam_idx in cam_indices:
        intr_path = os.path.join(cam_params_dir, f"c{cam_idx}_params_color.yaml")
        if not os.path.exists(intr_path):
            raise FileNotFoundError(
                f"Missing intrinsics for cam {cam_idx}: {intr_path}"
            )
        Ki, Di = load_cam_params(intr_path)
        K[cam_idx] = Ki
        D[cam_idx] = Di

    # Extract keypoints for base camera once
    base_video = last_session[base_cam]
    print(f"\nExtracting RTMPose keypoints for base camera {base_cam}...")
    kps_base, conf_base = extract_keypoints_from_video(
        base_video,
        det_model_path=settings.det_model_path,
        pose_model_path=settings.pose_model_path,
    )
    T_base, J_base, _ = kps_base.shape

    for cam_idx in cam_indices:
        if cam_idx == base_cam:
            continue

        other_video = last_session[cam_idx]
        print(f"\nExtracting RTMPose keypoints for camera {cam_idx}...")
        kps_other, conf_other = extract_keypoints_from_video(
            other_video,
            det_model_path=settings.det_model_path,
            pose_model_path=settings.pose_model_path,
        )

        # Temporal alignment: truncate to min length
        T_other = kps_other.shape[0]
        T = min(T_base, T_other)
        kps1 = kps_base[:T]
        kps2 = kps_other[:T]
        c1 = conf_base[:T]
        c2 = conf_other[:T]

        # Check joint count consistency
        if kps1.shape[1] != kps2.shape[1]:
            raise RuntimeError(
                f"Joint count mismatch between base_cam ({kps1.shape[1]}) "
                f"and cam {cam_idx} ({kps2.shape[1]})"
            )

        print(f"Estimating extrinsics between cam {base_cam} and cam {cam_idx}...")
        R, t, rmse = estimate_extrinsics_from_keypoints(
            kps1, kps2, c1, c2, K[base_cam], D[base_cam], K[cam_idx], D[cam_idx],
            conf_thresh=0.5,
        )

        print(f"  -> RMSE reprojection error: {rmse:.3f} pixels")
        # Save stereo parameters
        out_path = os.path.join(
            cam_params_dir,
            f"c{base_cam}_to_c{cam_idx}_params_color_rtmpose.yaml",
        )
        save_cam_to_cam_params(
            K[base_cam],
            D[base_cam],
            K[cam_idx],
            D[cam_idx],
            R,
            t.reshape(3, 1),
            rmse,
            out_path,
        )
        print(f"  Saved extrinsics to: {out_path}")


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
