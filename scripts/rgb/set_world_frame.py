# To run the code from repo root : python3 scripts/rgb/set_world_frame.py
# Add --offline to skip live camera capture and use existing images instead.
# Add --soder to compute world-frame extrinsics from soder{N}.txt (mocap-based)
# instead of the ArUco-wand method.
# Add --num-cameras {2,4} to select a 2-camera (c0,c2) or 4-camera (c0,c2,c4,c6) rig.

import os
import sys
import argparse

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.settings import Settings
settings = Settings()

import cv2
import numpy as np
from utils.calib_utils import (
    load_cam_params,
    save_pose_to_yaml,
    get_aruco_pose,
    get_relative_pose_world_in_cam,
    list_cameras_with_v4l2,
    load_transformation,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                         help="Skip live camera capture and use images already saved on disk.")
    parser.add_argument("--soder", action="store_true",
                         help="Compute world-frame extrinsics from soder{N}.txt "
                              "(mocap-based) instead of the ArUco-wand method. Skips "
                              "capture and ArUco detection entirely.")
    parser.add_argument("--num-cameras", type=int, choices=[2, 4], default=2,
                         help="Number of physical cameras: 2 (c0, c2) or 4 (c0, c2, c4, c6).")
    return parser.parse_args()


def camera_ids_for(num_cameras):
    return [i * 2 for i in range(num_cameras)]


def images_dir_for(repo_path, cam_id):
    # Preserves the original folder naming (images_world_cam_1 for cam 0,
    # images_world_cam_2 for cam 2) for cam0/cam2, extends numerically for cam4/cam6.
    legacy_names = {0: "images_world_cam_1", 2: "images_world_cam_2"}
    name = legacy_names.get(cam_id, f"images_world_cam_{cam_id}")
    return os.path.join(repo_path, name, "color")


def extrinsics_path(cam_params_dir, cam_id):
    return os.path.join(cam_params_dir, f"camera_{cam_id}_extrinsics.yaml")


def intrinsics_path(cam_params_dir, cam_id):
    return os.path.join(cam_params_dir, f"camera_{cam_id}_intrinsics.yaml")


def run_soder_world_frame(cam_ids, cam_params_dir):
    for cam_id in cam_ids:
        soder_path = os.path.join(cam_params_dir, f"soder{cam_id}.txt")
        if not os.path.isfile(soder_path):
            raise FileNotFoundError(f"soder{cam_id}.txt not found at {soder_path}")

        R, d, s, rms = load_transformation(soder_path)
        out_path = extrinsics_path(cam_params_dir, cam_id)

        save_pose_to_yaml(
            R, d, out_path,
            frame_from=f"camera_{cam_id}", frame_to="world",
            scale_factor=s, rms_error=rms, source_file=f"soder{cam_id}.txt",
        )
        print(f"[SODER] Saved {out_path}  (rms={rms:.6f}, scale={s:.6f})")


def run_aruco_world_frame(cam_ids, cam_params_dir, imgs_dirs, offline):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = settings.wand_marker_size
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    wand_local = settings.wand_end_effector_local_pos

    K = {}
    D = {}
    for cam_id in cam_ids:
        K[cam_id], D[cam_id] = load_cam_params(intrinsics_path(cam_params_dir, cam_id))

    img_idx = 0

    if not offline:
        camera_dict = list_cameras_with_v4l2()
        device_indices = list(camera_dict.keys())[:len(cam_ids)]
        captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in device_indices]

        for cap in captures:
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
            cap.set(cv2.CAP_PROP_FPS, settings.fs)

        window_names = [f'Camera {cam_id} Pose Estimation' for cam_id in cam_ids]
        for name in window_names:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)

        try:
            while True:
                frames = [cap.read()[1] for cap in captures]

                if not all(frame is not None for frame in frames):
                    continue

                display_frames = []
                for cam_id, frame, name in zip(cam_ids, frames, window_names):
                    f = np.asanyarray(frame.copy())
                    tm, corners, rvec, tvec = get_aruco_pose(f, K[cam_id], D[cam_id], detector, marker_size)
                    if tm is not None:
                        tip_pos = tvec + tm[:3, :3] @ wand_local
                        image_points, _ = cv2.projectPoints(tip_pos, np.zeros(3,), np.zeros(3,), K[cam_id], D[cam_id])
                        image_points = image_points[0][0]
                        cv2.aruco.drawDetectedMarkers(f, [corners])
                        cv2.drawFrameAxes(f, K[cam_id], D[cam_id], rvec, tvec, 0.1)
                        f = cv2.circle(f, (int(image_points[0]), int(image_points[1])), 5, (0, 0, 255), -1)
                    display_frames.append(f)
                    cv2.imshow(name, f)

                c = cv2.waitKey(10)
                if c == ord('s'):
                    print("Images taken")
                    for cam_id, frame in zip(cam_ids, frames):
                        cv2.imwrite(os.path.join(images_dir_for(repo_path, cam_id), f"img_{img_idx}.png"), frame)
                    img_idx += 1
                if c == ord('q'):
                    print("quit")
                    break

                if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in window_names):
                    print("window closed, moving on to extrinsics computation")
                    break
        finally:
            for cap in captures:
                cap.release()
            cv2.destroyAllWindows()
    else:
        print(f"[OFFLINE] Skipping live capture, using existing images.")

    cam_T = {}
    cam_R = {}
    for cam_id in cam_ids:
        img_dir = images_dir_for(repo_path, cam_id)
        cam_T[cam_id], cam_R[cam_id] = get_relative_pose_world_in_cam(
            os.path.join(img_dir, "*.png"), K[cam_id], D[cam_id], detector, marker_size
        )
        out_path = extrinsics_path(cam_params_dir, cam_id)
        save_pose_to_yaml(cam_R[cam_id], cam_T[cam_id], out_path, f"camera_{cam_id}")
        print(f"Saved {out_path}")

    # Optional visual sanity check
    cv2.namedWindow('Reprojected Image', cv2.WINDOW_NORMAL)
    for cam_id in cam_ids:
        img_dir = images_dir_for(repo_path, cam_id)
        image = cv2.imread(os.path.join(img_dir, "img_0.png"))
        if image is None:
            print(f"Skipping reprojection preview for camera_{cam_id}: image not found on disk.")
            continue

        cam_rodrigues = cv2.Rodrigues(cam_R[cam_id])[0]
        cv2.drawFrameAxes(image, K[cam_id], D[cam_id], cam_rodrigues, cam_T[cam_id], 0.1)
        cv2.imshow('Reprojected Image', image)

        while True:
            key = cv2.waitKey(30)
            if key == ord('q') or key == ord('a'):
                break
            if cv2.getWindowProperty('Reprojected Image', cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()


def main():
    args = parse_args()

    cam_params_dir = os.path.join(repo_path, "config", "cam_params")
    os.makedirs(cam_params_dir, exist_ok=True)

    cam_ids = camera_ids_for(args.num_cameras)
    for cam_id in cam_ids:
        os.makedirs(images_dir_for(repo_path, cam_id), exist_ok=True)

    if args.soder:
        run_soder_world_frame(cam_ids, cam_params_dir)
        return

    imgs_dirs = [images_dir_for(repo_path, cam_id) for cam_id in cam_ids]
    run_aruco_world_frame(cam_ids, cam_params_dir, imgs_dirs, args.offline)


if __name__ == "__main__":
    main()