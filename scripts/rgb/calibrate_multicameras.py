import os
import sys
import numpy as np
import cv2

# Adjust paths
script_path = os.path.abspath(__file__)
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(repo_path))  # Repo root
print(sys.path)

from utils.settings import Settings
from utils.calib_utils import calibrate_camera, save_cam_params, load_cam_params, stereo_calibrate, save_cam_to_cam_params, list_cameras_with_v4l2, save_global_cam_params, load_global_cam_params

# Load settings
settings = Settings()
config_path= "/root/workspace/ros_ws/src/rt-cosmik/config"


nbr_cam = 4   # Set number of cameras
pairs = [(0, 1), (1, 2), (2, 3)]

## Initialize camera streams
camera_dict = list_cameras_with_v4l2()
captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_dict.keys()]

# Check if cameras opened correctly
for idx, cap in enumerate(captures):
    if not cap.isOpened():
        print(f"Warning: Camera {idx} not opened.")

    # Apply settings
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

# Create image directories for all cameras
cam_dirs = []
for i in range(nbr_cam):
    dir_path = os.path.join(config_path, f"images_calib_cam_{i+1}", "color")
    os.makedirs(dir_path, exist_ok=True)
    cam_dirs.append(dir_path)

# Define paths for camera parameters
calib_params_paths = [
    os.path.join(config_path, "config", "cam_params", f"c{i+1}_params_color.yaml") for i in range(nbr_cam)
]

# Define paths for stereo calibration between camera pairs
stereo_params_paths = {}
for (i, j) in pairs:
    stereo_params_paths[(i, j)] = os.path.join(
        config_path, "config", "cam_params", f"c{i+1}_to_c{j+1}_params_color.yaml")

print(stereo_params_paths)
# Image capturing loop
img_idx = 0
try:
    while True:
        frames = [cap.read()[1] for cap in captures]

        if not all(frame is not None for frame in frames):
            print("Warning: Missing frame(s), retrying...")
            continue

        # Resize frames for display
        resized_frames = [cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST) for frame in frames]

        # Stack images for visualization (2x2 grid)
        images_hstack_1 = np.hstack((resized_frames[0], resized_frames[1]))
        images_hstack_2 = np.hstack((resized_frames[2], resized_frames[3]))
        images_vstack = np.vstack((images_hstack_1, images_hstack_2))

        # Display images
        cv2.namedWindow('RGB cams', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB cams', images_vstack)
        c = cv2.waitKey(10)

        if c == ord('s'):
            print("Images taken")
            # Save images from all cameras
            for i in range(nbr_cam):
                img_path = os.path.join(cam_dirs[i], f"img_{img_idx}.png")
                cv2.imwrite(img_path, frames[i])
            img_idx += 1

        if c == ord('q'):
            print("Quit")
            break

finally:
    # Release cameras
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()

# Intrinsic calibration for each camera
for i in range(nbr_cam):
    reproj_error, mtx, dist = calibrate_camera(images_folder=os.path.join(cam_dirs[i], "*.png"))
    save_cam_params(mtx, dist, reproj_error, calib_params_paths[i])
    print(f"rmse cam {i+1} RGB = ", reproj_error)

cv2.destroyAllWindows()

# Stereo calibration for all camera pairs
extrinsics = {}
for (i, j), path in stereo_params_paths.items():
    mtx_1, dist_1 = load_cam_params(calib_params_paths[i])
    mtx_2, dist_2 = load_cam_params(calib_params_paths[j])

    rmse, R, T = stereo_calibrate(mtx_1, dist_1, mtx_2, dist_2,
                                  os.path.join(cam_dirs[i], "*.png"),
                                  os.path.join(cam_dirs[j], "*.png"))

    # save_cam_to_cam_params(mtx_1, dist_1, mtx_2, dist_2, R, T, rmse, path)
    extrinsics[(i, j)] = (R, T)
    print(f"Translation from cam {i+1} to cam {j+1}: {T}")
    print(f"rmse cam {i+1} to cam {j+1}: {rmse}")

cv2.destroyAllWindows()

#compute global poses (ref frame is cam0)
if (0, 1) in extrinsics and (1, 2) in extrinsics and (2, 3) in extrinsics:
    # Camera 0 (global reference)
    R0_0 = np.eye(3)
    T0_0 = np.zeros((3, 1))
    
    # Camera 1 pose in camera 0 frame:
    R01, T01 = extrinsics[(0, 1)]
    R0_1 = R01
    T0_1 = T01

    # Camera 2 pose in camera 0 frame:
    R12, T12 = extrinsics[(1, 2)]
    R0_2 = R0_1.dot(R12)
    T0_2 = T0_1 + R0_1.dot(T12)

    # Camera 3 pose in camera 0 frame:
    R23, T23 = extrinsics[(2, 3)]
    R0_3 = R0_2.dot(R23)
    T0_3 = T0_2 + R0_2.dot(T23)

    print("Global poses with camera 0 as reference:")
    print("\nCamera 0 (reference) pose:")
    print("Rotation:\n", R0_0)
    print("Translation:\n", T0_0)
    print("\nCamera 1 pose in camera 0 frame:")
    print("Rotation:\n", R0_1)
    print("Translation:\n", T0_1)
    print("\nCamera 2 pose in camera 0 frame:")
    print("Rotation:\n", R0_2)
    print("Translation:\n", T0_2)
    print("\nCamera 3 pose in camera 0 frame:")
    print("Rotation:\n", R0_3)
    print("Translation:\n", T0_3)
else:
    print("Not all required extrinsics for chaining (0,1), (1,2), (2,3) are available.")

global_poses_path = os.path.join(config_path, "config", "cam_params", "global_poses.yaml")
global_params = {
    0: (R0_0, T0_0),
    1: (R0_1, T0_1),
    2: (R0_2, T0_2),
    3: (R0_3, T0_3)
}
save_global_cam_params(global_params, global_poses_path)
for i in range (nbr_cam):

    R, T = load_global_cam_params(global_poses_path, i)
    print("Camera Global Rotation:\n", R)
    print("Camera Global Translation:\n", T)