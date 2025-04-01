import os
# Get the absolute path to the current file (script_to_launch.py)
script_path = os.path.abspath(__file__)
# Go up two directories: from 'rgb' to 'scripts', then from 'scripts' to 'repo'
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # Repo root

import numpy as np
import cv2
from utils.settings import Settings
from utils.calib_utils import (calibrate_camera, save_cam_params, load_cam_params, 
                               stereo_calibrate, save_cam_to_cam_params, list_cameras_with_v4l2, 
                               calibrate_camera_runtime)

# Get repo root path
script_path = os.path.abspath(__file__)
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, repo_path)

# Load settings
settings = Settings()

# Initialize camera streams
camera_dict = list_cameras_with_v4l2()
captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_dict.keys()]

for idx, cap in enumerate(captures):
    if not cap.isOpened():
        continue
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

# Create directories for saved images
c1_color_imgs_dir = os.path.join(repo_path, "images_calib_cam_1", "color")
c2_color_imgs_dir = os.path.join(repo_path, "images_calib_cam_2", "color")
os.makedirs(c1_color_imgs_dir, exist_ok=True)
os.makedirs(c2_color_imgs_dir, exist_ok=True)

# Define paths for calibration parameters
c1_color_params_path = os.path.join(repo_path, "config", "cam_params", "c1_params_color.yaml")
c2_color_params_path = os.path.join(repo_path, "config", "cam_params", "c2_params_color.yaml")
c1_to_c2_color_params_path = os.path.join(repo_path, "config", "cam_params", "c1_to_c2_params_color.yaml")

# Capture images for calibration
img_idx = 0
runtime_images = []
try:
    while True:
        frames = [cap.read()[1] for cap in captures]
        
        if not all(frame is not None for frame in frames):
            continue

        color_image_1 = frames[1]  # Assuming first camera
        resized_color_image_1 = cv2.resize(color_image_1, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        
        # Display image
        cv2.imshow('RGB cams', resized_color_image_1)
        c = cv2.waitKey(10)
        
        if c == ord('s'):
            print("Images taken")
            img1_path = os.path.join(c1_color_imgs_dir, f"img_{img_idx}.png")
            cv2.imwrite(img1_path, color_image_1)
            runtime_images.append(color_image_1)  # Store in memory for runtime calibration
            img_idx += 1
        
        if c == ord('q'):
            print("Quit")
            break

finally:
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()

# Calibrate using saved PNG images
reproj1_color, mtx1_color, dist1_color = calibrate_camera(images_folder=os.path.join(c1_color_imgs_dir, "*.png"))
save_cam_params(mtx1_color, dist1_color, reproj1_color, c1_color_params_path)
print("rmse cam1 RGB =", reproj1_color)

# Runtime calibration using images in memory
if runtime_images:
    reproj_runtime, mtx_runtime, dist_runtime = calibrate_camera_runtime(runtime_images)
    print("rmse cam1 runtime RGB =", reproj_runtime)
