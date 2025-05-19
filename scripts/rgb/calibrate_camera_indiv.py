# To run the code from repo root : python3 scripts/rgb/calibrate_cameras.py

import os
# Get the absolute path to the current file (script_to_launch.py)
script_path = os.path.abspath(__file__)
# Go up two directories: from 'rgb' to 'scripts', then from 'scripts' to 'repo'
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # Repo root

num_cam = sys.argv[1]

from utils.settings import Settings
# FIRST, PARAM LOADING
settings = Settings()

import numpy as np
import cv2
from utils.calib_utils import calibrate_camera, save_cam_params, load_cam_params, stereo_calibrate, save_cam_to_cam_params, list_cameras_with_v4l2

config_path= "/root/workspace/ros_ws/src/rt-cosmik/config"
## Initialize cams stream
camera_dict = list_cameras_with_v4l2()
captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_dict.keys()]

for idx, cap in enumerate(captures):
    if not cap.isOpened():
        continue

    # Apply settings
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

# Use os.makedirs() to create your directory; exist_ok=True means it won't throw an error if the directory already exists
os.makedirs(os.path.join(config_path,f"images_calib_cam_{num_cam}","color"), exist_ok=True)

# Define paths 
color_imgs_dir = os.path.join(config_path, f"images_calib_cam_{num_cam}", "color")
color_params_path = os.path.join(config_path,"cam_params",f"c{num_cam}_params_color.yaml")

img_idx = 0
print("In one sec you can start typing s to save images for calibration and then press q to quit")
try:
    while True:
        frames = [cap.read()[1] for cap in captures]
            
        if not all(frame is not None for frame in frames):
            continue

        # Convert images to numpy arrays
        color_image = frames[int(num_cam)-1]

        resized_color_image = cv2.resize(color_image, (640, 480), interpolation = cv2.INTER_NEAREST)
        
        # Show images
        cv2.namedWindow('RGB cams', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB cams', resized_color_image)
        c = cv2.waitKey(10)
        if c == ord('s'):
            print("Images taken")
            # Build the full file paths for the images
            img_path = os.path.join(color_imgs_dir, "img_" + str(img_idx) + ".png")
            cv2.imwrite(img_path, color_image)
            img_idx += 1
        if c == ord('q'):
            print("quit")
            break

finally:
    # Release the camera captures
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()

# Calibrate each camera using the saved images.
reproj_color, mtx_color, dist_color = calibrate_camera(images_folder=os.path.join(color_imgs_dir, "*.png"))
save_cam_params(mtx_color, dist_color, reproj_color, color_params_path)
print(f"rmse cam{num_cam} RGB = ", reproj_color)
cv2.destroyAllWindows()