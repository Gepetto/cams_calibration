# To run the code from repo root : python3 scripts/rgb/calibrate_cameras.py

import os
# Get the absolute path to the current file (script_to_launch.py)
script_path = os.path.abspath(__file__)
# Go up two directories: from 'rgb' to 'scripts', then from 'scripts' to 'repo'
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # Repo root

from utils.settings import Settings
# FIRST, PARAM LOADING
settings = Settings()

import numpy as np
import cv2
from utils.calib_utils import calibrate_camera, save_cam_params, load_cam_params, stereo_calibrate, save_cam_to_cam_params, list_cameras_with_v4l2

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
os.makedirs(os.path.join(repo_path,"images_calib_cam_1","color"), exist_ok=True)
os.makedirs(os.path.join(repo_path,"images_calib_cam_2","color"), exist_ok=True)

# Define paths 
c1_color_imgs_dir = os.path.join(repo_path, "images_calib_cam_1", "color")
c2_color_imgs_dir = os.path.join(repo_path, "images_calib_cam_2", "color")
c1_color_params_path = os.path.join(repo_path, "config","cam_params","c1_params_color.yaml")
c2_color_params_path = os.path.join(repo_path, "config","cam_params","c2_params_color.yaml")
c1_to_c2_color_params_path = os.path.join(repo_path, "config","cam_params","c1_to_c2_params_color.yaml")

img_idx = 0
try:
    while True:
        frames = [cap.read()[1] for cap in captures]
            
        if not all(frame is not None for frame in frames):
            continue

        # Convert images to numpy arrays
        color_image_1 = frames[0]
        color_image_2 = frames[1]

        resized_color_image_1 = cv2.resize(color_image_1, (640, 480), interpolation = cv2.INTER_NEAREST)
        resized_color_image_2 = cv2.resize(color_image_2, (640, 480), interpolation = cv2.INTER_NEAREST) 

        images_hstack_1 = np.hstack((resized_color_image_1, resized_color_image_2))
        
        # Show images
        cv2.namedWindow('RGB cams', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB cams', images_hstack_1)
        c = cv2.waitKey(10)
        if c == ord('s'):
            print("Images taken")
            # Build the full file paths for the images
            img1_path = os.path.join(c1_color_imgs_dir, "img_" + str(img_idx) + ".png")
            img2_path = os.path.join(c2_color_imgs_dir, "img_" + str(img_idx) + ".png")
            cv2.imwrite(img1_path, color_image_1)
            cv2.imwrite(img2_path, color_image_2)
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
reproj1_color, mtx1_color, dist1_color = calibrate_camera(images_folder=os.path.join(c1_color_imgs_dir, "*.png"))
save_cam_params(mtx1_color, dist1_color, reproj1_color, c1_color_params_path)
reproj2_color, mtx2_color, dist2_color = calibrate_camera(images_folder=os.path.join(c2_color_imgs_dir, "*.png"))
save_cam_params(mtx2_color, dist2_color, reproj2_color, c2_color_params_path)
print("rmse cam1 RGB = ", reproj1_color)
print("rmse cam2 RGB = ", reproj2_color)
cv2.destroyAllWindows()

# Load camera parameters and perform stereo calibration
mtx_1_color, dist_1_color = load_cam_params(c1_color_params_path)
mtx_2_color, dist_2_color = load_cam_params(c2_color_params_path)
rmse_color, R_color, T_color = stereo_calibrate(mtx_1_color, dist_1_color,
                                                mtx2_color, dist2_color,
                                                os.path.join(c1_color_imgs_dir, "*.png"), os.path.join(c2_color_imgs_dir, "*.png"))
save_cam_to_cam_params(mtx_1_color, dist_1_color, mtx2_color, dist2_color,
                       R_color, T_color, rmse_color, c1_to_c2_color_params_path)
cv2.destroyAllWindows()

print("Computed translation RGB:")
print(T_color)
print("rmse cam to cam: ", rmse_color)