# To run the code from repo root : python3 scripts/rgb/set_world_frame.py

import os
# Get the absolute path to the current file (script_to_launch.py)
script_path = os.path.abspath(__file__)
# Go up two directories: from 'rgb' to 'scripts', then from 'scripts' to 'repo'
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_path = "/home/ngouget/Codes/rt-cosmik/config"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # Repo root

from utils.settings import Settings
# FIRST, PARAM LOADING
settings = Settings()

import cv2
import numpy as np
from utils.calib_utils import load_cam_params, save_pose_matrix_to_yaml, get_aruco_pose, get_relative_pose_world_in_cam, list_cameras_with_v4l2

### Initialize cams stream
cameras = list_cameras_with_v4l2()

for idx_cam in cameras.keys():
    cap = cv2.VideoCapture(int(idx_cam), cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Could not open camera {idx_cam}")
        exit()

    # Apply settings
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

    # Use os.makedirs() to create your directory; exist_ok=True means it won't throw an error if the directory already exists
    os.makedirs(os.path.join(config_path,f"images_world_cam_{idx_cam}","color"), exist_ok=True)

    os.makedirs(os.path.join(config_path,"cam_params"), exist_ok=True)

    globals()[f"c{idx_cam}_color_imgs_dir"] = os.path.join(config_path, f"images_world_cam_{idx_cam}", "color")

    globals()[f"c{idx_cam}_color_params_path"] = os.path.join(config_path,"cam_params",f"camera{idx_cam}_pose.yaml")

    globals()[f"K{idx_cam}"], globals()[f"D{idx_cam}"] = load_cam_params(os.path.join(config_path,"cam_params",f"c{idx_cam}_params_color.yaml"))

    # Camera intrinsic parameters (from your YAML file)
    globals()[f"camera_matrix_{idx_cam}"] = globals()[f"K{idx_cam}"]

    # Distortion coefficients (from your YAML file)
    globals()[f"dist_coeffs_{idx_cam}"] = globals()[f"D{idx_cam}"]

# Define the ArUco dictionary and marker size
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_size = settings.wand_marker_size  # Marker size in meters (17.6 cm)

# Initialize the ArUco detection parameters
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

wand_local = settings.wand_end_effector_local_pos

captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in cameras.keys()]

img_idx=0
print("In one sec you can start typing s after pointing origin, then axis x then axis y for setting world frame and then press q to quit")
try : 
    while True:
        frames = [cap.read()[1] for cap in captures]
            
        if not all(frame is not None for frame in frames):
            continue

        for ind, idx_cam in enumerate(cameras.keys()):
            globals()[f"color_frame_{idx_cam}"] = frames[ind]

            # Convert images to numpy arrays
            globals()[f"frame_{idx_cam}"] = np.asanyarray(globals()[f"color_frame_{idx_cam}"].copy())

            # Get the camera pose relative to the global frame defined by the ArUco marker
            globals()[f"transformation_matrix_{idx_cam}"], globals()[f"corners_{idx_cam}"], globals()[f"rvec_{idx_cam}"], globals()[f"tvec_{idx_cam}"] = get_aruco_pose(globals()[f"frame_{idx_cam}"], globals()[f"K{idx_cam}"], globals()[f"D{idx_cam}"], detector, marker_size)


            if globals()[f"transformation_matrix_{idx_cam}"] is not None:
                globals()[f"tip_pos{idx_cam}"]= globals()[f"tvec_{idx_cam}"] + globals()[f"transformation_matrix_{idx_cam}"][:3, :3]@wand_local 

                # Project the 3D wand tip position to 2D image coordinates
                globals()[f"image_points{idx_cam}"], _ = cv2.projectPoints(globals()[f"tip_pos{idx_cam}"], np.zeros(3,), np.zeros(3,), globals()[f"camera_matrix_{idx_cam}"], globals()[f"dist_coeffs_{idx_cam}"])
                globals()[f"image_points{idx_cam}"] = globals()[f"image_points{idx_cam}"][0][0]
            
                # Draw the marker and its pose on the frame for Camera 1
                cv2.aruco.drawDetectedMarkers(globals()[f"frame_{idx_cam}"], [globals()[f"corners_{idx_cam}"]])
                cv2.drawFrameAxes(globals()[f"frame_{idx_cam}"], globals()[f"K{idx_cam}"], globals()[f"D{idx_cam}"], globals()[f"rvec_{idx_cam}"], globals()[f"tvec_{idx_cam}"], 0.1)

                # Draw the reprojected wand tip on the image
                globals()[f"frame_{idx_cam}"] = cv2.circle(globals()[f"frame_{idx_cam}"], (int(globals()[f"image_points{idx_cam}"][0]), int(globals()[f"image_points{idx_cam}"][1])), 5, (0, 0, 255), -1)


            # Display the frames for both cameras
            cv2.imshow(f'Camera {idx_cam} Pose Estimation', globals()[f"frame_{idx_cam}"])

        c = cv2.waitKey(10)
        if c == ord('s'):
            print("Images taken")

            for idx_cam in cameras.keys():
                # Build the full file paths for the images
                globals()[f"img{idx_cam}_path"] = os.path.join(globals()[f"c{idx_cam}_color_imgs_dir"], "img_" + str(img_idx) + ".png")

                cv2.imwrite(globals()[f"img{idx_cam}_path"], globals()[f"color_frame_{idx_cam}"])

            img_idx += 1
        if c == ord('q'):
            print("quit")
            break
finally : 
    # Release the camera captures
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()

camera_data = []

for idx_cam in cameras.keys():
    globals()[f"cam_T{idx_cam}_world"], globals()[f"cam_R{idx_cam}_world"] = get_relative_pose_world_in_cam(os.path.join(globals()[f"c{idx_cam}_color_imgs_dir"], "*.png"),globals()[f"K{idx_cam}"],globals()[f"D{idx_cam}"],detector, marker_size)

    # Save the rotation matrix and translation vector to a YAML file for Camera 1
    save_pose_matrix_to_yaml(globals()[f"cam_R{idx_cam}_world"], globals()[f"cam_T{idx_cam}_world"], globals()[f"c{idx_cam}_color_params_path"])

    # Camera transformations 
    globals()[f"cam{idx_cam}_data"] = {   
            "K": globals()[f"K{idx_cam}"], "D": globals()[f"D{idx_cam}"],
            "cam_T_world": globals()[f"cam_T{idx_cam}_world"], "cam_R_world": globals()[f"cam_R{idx_cam}_world"],
            "image": cv2.imread(os.path.join(globals()[f"c{idx_cam}_color_imgs_dir"],"img_0.png"))
        }
    camera_data.append(globals()[f"cam{idx_cam}_data"])

for cam_data in camera_data:
    cam_R_world = cam_data["cam_R_world"]
    cam_rodrigues_world = cv2.Rodrigues(cam_R_world)[0]
    cam_T_world = cam_data["cam_T_world"]

    image = cam_data["image"]
    K= cam_data["K"]
    D= cam_data["D"]

    cv2.drawFrameAxes(image, K, D, cam_rodrigues_world, cam_T_world, 0.1)

    # Save or display the updated image
    cv2.imshow('Reprojected Image', image)
    cv2.waitKey(0)

    if c == ord('a'):
        break

cv2.destroyAllWindows()

