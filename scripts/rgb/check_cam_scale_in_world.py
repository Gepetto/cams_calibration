# To run the code from repo root : python3 -m cams_calibration.check_cam_scale_in_world
# This code allows to pinpoint points and display their position in the world frame
# It can be used for checking the scale of the cameras in the world frame

import os
# Get the absolute path to the current file (script_to_launch.py)
script_path = os.path.abspath(__file__)
# Go up two directories: from 'rgb' to 'scripts', then from 'scripts' to 'repo'
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.settings import Settings
# FIRST, PARAM LOADING
settings = Settings()

import cv2
import numpy as np
from utils.calib_utils import load_cam_params, load_cam_pose, get_aruco_pose, list_cameras_with_v4l2

### Initialize cams stream
camera_dict = list_cameras_with_v4l2()
captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_dict.keys()]

for idx, cap in enumerate(captures):
    if not cap.isOpened():
        continue

    # Apply settings
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

# Define the ArUco dictionary and marker size
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_size = settings.wand_marker_size  # Marker size in meters (17.6 cm)

K1, D1 = load_cam_params(os.path.join(repo_path,"config","cam_params","c1_params_color.yaml"))
K2, D2 = load_cam_params(os.path.join(repo_path,"config","cam_params","c2_params_color.yaml"))

cam_R1_world, cam_T1_world = load_cam_pose(os.path.join(repo_path,"config","cam_params","camera1_pose.yaml"))
cam_R2_world, cam_T2_world = load_cam_pose(os.path.join(repo_path,"config","cam_params","camera2_pose.yaml"))

# Inverse the pose to get cam in world frame 
world_R1_cam = cam_R1_world.T
world_T1_cam = -cam_R1_world.T@cam_T1_world
world_T1_cam = world_T1_cam.reshape((3,))

world_R2_cam = cam_R2_world.T
world_T2_cam = -cam_R2_world.T@cam_T2_world
world_T2_cam = world_T2_cam.reshape((3,))

# Camera intrinsic parameters (from your YAML file)
camera_matrix_1 = K1
camera_matrix_2 = K2

# Distortion coefficients (from your YAML file)
dist_coeffs_1 = D1
dist_coeffs_2 = D2

# Initialize the ArUco detection parameters
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

wand_local = settings.wand_end_effector_local_pos
img_idx=0

try : 
    while True:
        frames = [cap.read()[1] for cap in captures]
            
        if not all(frame is not None for frame in frames):
            continue

        color_frame_1 = frames[0]
        color_frame_2 = frames[1]

        # Convert images to numpy arrays
        frame_1 = np.asanyarray(color_frame_1.copy())
        frame_2 = np.asanyarray(color_frame_2.copy())

        # Get the camera pose relative to the global frame defined by the ArUco marker
        transformation_matrix_1, corners_1, rvec_1, tvec_1 = get_aruco_pose(frame_1, K1, D1, detector, marker_size)
        transformation_matrix_2, corners_2, rvec_2, tvec_2 = get_aruco_pose(frame_2, K2, D2, detector, marker_size)

        if transformation_matrix_1 is not None:
            tip_pos1=tvec_1 + transformation_matrix_1[:3, :3]@wand_local 

            # Project the 3D wand tip position to 2D image coordinates
            image_points1, _ = cv2.projectPoints(tip_pos1, np.zeros(3,), np.zeros(3,), camera_matrix_1, dist_coeffs_1)
            image_points1 = image_points1[0][0]
        
            # Draw the marker and its pose on the frame for Camera 1
            cv2.aruco.drawDetectedMarkers(frame_1, [corners_1])
            cv2.drawFrameAxes(frame_1, K1, D1, rvec_1, tvec_1, 0.1)

            # Draw the reprojected wand tip on the image
            frame_1 = cv2.circle(frame_1, (int(image_points1[0]), int(image_points1[1])), 5, (0, 0, 255), -1)

        if transformation_matrix_2 is not None:
            tip_pos2=tvec_2 + transformation_matrix_2[:3, :3]@wand_local 

            # Project the 3D wand tip position to 2D image coordinates
            image_points2, _ = cv2.projectPoints(tip_pos2, np.zeros(3,), np.zeros(3,), camera_matrix_2, dist_coeffs_2)
            image_points2=image_points2[0][0]
        
            # Draw the marker and its pose on the frame for Camera 2
            cv2.aruco.drawDetectedMarkers(frame_2, [corners_2])
            cv2.drawFrameAxes(frame_2, K2, D2, rvec_2, tvec_2, 0.1)

            # Draw the reprojected wand tip on the image
            frame_2 = cv2.circle(frame_2, (int(image_points2[0]), int(image_points2[1])), 5, (0, 0, 255), -1)


        resized_color_image_1 = cv2.resize(frame_1, (640, 480), interpolation = cv2.INTER_NEAREST)
        resized_color_image_2 = cv2.resize(frame_2, (640, 480), interpolation = cv2.INTER_NEAREST) 

        images_hstack_1 = np.hstack((resized_color_image_1, resized_color_image_2))
        # Display the frames for both cameras
        cv2.imshow('Stacked images', images_hstack_1)
        
        c = cv2.waitKey(10)
        if c == ord('s'):
            print('Display tip poses')
            print('Tip Pose in cam 1 = ', tip_pos1)
            print('Tip Pose in cam 2 = ', tip_pos2)
        if c == ord('q'):
            print("quit")
            break
finally : 
    # Release the camera captures
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()
