#!/usr/bin/env python3
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
print(SRC_ROOT)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2 as cv
import numpy as np

from utils.calib_utils import load_transformation, save_cam_to_cam_params, transform_to_local_frame

def load_cam_params(path):
    """
    Loads camera parameters from a given file.
    Args:
        path (str): The path to the file containing the camera parameters.
    Returns:
        tuple: A tuple containing the camera matrix and distortion matrix.
            - camera_matrix (numpy.ndarray): The camera matrix.
            - dist_matrix (numpy.ndarray): The distortion matrix.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return camera_matrix, dist_matrix


number_of_camera=4

if number_of_camera<=2:
    print("\nThe total number of camera must exceed 2 to run this script.\n If it doesn't then you don't need this script!\n")
    exit(-1)

if number_of_camera%2!=0:
     print("\nThe total number of camera must be even!\n")
     exit(-2)


K1, D1 = load_cam_params(f"/root/workspace/cams_calibration/config/cam_params/c0_params_color.yaml")
soder_dir = f"/root/workspace/cams_calibration/config/cam_params/calib_mocap_to_cam"

for i in range(4,number_of_camera*2,2):

    K2, D2 = load_cam_params(f"/root/workspace/cams_calibration/config/cam_params/c{i}_params_color.yaml")


    cam2cam_dir = f"/root/workspace/cams_calibration/config/cam_params/c0_to_c{i}_params_color.yaml"

    R_c1_in_mocap, d_c1_in_mocap, _, _ = load_transformation(f"{soder_dir}/calib_mocap_2_cam0/soder.txt")
    R_c2_in_mocap, d_c2_in_mocap, _, _ = load_transformation(f"{soder_dir}/calib_mocap_2_cam{i}/soder.txt")

    R_c1_in_c2 = np.transpose(R_c2_in_mocap)@R_c1_in_mocap
    d_c1_in_c2 = transform_to_local_frame(d_c1_in_mocap, d_c2_in_mocap, R_c2_in_mocap)

    print("rot", R_c1_in_c2)
    print("pos", d_c1_in_c2)

    save_cam_to_cam_params(K1, 
                        D1, 
                        K2, 
                        D2, 
                        R_c1_in_c2, d_c1_in_c2, 0.0, 
                        cam2cam_dir)