import cv2 as cv
import yaml
import os
import glob
import numpy as np
import subprocess
from mmdeploy_runtime import PoseTracker
from utils.settings import Settings

settings = Settings()

def calibrate_camera(images_folder):
    """
    Calibrates the camera using images of a checkerboard pattern.
    Args:
        images_folder (str): Path to the folder containing checkerboard images.
    Returns:
        tuple: A tuple containing the following elements:
            - ret (float): The overall RMS re-projection error.
            - mtx (numpy.ndarray): The camera matrix.
            - dist (numpy.ndarray): The distortion coefficients.
    """

    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
 
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # # LITTLE CHECKERBOARD
    # rows = 7 #number of checkerboard rows.
    # columns = 10 #number of checkerboard columns.
    # world_scaling = 0.025 #change this to the real world square size. Or not.

    # # BIGGER CHECKERBOARD AT LAAS
    # rows = 6 #number of checkerboard rows.
    # columns = 7 #number of checkerboard columns.
    # world_scaling = 0.108 #change this to the real world square size.

    # # BIGGER CHECKERBOARD AT NUS RLS
    # rows = 5 #number of checkerboard rows.
    # columns = 7 #number of checkerboard columns.
    # world_scaling = 0.107 #change this to the real world square size.
    
    rows = settings.checkerboard_rows
    columns = settings.checkerboard_columns
    world_scaling = settings.checkerboard_scaling

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
 
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
 
        if ret == True:
 
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.imshow('img', frame)
            k = cv.waitKey(0)
 
            objpoints.append(objp)
            imgpoints.append(corners)
 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    # print('rmse:', ret)
    # print('camera matrix:\n', mtx)
    # print('distortion coeffs:', dist)
    cv.destroyAllWindows()
    # print('Rs:\n', rvecs)
    # print('Ts:\n', tvecs)
 
    return ret, mtx, dist

def calibrate_camera_runtime(images):
    objpoints = []
    imgpoints = []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rows, columns, world_scaling = settings.checkerboard_rows, settings.checkerboard_columns, settings.checkerboard_scaling
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2) * world_scaling
    
    width, height = images[0].shape[1], images[0].shape[0]
    
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
    
    ret, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    return ret, mtx, dist


def calibrate_camera_multichessboard(images_folder, checkerboard_configs):
    """
    Calibrates the camera using images of multiple checkerboard patterns.
    
    Args:
        images_folder (str): Path to the folder containing checkerboard images.
        checkerboard_configs (list of dict): List containing chessboard configurations, each with:
            - 'rows': Number of checkerboard rows.
            - 'columns': Number of checkerboard columns.
            - 'square_size': Size of each square in meters or mm.

    Returns:
        tuple: (RMS error, camera matrix, distortion coefficients).
    """

    # Load images
    images_names = sorted(glob.glob(images_folder))
    images = [cv.imread(imname, 1) for imname in images_names if imname is not None]

    # Termination criteria for corner subpixel refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Storage for object points and image points
    objpoints = []  # 3D real-world points
    imgpoints = []  # 2D image points

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        found = False  # Track if a valid chessboard was found

        for config in checkerboard_configs:
            rows, columns = config['rows'], config['columns']
            square_size = config['square_size']

            # Define object points for this chessboard size
            objp = np.zeros((rows * columns, 3), np.float32)
            objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2) * square_size

            # Try detecting this chessboard
            ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

            if ret:
                # Refine corner positions
                corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Draw the detected chessboard (for debugging)
                cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
                cv.imshow('Detected Chessboard', frame)
                cv.waitKey(50)

                objpoints.append(objp)
                imgpoints.append(corners)
                found = True
                break  # Stop once we find a valid chessboard in this image

        if not found:
            print(f"Warning: No chessboard found in image {frame.shape}")

    # Perform calibration
    width, height = images[0].shape[1], images[0].shape[0]
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)

    cv.destroyAllWindows()

    print(f"Calibration completed with RMS error: {ret}")
    return ret, mtx, dist


def is_order_consistent(corners1, corners2):
    """
    Checks if the corner order is consistent between two sets of detected corners.
    Args:
        corners1 (numpy.ndarray): Detected corners in the first camera image.
        corners2 (numpy.ndarray): Detected corners in the second camera image.
    Returns:
        bool: True if the order is consistent, False otherwise.
    """
    # Get the relative position of the first and last corners in each image
    top_left_1, bottom_right_1 = corners1[0][0], corners1[-1][0]
    top_left_2, bottom_right_2 = corners2[0][0], corners2[-1][0]
    
    # Compute the direction vectors
    vector_1 = bottom_right_1 - top_left_1
    vector_2 = bottom_right_2 - top_left_2
    
    # Check if the vectors have the same orientation
    angle_diff = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    return angle_diff > 0.9  # Adjust threshold as needed to ensure similar orientation

def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder_1, frames_folder_2):
    """
    Perform stereo calibration using images from two cameras.
    Args:
        mtx1 (numpy.ndarray): Camera matrix for the first camera.
        dist1 (numpy.ndarray): Distortion coefficients for the first camera.
        mtx2 (numpy.ndarray): Camera matrix for the second camera.
        dist2 (numpy.ndarray): Distortion coefficients for the second camera.
        frames_folder_1 (str): Path to the folder containing images from the first camera.
        frames_folder_2 (str): Path to the folder containing images from the second camera.
    Returns:
        tuple: A tuple containing:
            - ret (float): The overall RMS re-projection error.
            - R (numpy.ndarray): The rotation matrix between the coordinate systems of the first and second cameras.
            - T (numpy.ndarray): The translation vector between the coordinate systems of the first and second cameras.
    """

    #read the synched frames
    c1_images_names = sorted(glob.glob(frames_folder_1))
    c2_images_names = sorted(glob.glob(frames_folder_2))

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    
    # # LITTLE CHECKERBOARD
    # rows = 7 #number of checkerboard rows.
    # columns = 10 #number of checkerboard columns.
    # world_scaling = 0.025 #change this to the real world square size. Or not.

    # # BIGGER CHECKERBOARD AT LAAS
    # rows = 6 #number of checkerboard rows.
    # columns = 7 #number of checkerboard columns.
    # world_scaling = 0.108 #change this to the real world square size.

    # BIGGER CHECKERBOARD AT NUS RLS
    # rows = 5 #number of checkerboard rows.
    # columns = 7 #number of checkerboard columns.
    # world_scaling = 0.107 #change this to the real world square size.

    rows = settings.checkerboard_rows
    columns = settings.checkerboard_columns
    world_scaling = settings.checkerboard_scaling

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            if is_order_consistent(corners1, corners2):
 
                cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
                cv.imshow('img', frame1)

                cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
                cv.imshow('img2', frame2)
                k = cv.waitKey(1)

                objpoints.append(objp)
                imgpoints_left.append(corners1)
                imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
    cv.destroyAllWindows()
    return ret, R, T

def save_cam_params(mtx, dist, reproj, path):
    """
    Save camera parameters to a file.
    Args:
        mtx (numpy.ndarray): Camera matrix.
        dist (numpy.ndarray): Distortion coefficients.
        reproj (numpy.ndarray): Reprojection error.
        path (str): Path to the file where parameters will be saved.
    Returns:
        None
    """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    cv_file.write('reproj', reproj)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_cam_pose(filename):
    """
        Load the rotation matrix and translation vector from a YAML file.
        Args:
            filename (str): The path to the YAML file.
        Returns:
            rotation_matrix (np.ndarray): The 3x3 rotation matrix.
            translation_vector (np.ndarray): The 3x1 translation vector.
    """

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    rotation_matrix = np.array(data['rotation_matrix']['data']).reshape((3, 3))
    translation_vector = np.array(data['translation_vector']['data']).reshape((3, 1))
    
    return rotation_matrix, translation_vector

def load_cam_pose_rpy(filename):
    """
        Load the euler angles and translation vector from a YAML file.
        Args:
            filename (str): The path to the YAML file.
        Returns:
            euler (np.ndarray): The 3x1 euler sequence.
            translation_vector (np.ndarray): The 3x1 translation vector.
    """

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    euler = np.array(data['rotation_rpy']['data']).reshape((3, 1))
    translation_vector = np.array(data['translation_vector']['data']).reshape((3, 1))
    
    return euler, translation_vector


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

def save_cam_to_cam_params(mtx1, dist1, mtx2, dist2, R, T, rmse, path):
    """
    Save stereo camera calibration parameters to a file.
    Args:
        mtx1 (numpy.ndarray): Camera matrix for the first camera.
        dist1 (numpy.ndarray): Distortion coefficients for the first camera.
        mtx2 (numpy.ndarray): Camera matrix for the second camera.
        dist2 (numpy.ndarray): Distortion coefficients for the second camera.
        R (numpy.ndarray): Rotation matrix between the two cameras.
        T (numpy.ndarray): Translation vector between the two cameras.
        rmse (float): Root Mean Square Error of the calibration.
        path (str): Path to the file where the parameters will be saved.
    Returns:
        None
    """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    cv_file.write('K1', mtx1)
    cv_file.write('D1', dist1)
    cv_file.write('K2', mtx2)
    cv_file.write('D2', dist2)
    cv_file.write('R', R)
    cv_file.write('T', T)
    cv_file.write('rmse', rmse)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def load_cam_to_cam_params(path):
    """
    Loads camera-to-camera calibration parameters from a given file.
    This function reads the rotation matrix (R) and translation vector (T) from a 
    specified file using OpenCV's FileStorage. The file should contain these parameters 
    stored under the keys 'R' and 'T'.
    Args:
        path (str): The file path to the calibration parameters.
    Returns:
        tuple: A tuple containing:
            - R (numpy.ndarray): The rotation matrix.
            - T (numpy.ndarray): The translation vector.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    R = cv_file.getNode('R').mat()
    T = cv_file.getNode('T').mat()

    cv_file.release()
    return R, T


def save_global_cam_params(global_params, path):
    """
    Save the global camera transformations (rotation and translation in camera 0's frame)
    for each camera to a YAML file.
    Args:
        global_params (dict): Dictionary with camera indices as keys and (R, T) as values.
        path (str): Path to the YAML file.
    """
    fs = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        print("Error: Could not open file for writing global poses.")
        return

    for cam, (R, T) in global_params.items():
        fs.write(f'camera_{cam}_R', R)
        fs.write(f'camera_{cam}_T', T)
    fs.release()
    print("\nGlobal poses have been saved to:", path)

def load_global_cam_params(path, cam_index):
    """
    Loads the global camera transformation parameters for a specified camera
    from a YAML file. This function reads the rotation matrix (R) and translation
    vector (T) stored under the keys 'camera_{cam_index}_R' and 'camera_{cam_index}_T'.
    
    Args:
        path (str): The file path to the YAML file.
        cam_index (int): The camera index to load.
        
    Returns:
        tuple: A tuple containing:
            - R (numpy.ndarray): The rotation matrix.
            - T (numpy.ndarray): The translation vector.
    """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    R = cv_file.getNode(f'camera_{cam_index}_R').mat()
    T = cv_file.getNode(f'camera_{cam_index}_T').mat()
    cv_file.release()
    return R, T

# Function to detect the ArUco marker and estimate the camera pose
def get_aruco_pose(frame, camera_matrix, dist_coeffs, detector, marker_size):
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
    # Detect the markers in the image
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None and len(corners) > 0:
        # Extract the corners of the first detected marker for pose estimation
        # Reshape the first marker's corners for solvePnP
        corners_for_solvePnP = corners[0].reshape(-1, 2)
        
        # Estimate the pose of each marker
        _, R, t = cv.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)
        
        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv.Rodrigues(R)
        
        # Now we can form the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = t.flatten()
        
        return transformation_matrix, corners[0], R, t
    else:
        return None, None, None, None
    
def get_relative_pose_robot_in_cam(images_folder,camera_matrix,dist_coeffs, detector, marker_size):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    assert len(images)==4, "number of images to get robot base must be 4"
    
    wand_local = settings.wand_end_effector_local_pos

    wand_pos_cam_frame = []
    for ii, frame in enumerate(images):
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        
        # Detect the markers in the image
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(corners) > 0:
            # Extract the corners of the first detected marker for pose estimation
            # Reshape the first marker's corners for solvePnP
            corners_for_solvePnP = corners[0].reshape(-1, 2)
            
            # Estimate the pose of each marker
            _, R, t = cv.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)
            
            # Convert the rotation vector to a rotation matrix
            rotation_matrix, _ = cv.Rodrigues(R)
            
            # Now we can form the transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = t.flatten()
        
            wand_pos_cam_frame.append((t+rotation_matrix@wand_local).flatten())

    P1 = cam_center_robot = (wand_pos_cam_frame[0]+wand_pos_cam_frame[1])/2
    P2 = (wand_pos_cam_frame[2]+wand_pos_cam_frame[3])/2
    P3 = wand_pos_cam_frame[0]

    P2P1 = P1-P2
    P1P3 = P3-P1

    Vz = np.cross(P2P1, P1P3)
    Vy = np.cross(Vz,P2P1)
    Vx = P2P1

    x_axis = Vx/np.linalg.norm(Vx)
    y_axis = Vy/np.linalg.norm(Vy)
    z_axis = Vz/np.linalg.norm(Vz)

    # 1. Construct the rotation matrix
    cam_R_robot = np.column_stack((x_axis, y_axis, z_axis))

    return cam_center_robot, cam_R_robot

def get_relative_pose_human_in_cam(images_folder,camera_matrix,dist_coeffs, detector, marker_size):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    assert len(images)==3, "number of images to get robot base must be 4"
    
    wand_local = settings.wand_end_effector_local_pos

    wand_pos_cam_frame = []
    for ii, frame in enumerate(images):
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        
        # Detect the markers in the image
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(corners) > 0:
            # Extract the corners of the first detected marker for pose estimation
            # Reshape the first marker's corners for solvePnP
            corners_for_solvePnP = corners[0].reshape(-1, 2)
            
            # Estimate the pose of each marker
            _, R, t = cv.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)
            
            # Convert the rotation vector to a rotation matrix
            rotation_matrix, _ = cv.Rodrigues(R)
            
            # Now we can form the transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = t.flatten()
        
            wand_pos_cam_frame.append((t+rotation_matrix@wand_local).flatten())

    P1 = cam_center_human = wand_pos_cam_frame[0]
    P2 = wand_pos_cam_frame[1]
    P3 = wand_pos_cam_frame[2]

    P1P2 = P2-P1
    P1P3 = P3-P1

    Vy = np.cross(P1P2,P1P3)
    Vz = np.cross(P1P2,Vy)
    Vx = P1P2

    x_axis = Vx/np.linalg.norm(Vx)
    y_axis = Vy/np.linalg.norm(Vy)
    z_axis = Vz/np.linalg.norm(Vz)

    # 1. Construct the rotation matrix
    cam_R_human = np.column_stack((x_axis, y_axis, z_axis))

    return cam_center_human, cam_R_human

def get_relative_pose_world_in_cam(images_folder,camera_matrix,dist_coeffs, detector, marker_size):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    assert len(images)==3, "number of images to get world must be 3"

    wand_local = settings.wand_end_effector_local_pos

    wand_pos_cam_frame = []
    for ii, frame in enumerate(images):
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        
        # Detect the markers in the image
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(corners) > 0:
            # Extract the corners of the first detected marker for pose estimation
            # Reshape the first marker's corners for solvePnP
            corners_for_solvePnP = corners[0].reshape(-1, 2)
            
            # Estimate the pose of each marker
            _, R, t = cv.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)
            
            # Convert the rotation vector to a rotation matrix
            rotation_matrix, _ = cv.Rodrigues(R)
            
            # Now we can form the transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = t.flatten()
        
            wand_pos_cam_frame.append((t+rotation_matrix@wand_local).flatten())

    P1 = cam_center_world = wand_pos_cam_frame[0]
    P2 = wand_pos_cam_frame[1]
    P3 = wand_pos_cam_frame[2]

    P1P2 = P2-P1
    P1P3 = P3-P1

    Vz = np.cross(P1P2, P1P3)
    Vy = np.cross(Vz,P1P2)
    Vx = P1P2

    x_axis = Vx/np.linalg.norm(Vx)
    y_axis = Vy/np.linalg.norm(Vy)
    z_axis = Vz/np.linalg.norm(Vz)

    # 1. Construct the rotation matrix
    cam_R_world = np.column_stack((x_axis, y_axis, z_axis))

    return cam_center_world, cam_R_world

def get_relative_pose_world_in_cam2(images_folder,camera_matrix,dist_coeffs, detector, marker_size):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    for ii, frame in enumerate(images):
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        
        # Detect the markers in the image
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(corners) > 0:
            # Extract the corners of the first detected marker for pose estimation
            # Reshape the first marker's corners for solvePnP
            corners_for_solvePnP = corners[0].reshape(-1, 2)
            
            # Estimate the pose of each marker
            _, R, t = cv.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)
            
            # Convert the rotation vector to a rotation matrix
            rotation_matrix, _ = cv.Rodrigues(R)
            
            # Now we can form the transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = t.flatten()
        
    # 1. Construct the rotation matrix
    cam_R_world = rotation_matrix
    t_final = t.flatten()

    return t_final, cam_R_world

# Function to save the translation vector to a YAML file
def save_pose_rpy_to_yaml(translation_vector, rotation_sequence, filename):

    # Ensure inputs are 1D or column vectors of correct shape
    assert translation_vector.shape in [(3,), (3, 1)], "Translation vector must have shape (3,) or (3, 1)"
    assert rotation_sequence.shape in [(3,), (3, 1)], "Rotation sequence must have shape (3,) or (3, 1)"
    
    # Prepare the data to be saved in YAML format
    data = {
        'translation_vector': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': translation_vector.flatten().tolist()
        },
        'rotation_rpy': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': rotation_sequence.flatten().tolist()
        }
    }
    
    # Write to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)  # Use block style for readability

# Function to detect the ArUco marker and estimate the camera pose
def get_camera_pose(frame, camera_matrix, dist_coeffs, detector, marker_size):
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
    # Detect the markers in the image
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None and len(corners) > 0:
        # Extract the corners of the first detected marker for pose estimation
        # Reshape the first marker's corners for solvePnP
        corners_for_solvePnP = corners[0].reshape(-1, 2)
        
        # Estimate the pose of each marker
        _, R, t = cv.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)

        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv.Rodrigues(R)
        
        # Now we can form the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = t.flatten()
        
        return transformation_matrix, corners[0], R, t
    else:
        return None, None, None, None

# Function to save the rotation matrix and translation vector to a YAML file
def save_pose_matrix_to_yaml(rotation_matrix, translation_vector, filename):
    # Prepare the data to be saved in YAML format
    data = {
        'rotation_matrix': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': rotation_matrix.flatten().tolist()
        },
        'translation_vector': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': translation_vector.flatten().tolist()
        }
    }
    
    # Write to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file)

def list_cameras_with_v4l2():
    """
    Use v4l2-ctl to list all connected cameras and their device paths.
    Returns a dictionary of camera indices and associated device names.
    """
    cameras = {}
    try:
        # Get list of video devices
        output = subprocess.check_output("v4l2-ctl --list-devices", shell=True).decode("utf-8")
        devices = output.split("\n\n")  # Separate different devices
        for device in devices:
            lines = device.split("\n")
            if len(lines) > 1:
                device_name = lines[0].strip()
                video_path = lines[1].strip()
                if "/dev/video" in video_path:
                    index = int(video_path.split("video")[-1])
                    cameras[index] = device_name
    except Exception as e:
        print("Error using v4l2-ctl:", e)
    # Sort the cameras dictionary by index
    sorted_cameras = {k: cameras[k] for k in sorted(cameras)}
    print(sorted_cameras)
    return sorted_cameras

def get_cameras_params(K1, D1, K2, D2, R, T):
    dict_cam = {
        "cam1": {
            "mtx":np.array(K1),
            "dist":D1,
            "rotation":np.eye(3),
            "translation":[
                0.,
                0.,
                0.,
            ],
        },
        "cam2": {
            "mtx":np.array(K2),
            "dist":D2,
            "rotation":R,
            "translation":T,
        },
    }

    rotations=[]
    translations=[]
    dists=[]
    mtxs=[]
    projections=[]

    for cam in dict_cam :
        rotation=np.array(dict_cam[cam]["rotation"])
        rotations.append(rotation)
        translation=np.array([dict_cam[cam]["translation"]]).reshape(3,1)
        translations.append(translation)
        projection = np.concatenate([rotation, translation], axis=-1)
        projections.append(projection)
        dict_cam[cam]["projection"] = projection
        dists.append(dict_cam[cam]["dist"])
        mtxs.append(dict_cam[cam]["mtx"])
    return mtxs, dists, projections, rotations, translations


# MMPOSE VISUALISATION CONFIG
# This configuration is used to define the skeleton, palette, link color, point color, and sigmas for the visualization of the pose estimator.

VISUALIZATION_CFG = dict(
    body26=dict(
        skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 2), (5, 18),(6, 18), (17,18), # Head, shoulders, and neck connections
                    (5, 7), (7, 9),                                                              # Right arm connections
                    (6, 8), (8, 10),                                                             # Left arm connections
                    (18, 19),                                                                    # Trunk connection
                    (11, 13), (13, 15), (15, 20), (15, 22), (15, 24),                            # Left leg and foot connections
                    (12, 14), (14, 16), (16, 21), (16, 23), (16, 25),                            # Right leg and foot connections
                    (12, 19), (11, 19)],                                                         # Hip connection

        # Updated palette
        palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]],
    
        # Updated link color
        link_color = [
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
            2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
            2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],

        # Updated point color
        point_color = [
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4,
            5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5,
            5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        sigmas = [0.026] * 26
    ),
    body17=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
        palette=[(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
                 (255, 153, 255), (153, 204, 255), (255, 102, 255),
                 (255, 51, 255), (102, 178, 255), (51, 153, 255),
                 (255, 153, 153), (255, 102, 102), (255, 51, 51),
                 (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
                 (0, 0, 255), (255, 0, 0), (255, 255, 255)],
        link_color=[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ],
        point_color=[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]),
    coco_wholebody=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                  (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                  (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                  (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                  (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                  (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                  (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                  (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                  (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                  (129, 130), (130, 131), (131, 132)],
        palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0), (255, 255, 255),
                 (255, 153, 255), (102, 178, 255), (255, 51, 51)],
        link_color=[
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1,
            1, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2,
            2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1,
            1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068,
            0.066, 0.066, 0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043,
            0.040, 0.035, 0.031, 0.025, 0.020, 0.023, 0.029, 0.032, 0.037,
            0.038, 0.043, 0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012,
            0.012, 0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007, 0.007,
            0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011,
            0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010, 0.034, 0.008,
            0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009, 0.009,
            0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
            0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
            0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032,
            0.02, 0.019, 0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047,
            0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
            0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031
        ]))

class PoseTrackerEstimator:
    def __init__(self, det_model, pose_model, device='cuda', thr=0.1, skeleton = 'body26'):
        self._det_model = det_model
        self._pose_model = pose_model
        self._device = device
        self._thr = thr
        self._skeleton = skeleton
        self.tracker = PoseTracker(det_model, pose_model, device)
        self.VISUALISATION_CFG = VISUALIZATION_CFG
        self.sigmas = VISUALIZATION_CFG[self._skeleton]['sigmas']
        self.state =  self.tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=self.sigmas)

    def estimate(self, frame):
        results = self.tracker(self.state, frame, detect=-1)
        return results
    
    def visualize(self, 
                  frame,
                  results,
                  idx,
                  resize=1280):
        
        skeleton = self.VISUALISATION_CFG[self._skeleton]['skeleton']
        palette = self.VISUALISATION_CFG[self._skeleton]['palette']
        link_color = self.VISUALISATION_CFG[self._skeleton]['link_color']
        point_color = self.VISUALISATION_CFG[self._skeleton]['point_color']

        scale = resize / max(frame.shape[0], frame.shape[1])
        keypoints, bboxes, _ = results
        scores = keypoints[..., 2]
        keypoints = (keypoints[..., :2] * scale).astype(int)
        bboxes *= scale
        img = cv.resize(frame, (0, 0), fx=scale, fy=scale)

        for kpts, score, bbox in zip(keypoints, scores, bboxes):
            show = [1] * len(kpts)

            for (u, v), color in zip(skeleton, link_color):
                if score[u] > self._thr and score[v] > self._thr:
                    cv.line(img, kpts[u], tuple(kpts[v]), palette[color], 1,
                            cv.LINE_AA)
                else:
                    show[u] = show[v] = 0

            for kpt, show, color in zip(kpts, show, point_color):
                if show:
                    cv.circle(img, kpt, 1, palette[color], 2, cv.LINE_AA)
           
        cv.imshow('pose_tracker'+str(idx), img)
        # If 'q' is pressed, exit visualization
        if cv.waitKey(1) & 0xFF == ord('q'):
            return False

        return True

def DLT(projections, points):
    """
    Perform Direct Linear Transformation (DLT) for adaptive triangulation.
    This function computes the 3D coordinates of a point given its projections
    in multiple views using the DLT algorithm. It constructs a system of linear
    equations from the projection matrices and the corresponding 2D points, and
    then solves it using Singular Value Decomposition (SVD).
    Parameters:
    -----------
    projections : list of numpy.ndarray
        A list of 3x4 projection matrices for each view.
    points : list of numpy.ndarray
        A list of 2D points corresponding to each view. Each element in the list
        is an array of shape (n, 2), where n is the number of points.
    Returns:
    --------
    numpy.ndarray
        A 1D array of length 3 representing the 3D coordinates of the point.
    """
    
    A=[]
    for i in range(len(projections)):
        P=projections[i]
        point = points[i]

        for j in range (len(point)):
            A.append(point[j][1]*P[2,:] - P[1,:])
            A.append(P[0,:] - point[j][0]*P[2,:])

    A = np.array(A).reshape((-1,4))
    B = A.transpose() @ A
    _, _, Vh = np.linalg.svd(B, full_matrices = False)

    return Vh[3,0:3]/Vh[3,3]

def triangulate_points(keypoints_list, mtxs, dists, projections):
    """
    Triangulates 3D points from multiple 2D keypoints using camera matrices and distortion coefficients.
    Args:
        keypoints_list (list of list of tuples): A list where each element is a list of 2D keypoints for a single frame.
        mtxs (list of numpy.ndarray): A list of camera matrices (K) for each frame.
        dists (list of numpy.ndarray): A list of distortion coefficients (D) for each frame.
        projections (list of numpy.ndarray): A list of projection matrices (R,t) for each frame.
    Returns:
        numpy.ndarray: An array of 3D points triangulated from the input 2D keypoints.
    """

    p3ds_frame=[]
    undistorted_points = []

    for ii in range(len(keypoints_list)):
        points = keypoints_list[ii] 
        distCoeffs_mat = np.array([dists[ii]]).reshape(-1, 1)
        points_undistorted = cv.undistortPoints(np.array(points).reshape(-1, 1, 2), mtxs[ii], distCoeffs_mat)
        undistorted_points.append(points_undistorted)

    for point_idx in range(26):
        points_per_point = [undistorted_points[i][point_idx] for i in range(len(undistorted_points))]
        _p3d = DLT(projections, points_per_point)
        p3ds_frame.append(_p3d)

    return np.array(p3ds_frame)

def calculate_anthropometric_segment_lengths(height, gender):
    lengths_names = [
            "L_pelvis_width",
            "L_abdomen",
            "L_thorax_cerv",
            "L_thorax_supr",
            "L_upperarm",
            "L_lowerarm",
            "L_upperleg",
            "L_lowerleg",
        ]

    ratios = np.array(
        [
            0.0634 if gender == "f" else 0.0505,  # L_pelvis_width (Dumas 2007)
            0.1183 if gender == "f" else 0.1237,  # L_abdomen MPT  from XYP to OMPH (De Leva 1996)
            0.1314 if gender == "f" else 0.1390,  # L_thorax UPT from CERV to XYPH (De Leva 1996)
            0.0821 if gender == "f" else 0.0980,  # from SUPR to XYPH (De Leva 1996)
            0.1510 if gender == "f" else 0.1531,  # L_upperarm (Dumas 2007)
            0.1534 if gender == "f" else 0.1593,  # L_lowerarm (Dumas 2007)
            0.2354 if gender == "f" else 0.2441,  # L_upperleg (Dumas 2007)
            0.2410 if gender == "f" else 0.2446,  # L_lowerleg (Dumas 2007)
        ]
    )

    lengths = np.round(ratios * height, 3)  # mm accuracy
    dict_lengths = dict(zip(lengths_names, lengths))
    return dict_lengths


# AUTOCALIBRATION WITH RTMPOSE

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
        cap = cv.VideoCapture(cam_idx, cv.CAP_V4L2)
        if not cap.isOpened():
            print(f"WARNING: could not open camera {cam_idx}")
        else:
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*settings.fourcc))
            cap.set(cv.CAP_PROP_FRAME_WIDTH, settings.width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, settings.height)
            cap.set(cv.CAP_PROP_FPS, settings.fs)
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
    mp4_fourcc = cv.VideoWriter_fourcc(*"mp4v")
    fps = float(settings.fs)

    def _find_existing_sessions(video_dirs_map, cam_ids):
        """
        Scan existing calib_video_*.mp4 files and rebuild complete sessions
        (only sessions that have a file for every camera are kept).
        """
        sessions = {}
        for cam_idx in cam_ids:
            pattern = os.path.join(video_dirs_map[cam_idx], "calib_video_*.mp4")
            for path in glob.glob(pattern):
                base = os.path.basename(path)
                try:
                    sid = int(base.split("_")[-1].split(".")[0])
                except (ValueError, IndexError):
                    continue
                sessions.setdefault(sid, {})[cam_idx] = path
        complete = []
        for sid, paths in sessions.items():
            if set(paths.keys()) == set(cam_ids):
                complete.append((sid, paths))
        complete.sort(key=lambda x: x[0])
        return [p for _, p in complete]

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
                tile = cv.resize(frame, (640, 480), interpolation=cv.INTER_NEAREST)
                cv.putText(
                    tile,
                    f"Cam {cam_idx}",
                    (5, 25),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )
                preview_tiles.append(tile)

            mosaic = build_mosaic(preview_tiles, cols=2)
            if recording and mosaic is not None:
                cv.putText(
                    mosaic,
                    f"REC #{session_id}",
                    (10, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv.LINE_AA,
                )

            if mosaic is not None:
                cv.imshow("RGB calib mosaic", mosaic)

            key = cv.waitKey(1) & 0xFF

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
                        writers[cam_idx] = cv.VideoWriter(
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
        cv.destroyAllWindows()

    # If no new recordings, try to reuse existing ones on disk
    if not recorded_sessions:
        existing = _find_existing_sessions(video_dirs, cam_indices)
        if existing:
            print("\nNo new recording made; reusing existing sessions on disk.")
            recorded_sessions.extend(existing)

    print("\nRecorded sessions (including reused if found):")
    for i, sess in enumerate(recorded_sessions):
        print(f"  Session #{i}:")
        for cam_idx, path in sess.items():
            print(f"    cam {cam_idx}: {path}")

    return recorded_sessions


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

    cap = cv.VideoCapture(video_path)
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
#  Liu-style binocular auto-calibration helpers
#  (RTMPose -> correspondences -> E -> R,t -> triangulate -> refine K)
# -------------------------------------------------------------------

# RTMPose "body26" joints we consider reliable for geometry:
# torso + arms + legs; we drop hands, head, feet.
RELIABLE_JOINTS_BODY26 = [
    5, 6, 7, 8, 9, 10,      # shoulders, elbows, wrists
    11, 12, 13, 14, 15, 16, # hips, knees, ankles
    18, 19,                 # trunk points used for abdomen+thorax_cerv
]

def build_correspondences(
    kps1,
    kps2,
    conf1,
    conf2,
    conf_thresh=0.5,
    restrict_to_reliable=True,
    return_indices=False,
):
    """
    Aggregate 2D–2D correspondences across all frames/joints.

    kps*:  (T, J, 2)
    conf*: (T, J)

    Args:
        conf_thresh:          minimum confidence in both views.
        restrict_to_reliable: if True, only use a subset of body26 joints
                              (no head / hands / feet) to reduce noise.
        return_indices:       if True, also return (t, j) indices for each match.

    Returns:
        pts1_pix, pts2_pix: (N, 2) pixel coordinates (float32)
        idxs (optional):    (N, 2) int array of (t_idx, j_idx)
    """
    T, J, _ = kps1.shape
    pts1 = []
    pts2 = []
    idxs = []

    if restrict_to_reliable:
        joint_iter = [j for j in RELIABLE_JOINTS_BODY26 if j < J]
    else:
        joint_iter = range(J)

    for t in range(T):
        for j in joint_iter:
            if conf1[t, j] > conf_thresh and conf2[t, j] > conf_thresh:
                p1 = kps1[t, j]
                p2 = kps2[t, j]
                if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                    continue
                pts1.append(p1)
                pts2.append(p2)
                if return_indices:
                    idxs.append((t, j))

    if not pts1:
        raise RuntimeError("No valid correspondences above confidence threshold.")

    pts1_pix = np.asarray(pts1, dtype=np.float32)
    pts2_pix = np.asarray(pts2, dtype=np.float32)

    if return_indices:
        idxs = np.asarray(idxs, dtype=np.int32)
        return pts1_pix, pts2_pix, idxs

    return pts1_pix, pts2_pix


def estimate_extrinsics_epipolar(
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
    Step 1 (Sec. 4.2.1 in Liu et al.): estimate relative pose from 2D–2D
    correspondences using essential matrix + recoverPose.

    IMPORTANT: we interpret the output as the transform from cam1 to cam2:
        X_2 = R * X_1 + t
    in *normalized* (undistorted) coordinates.

    Returns:
        R (3,3), t (3,), pts1_pix_inliers (N,2), pts2_pix_inliers (N,2)
    """
    # 2D-2D correspondences in pixels
    pts1_pix, pts2_pix = build_correspondences(kps1, kps2, conf1, conf2, conf_thresh)

    # Undistort to normalized coordinates (camera coordinates, K = I)
    pts1_norm = cv.undistortPoints(
        pts1_pix.reshape(-1, 1, 2), K1, D1
    ).reshape(-1, 2)
    pts2_norm = cv.undistortPoints(
        pts2_pix.reshape(-1, 1, 2), K2, D2
    ).reshape(-1, 2)

    # Essential matrix in normalized space (focal = 1, pp = (0,0))
    E, mask = cv.findEssentialMat(
        pts1_norm,
        pts2_norm,
        1.0,
        (0.0, 0.0),
        method=cv.RANSAC,
        prob=0.999,
        threshold=1e-3,
    )
    if E is None:
        raise RuntimeError("findEssentialMat failed.")

    mask = mask.ravel().astype(bool)
    if not np.any(mask):
        raise RuntimeError("All correspondences rejected by RANSAC.")

    pts1_in = pts1_norm[mask]
    pts2_in = pts2_norm[mask]

    # recoverPose returns R, t such that x2 ~ R x1 + t (normalized camera coords)
    _, R, t, _ = cv.recoverPose(E, pts1_in, pts2_in)

    # Keep only inlier pixels as well (for later reprojection/triangulation)
    pts1_pix_inliers = pts1_pix[mask]
    pts2_pix_inliers = pts2_pix[mask]

    return R, t.reshape(3), pts1_pix_inliers, pts2_pix_inliers


def triangulate_correspondences(
    K1,
    D1,
    K2,
    D2,
    R,
    t,
    pts1_pix,
    pts2_pix,
):
    """
    Triangulate 3D points in the *base camera (cam1) frame* from 2D–2D
    correspondences and current extrinsics.

    We:
      - undistort -> normalized coords in each camera;
      - use P1 = [I | 0], P2 = [R | t] in normalized space;
      - cv2.triangulatePoints -> homogeneous -> 3D in cam1 frame.

    Returns:
        X_cam1: (N, 3)
    """
    if pts1_pix.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pts1_norm = cv.undistortPoints(
        pts1_pix.reshape(-1, 1, 2), K1, D1
    ).reshape(-1, 2)
    pts2_norm = cv.undistortPoints(
        pts2_pix.reshape(-1, 1, 2), K2, D2
    ).reshape(-1, 2)

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)  # cam1 at origin
    P2 = np.hstack([R, t.reshape(3, 1)]).astype(np.float64)           # cam2 in cam1 frame

    pts1_2xN = pts1_norm.T  # (2, N)
    pts2_2xN = pts2_norm.T

    X_h = cv.triangulatePoints(P1, P2, pts1_2xN, pts2_2xN)  # (4, N)
    X = (X_h[:3] / X_h[3]).T  # (N, 3)
    return X


def optimize_intrinsics_linear_ls(K_prev, X_cam, uv_pix):
    """
    Linear least-squares update of intrinsics (fx, fy, cx, cy) given:
      - 3D points in camera coordinates X_cam (N,3),
      - observed pixel locations uv_pix (N,2).

    We assume the standard pinhole model without skew:
        u = fx * (X/Z) + cx
        v = fy * (Y/Z) + cy

    Build a linear system in theta = [fx, cx, fy, cy]^T.

    Returns:
        K_new (3,3)
    """
    fx0 = K_prev[0, 0]
    fy0 = K_prev[1, 1]
    cx0 = K_prev[0, 2]
    cy0 = K_prev[1, 2]

    A = []
    b = []

    for X, uv in zip(X_cam, uv_pix):
        Xc, Yc, Zc = float(X[0]), float(X[1]), float(X[2])
        if Zc <= 1e-6:
            continue  # behind camera or invalid

        u, v = float(uv[0]), float(uv[1])

        # u equation: u = fx * X/Z + cx
        A.append([Xc / Zc, 1.0, 0.0, 0.0])
        b.append(u)

        # v equation: v = fy * Y/Z + cy
        A.append([0.0, 0.0, Yc / Zc, 1.0])
        b.append(v)

    if len(b) < 4:
        # Not enough constraints, keep previous intrinsics
        return K_prev

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    fx, cx, fy, cy = theta

    # Small regularisation: keep intrinsics within a reasonable band around the
    # previous ones to avoid exploding due to bad conditioning.
    fx = float(np.clip(fx, 0.5 * fx0, 1.5 * fx0))
    fy = float(np.clip(fy, 0.5 * fy0, 1.5 * fy0))
    # principal point drift is usually small; allow +- 100 px around initial
    cx = float(np.clip(cx, cx0 - 100.0, cx0 + 100.0))
    cy = float(np.clip(cy, cy0 - 100.0, cy0 + 100.0))

    K_new = K_prev.copy()
    K_new[0, 0] = fx
    K_new[1, 1] = fy
    K_new[0, 2] = cx
    K_new[1, 2] = cy
    return K_new


def compute_reprojection_error(
    K1,
    D1,
    K2,
    D2,
    R,
    t,
    pts1_pix,
    pts2_pix,
):
    """
    Compute symmetric RMSE reprojection error for a stereo pair:

      1. Triangulate 3D points in cam1 frame (normalized model).
      2. Reproject into both cameras using K1, K2 and R, t.
      3. Compute sqrt(mean squared error) over both cameras.

    Returns:
        rmse (float, pixels)
    """
    if pts1_pix.shape[0] == 0:
        return np.nan

    X_cam1 = triangulate_correspondences(K1, D1, K2, D2, R, t, pts1_pix, pts2_pix)

    total_err = 0.0
    total_count = 0

    for X, p1_obs, p2_obs in zip(X_cam1, pts1_pix, pts2_pix):
        X = np.asarray(X, dtype=np.float64)
        Xc1 = X
        Xc2 = R @ X + t.reshape(3)

        if Xc1[2] <= 1e-6 or Xc2[2] <= 1e-6:
            continue

        # cam1 projection
        x1_n = Xc1[:2] / Xc1[2]
        p1_hat = (K1 @ np.array([x1_n[0], x1_n[1], 1.0], dtype=np.float64))[:2]

        # cam2 projection
        x2_n = Xc2[:2] / Xc2[2]
        p2_hat = (K2 @ np.array([x2_n[0], x2_n[1], 1.0], dtype=np.float64))[:2]

        err1 = float(np.sum((p1_hat - p1_obs) ** 2))
        err2 = float(np.sum((p2_hat - p2_obs) ** 2))

        total_err += err1 + err2
        total_count += 2

    if total_count == 0:
        return np.nan

    rmse = float(np.sqrt(total_err / total_count))
    return rmse

def estimate_scale_from_anthropometry(
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
    height_m,
    gender,
    conf_thresh=0.8,
):
    """
    Estimate a global metric scale for the stereo rig using anthropometric
    constraints on a single subject.

    Steps:
      - build multi-frame 2D correspondences on reliable joints only;
      - triangulate 3D joints in cam1 frame with current R, t, K1, K2;
      - for each limb segment, compute the average 3D length over time;
      - compare to target lengths from anthropometry (Dumas/De Leva);
      - return a single scale factor s, so that:
            L_target ~= s * L_reconstructed.
    """
    # 1) Anthropometric target lengths (same unit as height_m, typically meters)
    seg_lengths = calculate_anthropometric_segment_lengths(height_m, gender)

    target_pelvis    = seg_lengths["L_pelvis_width"]
    target_trunk     = seg_lengths["L_abdomen"] + seg_lengths["L_thorax_cerv"]
    target_upperarm  = seg_lengths["L_upperarm"]
    target_lowerarm  = seg_lengths["L_lowerarm"]
    target_upperleg  = seg_lengths["L_upperleg"]
    target_lowerleg  = seg_lengths["L_lowerleg"]

    # 2) Correspondences on reliable joints, keep (t, j) indices
    pts1_pix, pts2_pix, idxs = build_correspondences(
        kps1,
        kps2,
        conf1,
        conf2,
        conf_thresh=conf_thresh,
        restrict_to_reliable=True,
        return_indices=True,
    )

    # 3) Triangulate in cam1 frame
    X_cam1 = triangulate_correspondences(K1, D1, K2, D2, R, t, pts1_pix, pts2_pix)
    X_cam1 = np.asarray(X_cam1, dtype=np.float64)  # (N, 3)

    # 4) Organize as [T, J, 3] for easy segment-length computation
    T = kps1.shape[0]
    J = kps1.shape[1]
    X_by_frame_joint = np.full((T, J, 3), np.nan, dtype=np.float64)
    for X, (t_idx, j_idx) in zip(X_cam1, idxs):
        if 0 <= t_idx < T and 0 <= j_idx < J:
            X_by_frame_joint[t_idx, j_idx, :] = X

    def collect_segment_scale(joint_pairs, L_target):
        """
        For a list of (j1, j2) joint index pairs and a target length, compute
        the average reconstructed length over time and return L_target / L_rec.
        """
        all_lengths = []
        for (j1, j2) in joint_pairs:
            if j1 >= J or j2 >= J:
                continue
            X1 = X_by_frame_joint[:, j1, :]  # (T, 3)
            X2 = X_by_frame_joint[:, j2, :]
            valid = np.isfinite(X1[:, 0]) & np.isfinite(X2[:, 0])
            if np.count_nonzero(valid) < 5:
                continue
            dists = np.linalg.norm(X1[valid] - X2[valid], axis=1)
            if dists.size > 0:
                all_lengths.append(np.mean(dists))

        if not all_lengths:
            return None
        L_rec = float(np.mean(all_lengths))
        if L_rec <= 1e-6:
            return None
        return L_target / L_rec

    scales = []

    # Your mapping: -------------------------------
    # L_pelvis_width   : 11–12
    s_pelvis = collect_segment_scale([(11, 12)], target_pelvis)
    if s_pelvis is not None:
        scales.append(s_pelvis)
    else:
        print("  [Anthropometry] Warning: could not compute pelvis width scale.")

    # L_abdomen + L_thorax_cerv : 19–18
    s_trunk = collect_segment_scale([(19, 18)], target_trunk)
    if s_trunk is not None:
        scales.append(s_trunk)
    else:
        print("  [Anthropometry] Warning: could not compute trunk length scale.")

    # L_upperarm : 6–8 (right), 5–7 (left)
    s_upperarm = collect_segment_scale([(6, 8), (5, 7)], target_upperarm)
    if s_upperarm is not None:
        scales.append(s_upperarm)
    else:
        print("  [Anthropometry] Warning: could not compute upper arm scale.")

    # L_lowerarm : 8–10 (right), 7–9 (left)
    s_lowerarm = collect_segment_scale([(8, 10), (7, 9)], target_lowerarm)
    if s_lowerarm is not None:
        scales.append(s_lowerarm)
    else:
        print("  [Anthropometry] Warning: could not compute lower arm scale.")

    # L_upperleg : 12–14 (right), 11–13 (left)
    s_upperleg = collect_segment_scale([(12, 14), (11, 13)], target_upperleg)
    if s_upperleg is not None:
        scales.append(s_upperleg)
    else:
        print("  [Anthropometry] Warning: could not compute upper leg scale.")

    # L_lowerleg : 14–16 (right), 13–15 (left)
    s_lowerleg = collect_segment_scale([(14, 16), (13, 15)], target_lowerleg)
    if s_lowerleg is not None:
        scales.append(s_lowerleg)
    else:
        print("  [Anthropometry] Warning: could not compute lower leg scale.")
    # ---------------------------------------------

    if not scales:
        print("[WARN] Anthropometric scale estimation failed (no valid segments).")
        return 1.0

    scales = np.array(scales, dtype=np.float64)
    s_global = float(np.median(scales))
    print(f"  [Anthropometry] per-segment scales: {scales}")
    print(f"  [Anthropometry] chosen global scale s = {s_global:.3f}")
    return s_global


def binocular_autocalib_from_human(
    kps1,
    kps2,
    conf1,
    conf2,
    K1_init,
    D1,
    K2_init,
    D2,
    conf_thresh=0.8,
    max_outer_iters=100,
    verbose=True,
):
    """
    Implement the binocular auto-calibration loop of Liu et al. (Sec. 4.2):

      - start from initial intrinsics (checkerboard or factory values);
      - estimate extrinsics from epipolar geometry (E + recoverPose);
      - iterate:
         * triangulate 3D joints;
         * refine intrinsics (linear LS on fx, fy, cx, cy);
         * re-estimate extrinsics with updated intrinsics;
         * monitor reprojection RMSE.

    Inputs:
        kps1, kps2 : (T, J, 2) RTMPose keypoints
        conf1, conf2 : (T, J) confidence
        K1_init, K2_init : (3,3) intrinsics
        D1, D2 : (distortion vectors) are used only for undistortion
    Returns:
        K1_opt, D1_opt, K2_opt, D2_opt, R_opt, t_opt
    """

    # Copy intrinsics so we don't mutate the originals
    K1 = K1_init.copy()
    K2 = K2_init.copy()

    # Initial extrinsics from epipolar geometry
    R, t, pts1_pix, pts2_pix = estimate_extrinsics_epipolar(
        kps1, kps2, conf1, conf2, K1, D1, K2, D2, conf_thresh
    )

    rmse_prev = np.inf

    for it in range(max_outer_iters):
        if verbose:
            print(f"\n  [Iter {it}]")

        # 1) Triangulate in base camera frame with current params
        X_cam1 = triangulate_correspondences(K1, D1, K2, D2, R, t, pts1_pix, pts2_pix)

        # Filter out points behind cameras
        Xc1 = X_cam1
        Xc2 = (R @ X_cam1.T + t.reshape(3, 1)).T
        valid = (Xc1[:, 2] > 1e-6) & (Xc2[:, 2] > 1e-6)
        if not np.any(valid):
            print("  No valid 3D points with positive depth in both views, stopping.")
            break

        Xc1 = Xc1[valid]
        Xc2 = Xc2[valid]
        pts1_valid = pts1_pix[valid]
        pts2_valid = pts2_pix[valid]

        # 2) Intrinsics refinement (linear LS) for each camera
        K1 = optimize_intrinsics_linear_ls(K1, Xc1, pts1_valid)
        K2 = optimize_intrinsics_linear_ls(K2, Xc2, pts2_valid)

        # 3) Re-estimate extrinsics with updated intrinsics
        R, t, pts1_pix, pts2_pix = estimate_extrinsics_epipolar(
            kps1, kps2, conf1, conf2, K1, D1, K2, D2, conf_thresh
        )

        # 4) Evaluate reprojection RMSE for monitoring
        rmse = compute_reprojection_error(K1, D1, K2, D2, R, t, pts1_pix, pts2_pix)
        if verbose:
            print(f"    Reprojection RMSE (pixels): {rmse:.3f}")

        if np.isfinite(rmse) and abs(rmse_prev - rmse) < 1e-3:
            if verbose:
                print("    Converged (RMSE change below 1e-3).")
            break

        rmse_prev = rmse

    # We do not change distortion in this implementation (paper ignores it).
    return K1, D1, K2, D2, R, t



def autocalibrate_from_human(
    recorded_sessions,
    config_dir: str,
    conf_thresh: float = 0.8,
    height_m: float = 1.80,
    gender: str = "m",
):

    if not recorded_sessions:
        print("No recorded sessions, nothing to calibrate.")
        return

    last_session = recorded_sessions[-1]  # dict[cam_idx -> video_path]
    # keys can be "0"/"2" (str) or ints; sort numerically for robustness
    cam_indices = sorted(last_session.keys(), key=lambda x: int(x))
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

    # 4) Binocular auto-calibration base_cam -> other cams (Liu-style)
    kps_base = kps_by_cam[base_cam]
    conf_base = conf_by_cam[base_cam]


    for cam_idx in cam_indices:
        if cam_idx == base_cam:
            continue

        print(f"\n=== Auto-calibrating base={base_cam} vs cam={cam_idx} from human ===")

        kps_other = kps_by_cam[cam_idx]
        conf_other = conf_by_cam[cam_idx]

        K1_init = K_by_cam[base_cam]
        D1 = D_by_cam[base_cam]
        K2_init = K_by_cam[cam_idx]
        D2 = D_by_cam[cam_idx]

        # Run Liu-style binocular auto-calibration
        K1_opt, D1_opt, K2_opt, D2_opt, R_opt, t_opt = binocular_autocalib_from_human(
            kps_base,
            kps_other,
            conf_base,
            conf_other,
            K1_init,
            D1,
            K2_init,
            D2,
            conf_thresh=conf_thresh,
            max_outer_iters=100,
            verbose=True,
        )

        # --- Anthropometric scale: fix metric baseline for t_opt ---
        s_scale = estimate_scale_from_anthropometry(
            kps_base,
            kps_other,
            conf_base,
            conf_other,
            K1_opt,
            D1_opt,
            K2_opt,
            D2_opt,
            R_opt,
            t_opt,
            height_m=height_m,
            gender=gender,
            conf_thresh=conf_thresh,
        )
        t_opt = s_scale * t_opt

        # Build correspondences again (full set above threshold) and compute final RMSE
        pts1_pix, pts2_pix = build_correspondences(
            kps_base, kps_other, conf_base, conf_other, conf_thresh
        )
        rmse_final = compute_reprojection_error(
            K1_opt, D1_opt, K2_opt, D2_opt, R_opt, t_opt, pts1_pix, pts2_pix
        )
        print(
            f"  Final pairwise RMSE (base={base_cam}, cam={cam_idx}) "
            f"with auto-calibrated extrinsics: {rmse_final:.3f} px"
        )

        # At this stage, you have two options:
        #   - keep the original intrinsics from checkerboard and only trust R_opt, t_opt;
        #   - or trust the refined intrinsics too.
        #
        # For safety / compatibility with the rest of your pipeline, we
        # *save* only R_opt, t_opt while keeping the original intrinsics
        # in the YAML. If you want fully Liu-style auto-calibration,
        # replace K_by_cam[...] by K1_opt / K2_opt below.

        out_path = os.path.join(
            cam_params_dir,
            f"c{base_cam}_to_c{cam_idx}_params_color_rtmpose_autocalib.yaml",
        )

        R_opt=R_opt.T
        t_opt = -R_opt @ t_opt.reshape(3)

        save_cam_to_cam_params(
            K_by_cam[base_cam],   # or K1_opt
            D_by_cam[base_cam],   # or D1_opt
            K_by_cam[cam_idx],    # or K2_opt
            D_by_cam[cam_idx],    # or D2_opt
            R_opt,
            t_opt.reshape(3, 1),
            rmse_final,
            out_path,
        )
        print(f"  Saved auto-calibrated extrinsics to: {out_path}")
