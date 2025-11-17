import cv2 as cv
import yaml
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