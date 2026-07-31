import cv2 as cv
import yaml
import glob
import numpy as np
import subprocess
from utils.settings import Settings
import os
import glob

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
            k = cv.waitKey(50)
 
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
    """

    c1_images_names = sorted(glob.glob(frames_folder_1))
    c2_images_names = sorted(glob.glob(frames_folder_2))

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rows = settings.checkerboard_rows
    columns = settings.checkerboard_columns
    world_scaling = settings.checkerboard_scaling

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    imgpoints_left = []
    imgpoints_right = []
    objpoints = []

    window_names = ['img', 'img2']
    for name in window_names:
        cv.namedWindow(name, cv.WINDOW_NORMAL)

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

                # Wait for a keypress, OR for either window to be closed via the
                # close (X) icon. waitKey(0) alone blocks forever on a closed
                # window since closing doesn't generate a keypress. Either event
                # just advances to the next image pair, same as any keypress did.
                while True:
                    k = cv.waitKey(30)
                    if k != -1:
                        break
                    if any(cv.getWindowProperty(name, cv.WND_PROP_VISIBLE) < 1 for name in window_names):
                        # Re-create any closed window so imshow works again next iteration.
                        for name in window_names:
                            if cv.getWindowProperty(name, cv.WND_PROP_VISIBLE) < 1:
                                cv.namedWindow(name, cv.WINDOW_NORMAL)
                        break

                objpoints.append(objp)
                imgpoints_left.append(corners1)
                imgpoints_right.append(corners2)

    cv.destroyAllWindows()

    if len(objpoints) == 0:
        raise RuntimeError("No valid checkerboard pairs found for stereo calibration.")

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
        mtx2, dist2, (width, height), criteria=criteria, flags=stereocalibration_flags
    )
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

def load_camera_extrinsics(filename):
    """
        Load the rotation matrix (3x3) and translation matrix (3x1) from a YAML file.
        Parameters:
            filename (str): The path to the YAML file.
        Returns:
            rotation_matrix (np.ndarray): The 3x3 rotation matrix.
            translation_matrix (np.ndarray): The 3x1 translation matrix.
    """
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    extrinsics = data['camera_extrinsics']

    rotation_matrix = np.array(extrinsics['rotation_matrix']).reshape((3, 3))
    translation_matrix = np.array(extrinsics['translation_vector']).reshape((3, 1))

    return rotation_matrix, translation_matrix


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


# Function to save the rotation matrix and translation vector to a YAML file
def save_pose_to_yaml(rotation_matrix, translation_vector, filename,
                       frame_from, frame_to="world",
                       scale_factor=1.0, rms_error=0.0, source_file="soder.txt"):
    """
    Save the rotation matrix (3x3) and translation vector (3,) or (3,1)
    to a YAML file in the current camera_extrinsics format.

    Parameters:
        rotation_matrix (np.ndarray): The 3x3 rotation matrix.
        translation_vector (np.ndarray): The translation vector.
        filename (str): Path to the YAML file to write.
        frame_from (str): Name of the source frame (e.g. "camera_0").
        frame_to (str): Name of the target frame (default "world").
        scale_factor (float): Scale factor, default 1.0.
        rms_error (float): RMS calibration error, default 0.0.
        source_file (str): Reference to the originating calibration file.
    """
    data = {
        "camera_extrinsics": {
            "frame_from": frame_from,
            "frame_to": frame_to,
            "rotation_matrix": np.asarray(rotation_matrix).reshape(3, 3).tolist(),
            "translation_vector": np.asarray(translation_vector).reshape(3).tolist(),
            "scale_factor": float(scale_factor),
            "rms_error": float(rms_error),
            "source_file": source_file,
        }
    }

    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=None, sort_keys=False)

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
    return cameras

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

def load_transformation(file_path):
    """
    Loads the transformation parameters (R, d, s, rms) from a text file.

    Parameters:
    file_path: str
        Path to the file from which the transformation parameters will be read.

    Returns:
    R: ndarray
        Rotation matrix (3x3)
    d: ndarray
        Translation vector (3,)
    s: float
        Scale factor
    rms: float
        Root mean square fit error
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        R_start = lines.index("Rotation Matrix (R):\n") + 1
        R = np.loadtxt(lines[R_start:R_start + 3])
        d_start = lines.index("Translation Vector (d):\n") + 1
        d = np.loadtxt(lines[d_start:d_start + 1]).flatten()
        s_line = next(line for line in lines if line.startswith("Scale Factor (s):"))
        s = float(s_line.split(":")[1].strip())
        rms_line = next(line for line in lines if line.startswith("RMS Error:"))
        rms = float(rms_line.split(":")[1].strip())
    return R, d, s, rms


def save_cam_to_cam_params(mtx0, dist0, mtx2, dist2, R, T, rmse, path):
    """
    Save stereo camera calibration parameters to a file.
    Args:
        mtx0 (numpy.ndarray): Camera matrix for the first camera.
        dist0 (numpy.ndarray): Distortion coefficients for the first camera.
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
    cv_file.write('K0', mtx0)
    cv_file.write('D0', dist0)
    cv_file.write('K2', mtx2)
    cv_file.write('D2', dist2)
    cv_file.write('R', R)
    cv_file.write('T', T)
    cv_file.write('rmse', rmse)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def transform_to_local_frame(D, origin, rotation_matrix):
    # Compute D relative to B
    D_relative = D - origin
    
    # Transform D to the local frame
    D_local = rotation_matrix.T @ D_relative
    
    return D_local

def compose_via_world(R_a_to_w, d_a_to_w, R_b_to_w, d_b_to_w):
    """
    Given two cameras' poses relative to world (a->world, b->world),
    compute a->b directly.
    """
    R_w_to_b = R_b_to_w.T
    d_w_to_b = -R_w_to_b @ d_b_to_w

    R_a_to_b = R_w_to_b @ R_a_to_w
    d_a_to_b = R_w_to_b @ d_a_to_w + d_w_to_b
    return R_a_to_b, d_a_to_b

    # --- Path helpers, shared by calibrate_cameras.py and set_world_frame.py ---

def camera_ids_for(num_cameras):
    """Camera indices follow the c0, c2, c4, c6 naming convention."""
    if num_cameras not in (2, 4):
        raise ValueError("num_cameras must be 2 or 4.")
    return [i * 2 for i in range(num_cameras)]


def intrinsics_path(cam_params_dir, cam_id):
    return os.path.join(cam_params_dir, f"camera_{cam_id}_intrinsics.yaml")


def extrinsics_path(cam_params_dir, cam_id):
    return os.path.join(cam_params_dir, f"camera_{cam_id}_extrinsics.yaml")


def cam_to_cam_path(cam_params_dir, cam_a, cam_b):
    return os.path.join(cam_params_dir, f"camera_{cam_a}_to_camera_{cam_b}.yaml")


def calib_images_dir(repo_path, cam_id):
    """Image folder used by calibrate_cameras.py (checkerboard intrinsics/extrinsics)."""
    return os.path.join(repo_path, f"images_calib_cam_{cam_id}", "color")


def world_images_dir(repo_path, cam_id):
    """Image folder used by set_world_frame.py (ArUco-wand world frame).
    Preserves legacy naming (images_world_cam_1 for cam 0, images_world_cam_2
    for cam 2), extends numerically for cam4/cam6."""
    legacy_names = {0: "images_world_cam_1", 2: "images_world_cam_2"}
    name = legacy_names.get(cam_id, f"images_world_cam_{cam_id}")
    return os.path.join(repo_path, name, "color")


def has_images(img_dir, pattern="*.png"):
    if not os.path.isdir(img_dir):
        return False
    return len(glob.glob(os.path.join(img_dir, pattern))) > 0


# --- Soder world-frame generation, shared by calibrate_cameras.py (--soder mode) ---

def run_soder_world_frame(cam_ids, cam_params_dir):
    """Compute camera_N_extrinsics.yaml from soder{N}.txt for every camera.
    Skips any camera whose soder file is missing, with a warning.
    Returns the list of camera IDs that got a world pose written."""
    world_pose_cam_ids = []
    for cam_id in cam_ids:
        soder_path = os.path.join(cam_params_dir, f"soder{cam_id}.txt")
        if not os.path.isfile(soder_path):
            print(f"[SKIP] camera_{cam_id}: no soder{cam_id}.txt found at {soder_path}")
            continue

        R, d, s, rms = load_transformation(soder_path)
        out_path = extrinsics_path(cam_params_dir, cam_id)

        save_pose_to_yaml(
            R, d, out_path,
            frame_from=f"camera_{cam_id}", frame_to="world",
            scale_factor=s, rms_error=rms, source_file=f"soder{cam_id}.txt",
        )
        print(f"[SODER] Saved {out_path}  (rms={rms:.6f}, scale={s:.6f})")
        world_pose_cam_ids.append(cam_id)

    return world_pose_cam_ids@@ffff