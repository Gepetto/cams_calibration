import cv2 as cv
import yaml
import glob
import os
import shutil
import numpy as np
import subprocess
from utils.settings import Settings

settings = Settings()

# Sub-pixel corner refinement, and how hard to try. Loosened for the stereo
# solve, which needs corners to agree across two views.
_CORNER_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
_STEREO_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)


def checkerboard_grid():
    """The board's inner-corner layout and its corner positions in board space.

    Read from ``utils/settings.py`` -- ``checkerboard_rows``/``columns`` count
    inner corners (squares minus one) and ``checkerboard_scaling`` is the square
    size in metres. Every detection goes through here so the three places that
    use the board cannot disagree about its geometry.

    Returns:
        tuple: ``((rows, columns), objp)`` with ``objp`` of shape (rows*columns, 3).
    """
    rows = settings.checkerboard_rows
    columns = settings.checkerboard_columns
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    return (rows, columns), objp * settings.checkerboard_scaling


def read_image_folder(images_folder):
    """Load every image matching a glob, in sorted order.

    Sorted order is what pairs one camera's shots with another's, so the two
    folders must hold one image per shot under the same names.
    """
    return [cv.imread(name, 1) for name in sorted(glob.glob(images_folder))]


def find_checkerboard(image, size, refine=True):
    """Locate the checkerboard in one image, refined to sub-pixel accuracy.

    Returns:
        np.ndarray | None: the corners, or None when the board is not visible.
    """
    if image is None:
        return None
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, size, None)
    if not found:
        return None
    if refine:
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), _CORNER_CRITERIA)
    return corners


def calibrate_intrinsics(images_folder, show=True):
    """
    Calibrate one camera's intrinsics from checkerboard images.

    Args:
        images_folder (str): glob matching that camera's checkerboard images.
        show (bool): draw each detection in a window. Turn off to run without a
            display, e.g. recalibrating a recorded session over SSH or in CI.

    Returns:
        tuple: ``(rmse, mtx, dist)`` -- reprojection error, camera matrix and
        distortion coefficients.
    """
    images = read_image_folder(images_folder)
    size, objp = checkerboard_grid()

    objpoints, imgpoints = [], []
    for frame in images:
        corners = find_checkerboard(frame, size)
        if corners is None:
            continue
        if show:
            cv.drawChessboardCorners(frame, size, corners, True)
            cv.imshow('img', frame)
            cv.waitKey(50)
        objpoints.append(objp)
        imgpoints.append(corners)

    height, width = images[0].shape[:2]
    rmse, mtx, dist, _, _ = cv.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None)
    cv.destroyAllWindows()
    return rmse, mtx, dist


def corners_consistently_ordered(corners1, corners2):
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

def calibrate_stereo_pair(mtx1, dist1, mtx2, dist2, frames_folder_1, frames_folder_2,
                     show=True):
    """
    Solve the pose of one camera relative to another, from shared board views.

    The intrinsics are held fixed (``CALIB_FIX_INTRINSIC``), so this solves only
    the relative pose -- run :func:`calibrate_intrinsics` on each camera first. Only
    shots where *both* cameras see the board contribute, which is what lets a
    ring of cameras be calibrated pair by pair without the board ever being
    visible to all of them at once.

    Args:
        mtx1, dist1: first camera's intrinsics.
        mtx2, dist2: second camera's intrinsics.
        frames_folder_1, frames_folder_2 (str): globs for the two cameras'
            images. Sorted order pairs them, so shot *n* must be named alike.
        show (bool): draw each accepted pair. Turn off to run without a display.

    Returns:
        tuple: ``(rmse, R, T)`` in OpenCV's convention, ``p_2 = R @ p_1 + T``.
    """
    c1_images = read_image_folder(frames_folder_1)
    c2_images = read_image_folder(frames_folder_2)
    size, objp = checkerboard_grid()

    objpoints, imgpoints_left, imgpoints_right = [], [], []
    for frame1, frame2 in zip(c1_images, c2_images):
        corners1 = find_checkerboard(frame1, size)
        corners2 = find_checkerboard(frame2, size)
        if corners1 is None or corners2 is None:
            continue
        if not corners_consistently_ordered(corners1, corners2):
            continue

        if show:
            # waitKey(0) blocks until a key is pressed, which is what the
            # operator wants when checking detections by eye, but hangs forever
            # with no display or keyboard attached.
            cv.drawChessboardCorners(frame1, size, corners1, True)
            cv.imshow('img', frame1)
            cv.drawChessboardCorners(frame2, size, corners2, True)
            cv.imshow('img2', frame2)
            cv.waitKey(0)

        objpoints.append(objp)
        imgpoints_left.append(corners1)
        imgpoints_right.append(corners2)

    height, width = c1_images[0].shape[:2]
    rmse, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx1, dist1, mtx2, dist2,
        (width, height), criteria=_STEREO_CRITERIA, flags=cv.CALIB_FIX_INTRINSIC)
    cv.destroyAllWindows()
    return rmse, R, T


def write_intrinsics_file(mtx, dist, reproj, path):
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

def load_intrinsics(path):
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

def write_stereo_file(mtx1, dist1, mtx2, dist2, R, T, rmse, path):
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

def load_stereo_pose(path):
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
def wand_positions_in_camera(images_folder, camera_matrix, dist_coeffs, detector,
                             marker_size, expected):
    """Where the wand tip was, in camera coordinates, for each pointed position.

    One image per position, each showing the wand's aruco marker. The marker pose
    gives the wand's pose; the tip is a fixed offset from it
    (``settings.wand_end_effector_local_pos``).

    Args:
        images_folder (str): glob matching the images, one per pointed position.
        camera_matrix, dist_coeffs: that camera's intrinsics.
        detector: an ``cv.aruco.ArucoDetector``.
        marker_size (float): marker side length in metres.
        expected (int): how many positions this frame definition needs.

    Returns:
        list: one (3,) position per image, in camera coordinates.
    """
    images_names = sorted(glob.glob(images_folder))
    images = [cv.imread(name, 1) for name in images_names]
    assert len(images) == expected, \
        f"need exactly {expected} images to define this frame, found {len(images)}"

    wand_local = settings.wand_end_effector_local_pos
    half = marker_size / 2
    marker_points = np.array([[-half, half, 0], [half, half, 0],
                              [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

    positions = []
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(corners) == 0:
            continue
        _, rvec, tvec = cv.solvePnP(marker_points, corners[0].reshape(-1, 2),
                                    camera_matrix, dist_coeffs, False,
                                    cv.SOLVEPNP_IPPE_SQUARE)
        rotation, _ = cv.Rodrigues(rvec)
        positions.append((tvec + rotation @ wand_local).flatten())
    return positions


def frame_from_directions(origin, x_direction, in_plane_direction):
    """Build a right-handed frame at ``origin`` from two directions.

    ``x_direction`` becomes the x axis; ``in_plane_direction`` only has to be
    non-collinear with it, and fixes the remaining rotation about x. Both are
    vectors *from* the origin, given in whatever frame the points were measured
    in -- passing them as directions rather than as points keeps which way each
    axis runs explicit, since the two world frames differ precisely in that.

    Returns:
        tuple: ``(origin, R)`` where ``R``'s columns are the new frame's axes
        expressed in the measured frame, so ``R`` maps new-frame coordinates into
        the measured one.
    """
    Vx = np.asarray(x_direction, dtype=float)
    Vz = np.cross(Vx, np.asarray(in_plane_direction, dtype=float))
    Vy = np.cross(Vz, Vx)
    axes = [V / np.linalg.norm(V) for V in (Vx, Vy, Vz)]
    return origin, np.column_stack(axes)


def robot_frame_in_camera(images_folder, camera_matrix, dist_coeffs, detector,
                                   marker_size):
    """The robot base frame, expressed in camera coordinates.

    Four pointed positions: two pairs, each straddling an axis of the base, so
    the frame origin is the midpoint of the first pair and x runs towards the
    midpoint of the second.
    """
    P = wand_positions_in_camera(images_folder, camera_matrix, dist_coeffs, detector,
                                 marker_size, expected=4)
    origin = (P[0] + P[1]) / 2
    # x runs from the second pair's midpoint back towards the origin, which is
    # the opposite sense to the ground frame below.
    return frame_from_directions(origin, origin - (P[2] + P[3]) / 2, P[0] - origin)


def ground_frame_in_camera(images_folder, camera_matrix, dist_coeffs, detector,
                                   marker_size):
    """A frame marked on the ground, expressed in camera coordinates.

    Three pointed positions: the origin, a point along x, and a third fixing the
    plane.
    """
    P = wand_positions_in_camera(images_folder, camera_matrix, dist_coeffs, detector,
                                 marker_size, expected=3)
    return frame_from_directions(P[0], P[1] - P[0], P[2] - P[0])


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

# ---------------------------------------------------------------------------
# COMFI layout -- the calibration format RT-COSMIK reads
# ---------------------------------------------------------------------------
#
#   <root>/intrinsics/camera_<i>_intrinsics.yaml
#   <root>/extrinsics/cam_to_cam/camera_<a>_to_camera_<b>.yaml
#   <root>/extrinsics/cam_to_world/camera_<i>/camera_<i>_extrinsics.yaml
#
# Intrinsics and stereo results are OpenCV FileStorage documents, written in
# OpenCV's own convention. World poses are plain YAML and hold the pose of the
# *camera in the world frame* (p_world = R @ p_cam + T), which is the opposite
# of what solvePnP gives, so the conversion happens here rather than in every
# consumer.


def intrinsics_path(root, camera_id):
    """Path to one camera's intrinsics in the COMFI layout."""
    return os.path.join(root, "intrinsics", f"camera_{camera_id}_intrinsics.yaml")


def stereo_pose_path(root, cam_a, cam_b):
    """Path to a stereo result in the COMFI layout."""
    return os.path.join(root, "extrinsics", "cam_to_cam",
                        f"camera_{cam_a}_to_camera_{cam_b}.yaml")


def world_pose_path(root, camera_id):
    """Path to one camera's world pose in the COMFI layout."""
    return os.path.join(root, "extrinsics", "cam_to_world",
                        f"camera_{camera_id}", f"camera_{camera_id}_extrinsics.yaml")


def _ensure_parent(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Which physical camera is which
# ---------------------------------------------------------------------------
#
# The ids in the layout (camera_0, camera_2, ...) are v4l2 indices, which record
# the order the kernel happened to enumerate devices in. They are not identity:
# they can change on reboot or when a camera is replugged. If they change and
# nothing checks, RT-COSMIK silently applies one camera's intrinsics and pose to
# a different physical camera, which produces a plausible-looking but wrong
# reconstruction rather than an error.
#
# So the hardware behind each id is recorded at calibration time and checked
# before use. The USB port path is the discriminator: identical cameras of the
# same model usually share a placeholder serial (this rig's report "SN0001"), so
# a serial cannot tell two of them apart, while the port path always can.


def camera_manifest_path(root):
    """Path to the record of which physical camera each id refers to."""
    return os.path.join(root, "cameras.yaml")


def camera_hardware_info(index):
    """Describe the hardware behind one v4l2 index.

    Returns a dict with whatever could be read: ``bus_info`` (the USB port path,
    the only field that reliably distinguishes identical cameras), ``model``,
    ``serial``, and ``vendor_product``. Missing fields are omitted rather than
    guessed.
    """
    info = {}
    sysfs = f"/sys/class/video4linux/video{index}"
    name_path = os.path.join(sysfs, "name")
    if os.path.isfile(name_path):
        with open(name_path) as handle:
            info["model"] = handle.read().strip()

    device = os.path.join(sysfs, "device")
    if os.path.islink(device) or os.path.isdir(device):
        usb_root = os.path.realpath(os.path.join(device, ".."))
        for field, key in (("serial", "serial"), ("product", "product"),
                           ("manufacturer", "manufacturer")):
            path = os.path.join(usb_root, field)
            if os.path.isfile(path):
                with open(path) as handle:
                    info[key] = handle.read().strip()
        vendor = os.path.join(usb_root, "idVendor")
        product = os.path.join(usb_root, "idProduct")
        if os.path.isfile(vendor) and os.path.isfile(product):
            with open(vendor) as v, open(product) as p:
                info["vendor_product"] = f"{v.read().strip()}:{p.read().strip()}"

    # bus_info as v4l2 reports it, e.g. usb-0000:00:14.0-8.1
    try:
        output = subprocess.check_output(
            ["v4l2-ctl", "-d", f"/dev/video{index}", "--info"],
            stderr=subprocess.DEVNULL).decode()
        for line in output.splitlines():
            if "Bus info" in line:
                info["bus_info"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return info


def save_camera_manifest(root, camera_ids, labels=None):
    """Record which physical camera each id refers to, next to the calibration.

    Args:
        root (str): calibration root to write into.
        camera_ids (Sequence[int]): the ids being calibrated.
        labels (dict | None): optional ``{camera_id: human name}``, e.g.
            ``{0: "front_left"}``, for talking about cameras without indices.
    """
    labels = labels or {}
    entries = []
    for camera_id in camera_ids:
        entry = {"id": int(camera_id)}
        if camera_id in labels:
            entry["label"] = labels[camera_id]
        entry.update(camera_hardware_info(camera_id))
        entries.append(entry)

    path = _ensure_parent(camera_manifest_path(root))
    with open(path, "w") as handle:
        yaml.safe_dump({"cameras": entries}, handle,
                       default_flow_style=False, sort_keys=False)
    return path


def count_shared_checkerboard_views(frames_folder_1, frames_folder_2):
    """How many image pairs show the checkerboard to *both* cameras.

    Stereo calibration uses only those shots, so with more than two cameras this
    is the number that matters: adjacent cameras in a ring share less and less of
    their view, and a pair can end up with too few shared shots to calibrate
    while each camera's own intrinsics still look fine. Counting first turns that
    into a clear message instead of a confusing failure inside cv.stereoCalibrate.

    Returns:
        tuple: ``(shared, total)``.
    """
    images_1 = read_image_folder(frames_folder_1)
    images_2 = read_image_folder(frames_folder_2)
    size, _ = checkerboard_grid()

    # Corner refinement does not change whether the board was found, and this
    # runs over every shot before calibrating, so skip it here.
    shared = sum(1 for f1, f2 in zip(images_1, images_2)
                 if find_checkerboard(f1, size, refine=False) is not None
                 and find_checkerboard(f2, size, refine=False) is not None)
    return shared, min(len(images_1), len(images_2))


def save_intrinsics(mtx, dist, reproj, camera_id, root):
    """Write one camera's intrinsics into the COMFI layout."""
    path = _ensure_parent(intrinsics_path(root, camera_id))
    write_intrinsics_file(mtx, dist, reproj, path)
    return path


def save_stereo_pose(mtx1, dist1, mtx2, dist2, R, T, rmse, cam_a, cam_b, root):
    """Write a stereo result into the COMFI layout.

    ``R``/``T`` are stored exactly as ``cv2.stereoCalibrate`` returns them
    (``p_b = R @ p_a + T``): they relate two cameras, with no world involved, so
    no conversion applies.
    """
    path = _ensure_parent(stereo_pose_path(root, cam_a, cam_b))
    write_stereo_file(mtx1, dist1, mtx2, dist2, R, T, rmse, path)
    return path


def invert_pose(R, T):
    """Swap a pose between the two directions.

    Given ``p_b = R @ p_a + T`` returns the pair for ``p_a = R' @ p_b + T'``.
    Useful because ``cv2.solvePnP`` and the ``get_relative_pose_*_in_cam``
    helpers return the world expressed *in camera* coordinates, while RT-COSMIK
    stores the camera expressed *in world* coordinates.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    T = np.asarray(T, dtype=float).reshape(3)
    return R.T, -R.T @ T


def save_world_pose(world_R_cam, world_T_cam, camera_id, root):
    """Write a camera's world pose into the COMFI layout.

    Expects the pose already in RT-COSMIK's convention: the pose of the *camera
    in the world frame*, so ``world_R_cam`` is the camera's orientation in world
    coordinates and ``world_T_cam`` its position in world coordinates, giving
    ``p_world = R @ p_cam + T``. Nothing is inverted here -- a function that
    silently flipped the direction would be impossible to reason about when the
    caller already holds the right one.

    The ``get_relative_pose_*_in_cam`` helpers return the *opposite* direction
    (world expressed in camera coordinates), so their output must be passed
    through :func:`invert_pose` first. As a sanity check, ``world_T_cam`` is the
    camera's physical position in the room: if it comes out near the origin, the
    pose is the wrong way round.

    Args:
        world_R_cam: camera orientation in world coordinates (3x3).
        world_T_cam: camera position in world coordinates (3,), metres.
        camera_id (int): camera this pose belongs to.
        root (str): calibration root to write into.
    """
    R = np.asarray(world_R_cam, dtype=float).reshape(3, 3)
    T = np.asarray(world_T_cam, dtype=float).reshape(3)

    path = _ensure_parent(world_pose_path(root, camera_id))
    data = {
        "camera_extrinsics": {
            "frame_from": f"camera_{camera_id}",
            "frame_to": "world",
            "rotation_matrix": [[float(v) for v in row] for row in R],
            "translation_vector": [float(v) for v in T],
        }
    }
    with open(path, "w") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
    return path


def chain_stereo_poses(root, camera_ids):
    """Chain the stereo results for consecutive cameras into relative poses.

    Returns ``{camera_id: (R, T)}`` with ``p_cam = R @ p_ref + T``, where the
    reference is ``camera_ids[0]``. This walks the consecutive pairs this repo
    writes; RT-COSMIK carries a more general version that treats the pairs as a
    graph and can take them in any order or direction.
    """
    camera_ids = list(camera_ids)
    poses = {camera_ids[0]: (np.eye(3), np.zeros(3))}
    for cam_a, cam_b in zip(camera_ids[:-1], camera_ids[1:]):
        path = stereo_pose_path(root, cam_a, cam_b)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"missing stereo result for camera_{cam_a} -> camera_{cam_b}: {path}")
        R, T = load_stereo_pose(path)
        if R is None or T is None:
            raise ValueError(f"missing 'R'/'T' in {path}")
        R = np.asarray(R, dtype=float).reshape(3, 3)
        T = np.asarray(T, dtype=float).reshape(3)
        R_prev, T_prev = poses[cam_a]
        poses[cam_b] = (R @ R_prev, R @ T_prev + T)
    return poses


def average_rotations(rotations):
    """Mean of several rotation matrices, as the nearest rotation to their sum.

    Averaging rotation matrices elementwise leaves a matrix that is no longer
    orthonormal, so the result is projected back onto SO(3) by SVD. For the small
    spreads seen between views of the same physical frame this is equivalent to
    the quaternion mean and needs no sign bookkeeping.
    """
    total = np.sum([np.asarray(R, dtype=float).reshape(3, 3) for R in rotations], axis=0)
    U, _, Vt = np.linalg.svd(total)
    mean = U @ Vt
    if np.linalg.det(mean) < 0:  # guard against a reflection
        U[:, -1] *= -1
        mean = U @ Vt
    return mean


def fuse_world_anchor(world_poses, relative_poses, reference,
                      wand_rot_deg=4.0, wand_pos_mm=10.0):
    """Combine every camera's world measurement into one anchor for the reference.

    Each camera that sees the world frame gives an independent, noisy measurement
    of it. Because the stereo chain relates the cameras accurately, each of those
    measurements can be re-expressed as an estimate of the *reference* camera's
    world pose, so all of them can be used instead of trusting whichever camera
    happens to be first.

    For camera ``i`` with world pose ``(Rw_i, Tw_i)`` and chain pose ``(R_i, T_i)``
    taking reference-frame points into camera ``i`` (``p_i = R_i p_ref + T_i``)::

        p_world = Rw_i (R_i p_ref + T_i) + Tw_i
                = (Rw_i R_i) p_ref + (Rw_i T_i + Tw_i)

    Rotation and translation are then combined differently, because they do not
    degrade the same way. A camera's rotation estimate is unaffected by how far
    it sits from the reference, so rotations are averaged evenly and the error
    falls with the square root of the number of cameras. A camera's *position*
    estimate, though, is carried across the baseline by that camera's own
    rotation error: a few degrees over a multi-metre baseline is hundreds of
    millimetres. Averaging positions evenly therefore makes the anchor markedly
    worse than simply believing the reference camera. Positions are instead
    combined by inverse variance, with each camera's variance being its own wand
    error plus the rotation error amplified by its distance from the reference::

        var_i = wand_pos^2 + (|T_i| * wand_rot)^2

    The reference camera has ``|T_i| = 0`` and so dominates, while a distant
    camera contributes only as much as its geometry allows.

    Args:
        world_poses (dict): ``{camera_id: (R, T)}`` measured world poses, in
            RT-COSMIK's convention (camera in world).
        relative_poses (dict): ``{camera_id: (R, T)}`` chain poses relative to
            ``reference``.
        reference (int): camera the anchor is expressed for.
        wand_rot_deg (float): expected rotational error of one wand measurement.
        wand_pos_mm (float): expected positional error of one wand measurement.

    Returns:
        tuple: ``(R, T, spread)`` -- the fused pose of the reference camera in the
        world, and a ``{camera_id: (angle_deg, distance_m)}`` mapping of how far
        each camera's own estimate sits from the fused one. A large spread means
        the world measurements disagree, which is the signal that one of them is
        badly pointed.
    """
    estimates = {}
    baselines = {}
    for camera_id, (Rw, Tw) in world_poses.items():
        if camera_id not in relative_poses:
            continue
        Rw = np.asarray(Rw, dtype=float).reshape(3, 3)
        Tw = np.asarray(Tw, dtype=float).reshape(3)
        R_rel, T_rel = relative_poses[camera_id]
        R_rel = np.asarray(R_rel, dtype=float).reshape(3, 3)
        T_rel = np.asarray(T_rel, dtype=float).reshape(3)
        estimates[camera_id] = (Rw @ R_rel, Rw @ T_rel + Tw)
        baselines[camera_id] = float(np.linalg.norm(T_rel))

    if not estimates:
        raise ValueError("no camera has both a world pose and a chain pose")

    R_mean = average_rotations([R for R, _ in estimates.values()])

    sigma_pos = wand_pos_mm / 1000.0
    sigma_rot = np.radians(wand_rot_deg)
    weights = np.array([1.0 / (sigma_pos ** 2 + (baselines[c] * sigma_rot) ** 2)
                        for c in estimates])
    weights /= weights.sum()
    T_mean = np.sum([w * T for w, (_, T) in zip(weights, estimates.values())], axis=0)

    spread = {}
    for camera_id, (R, T) in estimates.items():
        cos = (np.trace(R @ R_mean.T) - 1.0) / 2.0
        spread[camera_id] = (float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
                             float(np.linalg.norm(T - T_mean)))
    return R_mean, T_mean, spread


def rtcosmik_calib_path():
    """Where RT-COSMIK reads calibration from, or None if it is not installed."""
    try:
        from rtcosmik.config_loader import settings
    except ImportError:
        return None
    return settings.cam_calib_path


def install_to_rtcosmik(source, destination=None, verify=True):
    """Copy a COMFI-layout calibration into the place RT-COSMIK reads it from.

    Nothing is converted: the calibration scripts already write the layout and
    convention RT-COSMIK expects, so this is a copy plus a read-back. The
    read-back matters -- a calibration that lands in the right directory but
    cannot be loaded is not installed, and the failure would otherwise only
    surface at the start of a capture session.

    Args:
        source (str): calibration root to copy from.
        destination (str | None): root to copy into; defaults to RT-COSMIK's
            ``settings.cam_calib_path``.
        verify (bool): load the result back through RT-COSMIK.

    Returns:
        str: the destination written to.
    """
    destination = destination or rtcosmik_calib_path()
    if destination is None:
        raise RuntimeError(
            "RT-COSMIK is not importable, so its calibration path is unknown. "
            "Pass an explicit destination, or install RT-COSMIK with "
            "rt-cosmik/scripts/bash/setup_env.sh.")
    if os.path.abspath(source) == os.path.abspath(destination):
        return destination

    copied = 0
    for sub in ("intrinsics", "extrinsics"):
        src = os.path.join(source, sub)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(destination, sub)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied += sum(len(files) for _, _, files in os.walk(dst))

    # The manifest travels with the calibration: without it RT-COSMIK has no way
    # to tell that the rig was recabled since these files were produced.
    manifest = camera_manifest_path(source)
    if os.path.isfile(manifest):
        shutil.copyfile(manifest, camera_manifest_path(destination))
        copied += 1
    print(f"  installed {copied} files into {destination}")

    if not verify:
        return destination
    try:
        from rtcosmik.camera.cam_utils import (load_camera_parameters,
                                               describe_camera_placement)
    except ImportError:
        print("  RT-COSMIK not importable, skipping verification")
        return destination

    intrinsics_dir = os.path.join(destination, "intrinsics")
    cameras = sorted(
        int(name[len("camera_"):-len("_intrinsics.yaml")])
        for name in os.listdir(intrinsics_dir)
        if name.startswith("camera_") and name.endswith("_intrinsics.yaml"))
    mtxs, _, _, _, _ = load_camera_parameters(destination, camera_ids=cameras)
    print(f"  verified: RT-COSMIK loads {len(mtxs)} camera(s) {cameras}")
    for camera_id, position in describe_camera_placement(
            destination, camera_ids=cameras).items():
        print(f"    camera_{camera_id} anchored at {position.round(3)} m")
    return destination
