"""Rough camera-to-camera calibration from a person moving in the scene.

A side option, and a much less accurate one than the checkerboard: it places
cameras about 3% of their baseline out. Read the numbers below before choosing
it over calibrate_cameras.py.

A metric monocular pose estimator (RT-COSMIK uses NLF) regresses a 3D skeleton
in each camera's own frame, so relating two cameras needs no shared checkerboard
view -- only the same person seen at the same instant, from any two angles. That
is the point of this path: adjacent cameras in a ring overlap less the more
cameras there are, and opposed pairs may never share a usable board pose, which
is exactly where the checkerboard route runs out.

What it costs, measured against motion capture on 8 COMFI participants,
24 camera pairs:

    rotation        0.97 deg median   (0.69 - 1.72)
    baseline dir    0.84 deg median   (0.53 - 1.40)
    translation     2.8 - 3.1% of the baseline: ~25 mm at 0.8 m, ~150 mm at 5 m

against roughly 5 mm for a checkerboard on a comparable pair. That gap is not an
implementation shortfall: Lee et al. (RA-L 2022) report the same magnitude for
this class of method. The ceiling is the pose estimator's view-dependent
keypoint bias, ~7 px, which no amount of optimisation removes.

Whether that matters depends on what consumes the calibration. Reconstructing
the same recording with these extrinsics instead of the mocap ones moved every
point by 64 mm -- but removing a single rigid transform left 0.2 mm, and segment
lengths agreed to 0.01%. The error is a ~0.5 deg rotation of the whole scene,
not a deformation, because RT-COSMIK fuses per-camera monocular skeletons and so
takes the body's shape from the estimator, never from the calibration. Joint
angles came out identical against mocap (10.48 deg either way); what degrades is
absolute placement in the room.

A pipeline that triangulates 2D instead has no such protection: the same
extrinsics left 12.6 mm of real shape distortion and 1.33% on segment lengths.
Do not use this path for one.

Three things were tried and did not help, so they are deliberately absent:
bundle adjustment (free-point or bone-length constrained) drifts *away* from the
truth because it can absorb the estimator's view-dependent 2D bias into camera
pose; averaging over the full pair graph is worse than the direct links, since
the extra edges join distant cameras with poor overlap; and dropping
low-confidence or image-border keypoints changes nothing, because the bias is a
coherent shift of the whole skeleton rather than a few bad joints.
"""
import numpy as np
import cv2 as cv

#: Below this, the pooled joint cloud is too flat to pin the geometry down. A
#: person walking a straight line keeps every joint in a slab, which is a known
#: degeneracy for essential-matrix estimation; standing still is worse. Measured
#: on COMFI: circuits of the floor reach 0.49-0.54 and give ~1 deg, while every
#: stationary or straight-line trial sat at 0.08-0.19 and gave 2.3-5.1 deg.
MIN_CLOUD_THICKNESS = 0.35

#: Head landmark height above the floor, as a fraction of stature. The estimator
#: reports a head *centre*, not the top of the skull, so stature needs this to
#: become a distance between reconstructed points. Mean over 8 COMFI
#: participants (1.64-1.87 m), sd 1.5%, which is ~54 mm of baseline error at
#: 3.5 m -- the dominant error term, so treat a scale-critical rig with care.
STATURE_RATIO = 0.9515

#: Landmarks touching the floor, used to fit the ground plane. Taken over the
#: whole recording, this is steadier than the heel markers in any single frame
#: (sd 1.5% vs 1.7% for a heel-midpoint stature).
FOOT_LANDMARKS = ("RHEE", "LHEE", "RTOE", "LTOE", "R5MHD", "L5MHD")


def cloud_thickness(poses3d):
    """Singular values of the pooled joint cloud, in metres, largest first.

    The third one is what matters: it is how far the subject's body departs from
    a plane over the whole recording, and it predicts the calibration error far
    better than the number of frames does.

    Args:
        poses3d: (F, J, 3) metric poses from one camera, NaN where unseen.

    Returns:
        np.ndarray: three singular values, scaled so they read as metres.
    """
    pts = np.asarray(poses3d, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 4:
        return np.zeros(3)
    return np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False) / np.sqrt(len(pts))


def _normalise(points, mtx, dist):
    """Pixels to normalised image coordinates, undistorted."""
    p = cv.undistortPoints(np.asarray(points, np.float64).reshape(-1, 1, 2), mtx, dist)
    return p.reshape(-1, 2)


def register_pair(keypoints_a, keypoints_b, poses3d_a, poses3d_b,
                  mtx_a, dist_a, mtx_b, dist_b, threshold_px=1.0):
    """Relative pose between two cameras, as ``p_b = R @ p_a + T``.

    Rotation and the baseline *direction* come from the essential matrix on the
    2D keypoints, which are direct image measurements and so carry none of the
    estimator's depth error. Only the baseline *magnitude* is taken from the
    metric 3D, and it is a single scalar fitted over every frame at once.

    Splitting it this way matters: registering the two 3D skeletons rigidly
    instead gives a visibly worse rotation, because the estimator's per-camera
    depth is biased and a rigid fit absorbs that bias into the pose.

    Args:
        keypoints_a, keypoints_b: (F, J, 2) pixel keypoints, NaN where unseen.
        poses3d_a, poses3d_b: (F, J, 3) metric poses in each camera's own frame.
        mtx_a, dist_a, mtx_b, dist_b: the two cameras' intrinsics.
        threshold_px: RANSAC inlier threshold for the essential matrix.

    Returns:
        dict: ``R``, ``T``, ``inlier_frac`` and ``frames``, or None if too few
        frames showed the subject to both cameras.
    """
    keypoints_a = np.asarray(keypoints_a, float)
    keypoints_b = np.asarray(keypoints_b, float)
    poses3d_a = np.asarray(poses3d_a, float)
    poses3d_b = np.asarray(poses3d_b, float)

    seen = (np.isfinite(keypoints_a).all(axis=2) & np.isfinite(keypoints_b).all(axis=2))
    if seen.sum() < 8:
        return None

    flat_a = keypoints_a.reshape(-1, 2)[seen.ravel()]
    flat_b = keypoints_b.reshape(-1, 2)[seen.ravel()]
    norm_a = _normalise(flat_a, mtx_a, dist_a)
    norm_b = _normalise(flat_b, mtx_b, dist_b)

    focal = 0.5 * (mtx_a[0, 0] + mtx_b[0, 0])
    essential, mask = cv.findEssentialMat(
        norm_a, norm_b, np.eye(3), method=cv.USAC_MAGSAC, prob=0.9999,
        threshold=threshold_px / focal)
    if essential is None:
        return None
    _, rotation, direction, mask = cv.recoverPose(
        essential, norm_a, norm_b, np.eye(3), mask=mask)
    direction = direction.reshape(3)

    # With R and the direction fixed, the magnitude is a one-dimensional fit:
    # b = R a + s * d, so s is the median of (b - R a) projected on d. The
    # median rather than the mean because a mis-detected frame displaces the
    # whole skeleton, and there is no reason to let it drag the baseline.
    solid = seen & np.isfinite(poses3d_a).all(axis=2) & np.isfinite(poses3d_b).all(axis=2)
    if not solid.any():
        return None
    residual = (poses3d_b.reshape(-1, 3)[solid.ravel()]
                - poses3d_a.reshape(-1, 3)[solid.ravel()] @ rotation.T)
    magnitude = float(np.median(residual @ direction))

    return {"R": rotation, "T": magnitude * direction,
            "inlier_frac": float(np.asarray(mask).astype(bool).mean()),
            "frames": int(seen.any(axis=1).sum())}


def triangulate(keypoints, poses, intrinsics, distortions):
    """DLT every joint from all cameras that saw it, in the reference frame.

    Args:
        keypoints: (C, F, J, 2) pixel keypoints per camera, NaN where unseen.
        poses: per-camera ``(R, T)`` taking reference-frame points into that
            camera, with the reference camera at identity.
        intrinsics, distortions: per-camera intrinsics.

    Returns:
        np.ndarray: (F, J, 3) points in the reference camera's frame, NaN where
        fewer than two cameras contributed.
    """
    num_cams, num_frames, num_joints = np.shape(keypoints)[:3]
    obs = np.full((num_frames * num_joints, num_cams, 2), np.nan)
    for cam in range(num_cams):
        flat = np.asarray(keypoints[cam], float).reshape(-1, 2)
        seen = np.isfinite(flat).all(axis=1)
        obs[seen, cam] = _normalise(flat[seen], intrinsics[cam], distortions[cam])
    valid = np.isfinite(obs).all(axis=2)

    projections = [np.hstack([np.asarray(R, float).reshape(3, 3),
                              np.asarray(T, float).reshape(3, 1)]) for R, T in poses]
    points = np.full((num_frames * num_joints, 3), np.nan)
    for idx in range(len(points)):
        rows = []
        for cam in range(num_cams):
            if not valid[idx, cam]:
                continue
            x, y = obs[idx, cam]
            rows.append(x * projections[cam][2] - projections[cam][0])
            rows.append(y * projections[cam][2] - projections[cam][1])
        if len(rows) < 4:
            continue
        _, _, vt = np.linalg.svd(np.asarray(rows))
        if abs(vt[-1, 3]) > 1e-12:
            points[idx] = vt[-1, :3] / vt[-1, 3]
    return points.reshape(num_frames, num_joints, 3)


def stature_scale(points, marker_names, subject_height, ratio=STATURE_RATIO):
    """Metric scale for a reconstruction, from the subject's measured height.

    The reconstruction from 2D alone is only defined up to one global scale.
    Stature fixes it, and unlike the estimator's own metric output it is not
    systematically biased -- NLF's skeletons measured 3.4% oversized on the
    participant this was first checked against.

    Height is taken above a plane fitted to every foot landmark over the whole
    recording, which is steadier than any single frame's heel positions.

    Args:
        points: (F, J, 3) reconstruction in arbitrary units.
        marker_names: the J landmark names, positionally matching ``points``.
        subject_height: the person's stature in metres.
        ratio: head-landmark height as a fraction of stature.

    Returns:
        float: multiply the reconstruction and the baselines by this.

    Raises:
        ValueError: the landmarks needed are missing or never reconstructed.
    """
    names = list(marker_names)
    missing = [n for n in ("Head",) + FOOT_LANDMARKS if n not in names]
    if missing:
        raise ValueError(f"landmarks needed for scaling are missing: {missing}")

    feet = np.asarray(points, float)[:, [names.index(n) for n in FOOT_LANDMARKS]]
    feet = feet.reshape(-1, 3)
    feet = feet[np.isfinite(feet).all(axis=1)]
    if len(feet) < 8:
        raise ValueError("not enough reconstructed foot landmarks to fit a floor")
    centre = feet.mean(axis=0)
    normal = np.linalg.svd(feet - centre)[2][-1]

    head = np.asarray(points, float)[:, names.index("Head")]
    height = np.abs((head - centre) @ normal)
    height = np.nanmedian(height[np.isfinite(height)])
    if not np.isfinite(height) or height <= 0:
        raise ValueError("could not measure the subject in the reconstruction")
    return float(subject_height * ratio / height)


def describe_conditioning(singular_values):
    """One line on whether a recording is shaped well enough to calibrate on."""
    thickness = float(singular_values[2])
    if thickness >= MIN_CLOUD_THICKNESS:
        return True, f"joint cloud {np.round(singular_values, 2).tolist()} m, usable"
    return False, (
        f"joint cloud {np.round(singular_values, 2).tolist()} m: the subject stayed "
        f"too close to one plane (need a third axis of at least "
        f"{MIN_CLOUD_THICKNESS} m).\n"
        f"  Record the subject walking a circuit that covers the floor area, not "
        f"standing in one place and not walking a straight line.")
