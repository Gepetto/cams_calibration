# tests/test_calib_pipeline.py
#
# Run with: pytest tests/test_calib_pipeline.py -v
#
# Tests the calibration pipeline against reference files, using whichever
# input source is actually available on disk:
#   - soder.txt (mocap-based) -> load_transformation -> camera_N_extrinsics.yaml
#   - checkerboard images -> calibrate_camera/stereo_calibrate -> camera_0_to_camera_N.yaml
#
# Each test auto-skips if its required input files aren't present, so this
# suite can run against whatever calibration data you actually have without
# needing both soder AND image data simultaneously.

import os
import glob
import numpy as np
import pytest

from utils.calib_utils import (
    load_transformation,
    save_pose_to_yaml,
    load_camera_extrinsics,
    compose_via_world,
    save_cam_to_cam_params,
    load_cam_to_cam_params,
    load_cam_params,
    calibrate_camera,
    stereo_calibrate,
)


# ---------------------------------------------------------------------------
# Config: point these at your actual calibration directory
# ---------------------------------------------------------------------------
CAM_PARAMS_DIR = os.environ.get(
    "CAM_PARAMS_DIR", "/root/workspace/cams_calibration/config/cam_params"
)
IMAGES_DIR_C0 = os.environ.get(
    "IMAGES_DIR_C0", "/root/workspace/cams_calibration/images_calib_cam_0/color"
)
IMAGES_DIR_C2 = os.environ.get(
    "IMAGES_DIR_C2", "/root/workspace/cams_calibration/images_calib_cam_2/color"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rotation_angle_diff_deg(R1, R2):
    R_diff = R1 @ R2.T
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def translation_diff_m(t1, t2):
    return np.linalg.norm(np.asarray(t1).flatten() - np.asarray(t2).flatten())


def assert_poses_close(R1, t1, R2, t2, label="", rot_tol=0.5, trans_tol=1e-2):
    angle_diff = rotation_angle_diff_deg(R1, R2)
    trans_diff = translation_diff_m(t1, t2)
    assert angle_diff <= rot_tol, (
        f"{label}: rotation differs by {angle_diff:.4f} deg (tolerance {rot_tol} deg)"
    )
    assert trans_diff <= trans_tol, (
        f"{label}: translation differs by {trans_diff:.6f} m (tolerance {trans_tol} m)"
    )


def skip_if_missing(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"Required input not found on disk: {p}")


def skip_if_no_images(folder):
    if not os.path.isdir(folder) or len(glob.glob(os.path.join(folder, "*.png"))) == 0:
        pytest.skip(f"No calibration images found in: {folder}")


def is_valid_rotation(R, atol=1e-3):
    should_be_identity = R @ R.T
    return np.allclose(should_be_identity, np.eye(3), atol=atol) and \
        abs(np.linalg.det(R) - 1.0) < atol


# ===========================================================================
# PATH A: soder.txt (mocap-based) input
# ===========================================================================
class TestSoderInput:
    """Tests that use soder{N}.txt as input. Auto-skip if those files
    aren't present in CAM_PARAMS_DIR."""

    @pytest.fixture(params=[0, 2, 4, 6])
    def cam_id(self, request):
        return request.param

    def test_soder_parses_to_valid_rotation(self, cam_id):
        soder_path = os.path.join(CAM_PARAMS_DIR, f"soder{cam_id}.txt")
        skip_if_missing(soder_path)

        R, d, s, rms = load_transformation(soder_path)
        assert is_valid_rotation(R), f"soder{cam_id}.txt: R is not a valid rotation matrix"
        assert d.shape in [(3,), (3, 1)]

    def test_soder_matches_existing_extrinsics_yaml(self, cam_id):
        """If both soder{N}.txt and camera_{N}_extrinsics.yaml already exist,
        they should describe the same transform."""
        soder_path = os.path.join(CAM_PARAMS_DIR, f"soder{cam_id}.txt")
        existing_path = os.path.join(CAM_PARAMS_DIR, f"camera_{cam_id}_extrinsics.yaml")
        skip_if_missing(soder_path, existing_path)

        R_soder, d_soder, _, _ = load_transformation(soder_path)
        R_existing, d_existing = load_camera_extrinsics(existing_path)

        assert_poses_close(
            R_soder, d_soder, R_existing, d_existing,
            label=f"camera_{cam_id}: soder.txt vs existing extrinsics.yaml",
            rot_tol=1e-1, trans_tol=1e-4,   # tight -- same source, should match near-exactly
        )

    def test_soder_round_trips_through_save_pose_to_yaml(self, cam_id, tmp_path):
        soder_path = os.path.join(CAM_PARAMS_DIR, f"soder{cam_id}.txt")
        skip_if_missing(soder_path)

        R, d, s, rms = load_transformation(soder_path)

        out_path = tmp_path / f"camera_{cam_id}_extrinsics_test.yaml"
        save_pose_to_yaml(
            R, d, str(out_path),
            frame_from=f"camera_{cam_id}", frame_to="world",
            scale_factor=s, rms_error=rms, source_file=f"soder{cam_id}.txt",
        )
        R_loaded, d_loaded = load_camera_extrinsics(str(out_path))

        np.testing.assert_allclose(R_loaded, R, atol=1e-9)
        np.testing.assert_allclose(d_loaded.flatten(), d.flatten(), atol=1e-9)


    @pytest.mark.parametrize("target_cam_id", [4, 6])
    def test_composed_camera_0_to_camera_N_matches_reference(self, target_cam_id):
        """soder0 + soder{N} composed via world should match the existing
        camera_0_to_camera_{N}.yaml, if it exists."""
        soder0_path = os.path.join(CAM_PARAMS_DIR, "soder0.txt")
        soderN_path = os.path.join(CAM_PARAMS_DIR, f"soder{target_cam_id}.txt")
        reference_path = os.path.join(CAM_PARAMS_DIR, f"camera_0_to_camera_{target_cam_id}.yaml")
        skip_if_missing(soder0_path, soderN_path, reference_path)

        R_c0_to_w, d_c0_to_w, _, _ = load_transformation(soder0_path)
        R_cN_to_w, d_cN_to_w, _, _ = load_transformation(soderN_path)
        R_composed, d_composed = compose_via_world(R_c0_to_w, d_c0_to_w, R_cN_to_w, d_cN_to_w)

        R_ref, T_ref = load_cam_to_cam_params(reference_path)

        assert_poses_close(
            R_composed, d_composed, R_ref, T_ref,
            label=f"composed camera_0_to_camera_{target_cam_id} (soder) vs reference",
        )


# ===========================================================================
# PATH B: checkerboard image input
# ===========================================================================
class TestImageInput:
    """Tests that use checkerboard images as input, running the actual
    calibrate_camera / stereo_calibrate functions. Auto-skip if the image
    folders aren't present or empty."""

    def test_intrinsics_from_images_match_existing_yaml(self):
        """Re-running calibrate_camera on the existing cam_0 images should
        reproduce intrinsics close to the ones already saved."""
        skip_if_no_images(IMAGES_DIR_C0)
        existing_path = os.path.join(CAM_PARAMS_DIR, "camera_0_intrinsics.yaml")
        skip_if_missing(existing_path)

        reproj, mtx, dist = calibrate_camera(images_folder=os.path.join(IMAGES_DIR_C0, "*.png"))
        mtx_existing, dist_existing = load_cam_params(existing_path)

        # Checkerboard corner-detection is deterministic given the same images,
        # so this should match closely, not just approximately.
        np.testing.assert_allclose(mtx, mtx_existing, rtol=1e-3)
        np.testing.assert_allclose(dist, dist_existing, rtol=1e-2, atol=1e-3)

    def test_stereo_calibration_matches_reference_camera_0_to_camera_2(self):
        """Re-running stereo_calibrate on the existing cam_0/cam_2 image pairs
        should reproduce R/T close to the existing camera_0_to_camera_2.yaml."""
        skip_if_no_images(IMAGES_DIR_C0)
        skip_if_no_images(IMAGES_DIR_C2)
        c0_intrinsics_path = os.path.join(CAM_PARAMS_DIR, "camera_0_intrinsics.yaml")
        c2_intrinsics_path = os.path.join(CAM_PARAMS_DIR, "camera_2_intrinsics.yaml")
        reference_path = os.path.join(CAM_PARAMS_DIR, "camera_0_to_camera_2.yaml")
        skip_if_missing(c0_intrinsics_path, c2_intrinsics_path, reference_path)

        mtx1, dist1 = load_cam_params(c0_intrinsics_path)
        mtx2, dist2 = load_cam_params(c2_intrinsics_path)

        rmse, R, T = stereo_calibrate(
            mtx1, dist1, mtx2, dist2,
            os.path.join(IMAGES_DIR_C0, "*.png"),
            os.path.join(IMAGES_DIR_C2, "*.png"),
        )

        R_ref, T_ref = load_cam_to_cam_params(reference_path)

        assert_poses_close(
            R, T, R_ref, T_ref,
            label="re-run stereo_calibrate vs reference camera_0_to_camera_2.yaml",
        )
        assert rmse < 1.0, f"Unexpectedly high stereo calibration RMSE: {rmse}"


