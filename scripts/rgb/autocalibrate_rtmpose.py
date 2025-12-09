# To run from repo root : python3 scripts/rgb/calibrate_extrinsics_rtmpose.py

import os
import sys

# Resolve repo root similarly to calibrate_cameras.py
script_path = os.path.abspath(__file__)
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add repo root to PYTHONPATH
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from utils.settings import Settings
from utils.calib_utils import (
    autocalibrate_from_human,
    record_calibration_videos,
)

settings = Settings()


def main():
    # In your repo, config is usually under <repo_root>/config
    config_dir = os.path.join(repo_path, "config")

    # 1) Record videos for calibration (multi-cam, robust)
    recorded_sessions = record_calibration_videos(config_dir)
    # recorded_sessions = [{"0": "/root/workspace/ros_ws/src/rt-cosmik/tests/full/data/camera_0.mp4", "2": "/root/workspace/ros_ws/src/rt-cosmik/tests/full/data/camera_2.mp4"}]  # <-- For testing, skip recording step

    autocalibrate_from_human(
        recorded_sessions,
        config_dir=config_dir,
        height_m=settings.subject_height,  
        gender=settings.subject_gender,      # "m" or "f"
    )

if __name__ == "__main__":
    main()
