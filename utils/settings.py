from dataclasses import dataclass, field
import numpy as np

@dataclass
class Settings:
    # CAM PARAMS
    fs: int = 40
    dt: float = field(init=False)  # Mark `dt` as excluded from the constructor
    width: int = 1280 # image resolution
    height: int = 720 # image resolution

    # CAMS CALIB 
    # checkerboard params
    checkerboard_rows: int = 6 # number of rows on the checkerboard -1 
    checkerboard_columns: int = 7 # number of columns on the checkerboard -1 
    checkerboard_scaling: float = 0.108 # size of squares in meters
    # wand params
    wand_end_effector_local_pos: np.ndarray = field(
        default_factory=lambda: np.array([[0.000], [0.271], [0.000]]) # local pose of wand's end effector for pointing calibration
    )
    wand_marker_size: float = 0.176 # Marker size in meters (17.6 cm)
    
    def __post_init__(self):
        self.dt = 1 / self.fs  # Compute `dt` after initialization


