from dataclasses import dataclass, field
from typing import List, Dict, Union
import numpy as np

@dataclass
class Settings:
    # CAM PARAMS
    fs: int = 30
    width: int = 1280 # image resolution
    height: int = 800 # image resolution
    fourcc: str = "I420" # video codec

    # CAMS CALIB 
    # checkerboard params
    checkerboard_rows: int = 7 # number of rows on the checkerboard -1 
    checkerboard_columns: int = 9 # number of columns on the checkerboard -1 
    checkerboard_scaling: float = 0.069 # size of squares in meters
    # wand params
    wand_end_effector_local_pos: np.ndarray = field(
        default_factory=lambda: np.array([[0.000], [0.271], [0.000]]) # local pose of wand's end effector for pointing calibration
    )
    wand_marker_size: float = 0.176 # Marker size in meters (17.6 cm)

    # Define multiple chessboard configurations (small and large)
    checkerboard_configs: List[Dict[str, Union[int, float]]] = field(default_factory=lambda: [
        {"rows": 7, "columns": 10, "square_size": 0.025},  # Small chessboard
        {"rows": 6, "columns": 7, "square_size": 0.108},   # Large chessboard
        {"rows": 7, "columns": 9, "square_size": 0.069}    # Marie chessboard
    ])

