"""
Core type definitions for MotionToSCAIL pipeline
"""

import numpy as np
from typing import TypedDict, Literal, Optional, Dict, Any


class COCO3DJoints(TypedDict):
    """
    Standardized 3D joint positions in COCO-17 format.
    This is the central interchange format for all motion data.

    Joint order (COCO-17):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    Coordinate system: X-right, Y-down, Z-forward (matching NLF/camera convention)
    Units: meters (real-world scale)
    """
    joints_3d: np.ndarray  # shape: (num_frames, 17, 3)
    fps: float  # frames per second
    source_format: Literal["smpl", "humanml3d"]  # for debugging/logging


class RefSkeleton(TypedDict):
    """
    Reference character skeleton with proportions and viewpoint information.
    Extracted from a reference image (front-view A-pose).
    """
    keypoints_2d: np.ndarray  # shape: (17, 2) - COCO-17 in image pixel coords
    keypoints_3d: np.ndarray  # shape: (17, 3) - canonical 3D A-pose skeleton
    body_height: float  # total height in pixels (image space)
    shoulder_width: float  # shoulder-to-shoulder in pixels
    torso_length: float  # shoulder midpoint to hip midpoint
    upper_arm_length: float  # shoulder to elbow
    forearm_length: float  # elbow to wrist
    upper_leg_length: float  # hip to knee
    lower_leg_length: float  # knee to ankle
    head_height: float  # top of head to chin approximate
    image_height: int  # reference image dimensions
    image_width: int
    view_angle: Dict[str, Any]  # {"preset": str, "rotation_deg": [rx, ry, rz]}
    source: Literal["dwpose", "openpose", "mask"]  # detection method used


# COCO-17 skeleton connectivity for cylinder rendering
COCO17_SKELETON = [
    # Arms and shoulders
    (5, 6),   # left_shoulder to right_shoulder
    (5, 7),   # left_shoulder to left_elbow
    (7, 9),   # left_elbow to left_wrist
    (6, 8),   # right_shoulder to right_elbow
    (8, 10),  # right_elbow to right_wrist

    # Torso
    (5, 11),  # left_shoulder to left_hip
    (6, 12),  # right_shoulder to right_hip
    (11, 12), # left_hip to right_hip

    # Legs
    (11, 13), # left_hip to left_knee
    (13, 15), # left_knee to left_ankle
    (12, 14), # right_hip to right_knee
    (14, 16), # right_knee to right_ankle

    # Head (nose to eyes to ears)
    (0, 1),   # nose to left_eye
    (0, 2),   # nose to right_eye
    (1, 3),   # left_eye to left_ear
    (2, 4),   # right_eye to right_ear
]

# COCO-17 joint names for reference
COCO17_JOINT_NAMES = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle",   # 16
]
