"""
Joint mapping utilities for converting between different skeletal formats
"""

import numpy as np
from typing import Tuple


# HumanML3D 22-joint format to COCO-17 mapping
# HumanML3D joint order:
# 0: root/pelvis, 1: left_hip, 2: right_hip, 3: spine1, 4: left_knee, 5: right_knee, 6: spine2,
# 7: left_ankle, 8: right_ankle, 9: spine3, 10: left_foot, 11: right_foot, 12: neck,
# 13: left_collar, 14: right_collar, 15: head, 16: left_shoulder, 17: right_shoulder,
# 18: left_elbow, 19: right_elbow, 20: left_wrist, 21: right_wrist

HUMANML3D_TO_COCO17_DIRECT_MAPPING = {
    # Direct joint mappings
    1: 11,   # left_hip
    2: 12,   # right_hip
    4: 13,   # left_knee
    5: 14,   # right_knee
    7: 15,   # left_ankle
    8: 16,   # right_ankle
    16: 5,   # left_shoulder
    17: 6,   # right_shoulder
    18: 7,   # left_elbow
    19: 8,   # right_elbow
    20: 9,   # left_wrist
    21: 10,  # right_wrist
}


def humanml3d_to_coco17(joints_humanml3d: np.ndarray) -> np.ndarray:
    """
    Convert HumanML3D 22-joint format to COCO-17 format.

    Args:
        joints_humanml3d: (num_frames, 22, 3) array of 3D joint positions

    Returns:
        joints_coco17: (num_frames, 17, 3) array of 3D joint positions in COCO-17 format
    """
    num_frames = joints_humanml3d.shape[0]
    joints_coco17 = np.zeros((num_frames, 17, 3), dtype=np.float32)

    # Direct mappings
    for hml_idx, coco_idx in HUMANML3D_TO_COCO17_DIRECT_MAPPING.items():
        joints_coco17[:, coco_idx, :] = joints_humanml3d[:, hml_idx, :]

    # Derived joints (need to be computed from existing joints)
    # Head joint (idx 15 in HumanML3D) -> nose (idx 0 in COCO-17)
    # Offset forward from head position
    head_pos = joints_humanml3d[:, 15, :]  # head joint
    neck_pos = joints_humanml3d[:, 12, :]  # neck joint

    # Nose is slightly forward and up from head
    head_to_neck = head_pos - neck_pos
    head_length = np.linalg.norm(head_to_neck, axis=1, keepdims=True) + 1e-8
    head_direction = head_to_neck / head_length

    # Place nose at head position + 0.1 * head_length forward
    joints_coco17[:, 0, :] = head_pos + 0.1 * head_direction * head_length

    # Eyes (synthesized from head position)
    # Left eye (idx 1)
    joints_coco17[:, 1, :] = head_pos + np.array([0.03, -0.05, 0.08])
    # Right eye (idx 2)
    joints_coco17[:, 2, :] = head_pos + np.array([-0.03, -0.05, 0.08])

    # Ears (synthesized from head position)
    # Left ear (idx 3)
    joints_coco17[:, 3, :] = head_pos + np.array([0.08, -0.03, 0.0])
    # Right ear (idx 4)
    joints_coco17[:, 4, :] = head_pos + np.array([-0.08, -0.03, 0.0])

    return joints_coco17


def extract_humanml3d_joints_from_feature_vector(motion_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 3D joint positions from HumanML3D 263-dimensional feature vector.

    HumanML3D feature structure (263-dim):
    - [0]: root angular velocity (1D)
    - [1:3]: root linear velocity XZ (2D)
    - [3]: root height Y (1D)
    - [4:67]: joint positions relative to root (21 joints × 3D = 63)
    - [67:130]: joint velocities (21 joints × 3D = 63)
    - [130:193]: joint rotations (21 joints × 3D = 63, 6D representation)
    - [193:259]: foot contact features
    - [259:263]: foot contact labels (4D)

    Args:
        motion_features: (num_frames, 263) feature array

    Returns:
        Tuple of:
        - joints_3d: (num_frames, 22, 3) absolute joint positions
        - root_trajectory: (num_frames, 3) root position trajectory
    """
    num_frames = motion_features.shape[0]

    # Extract root information
    root_angular_vel = motion_features[:, 0]
    root_linear_vel_xz = motion_features[:, 1:3]
    root_height = motion_features[:, 3]

    # Extract joint positions (relative to root)
    # 21 joints starting from index 4
    joints_relative = motion_features[:, 4:67].reshape(num_frames, 21, 3)

    # Reconstruct absolute root position by integrating velocity
    root_trajectory = np.zeros((num_frames, 3), dtype=np.float32)
    root_trajectory[0] = np.array([0.0, root_height[0], 0.0])

    for i in range(1, num_frames):
        # Integrate XZ velocity (assuming dt = 1/fps)
        dt = 1.0 / 20.0  # HumanML3D is typically 20 FPS
        root_trajectory[i, 0] = root_trajectory[i-1, 0] + root_linear_vel_xz[i, 0] * dt
        root_trajectory[i, 1] = root_height[i]
        root_trajectory[i, 2] = root_trajectory[i-1, 2] + root_linear_vel_xz[i, 1] * dt

    # Construct full skeleton (22 joints = 1 root + 21 others)
    joints_3d = np.zeros((num_frames, 22, 3), dtype=np.float32)
    joints_3d[:, 0, :] = root_trajectory  # root/pelvis
    joints_3d[:, 1:, :] = joints_relative + root_trajectory[:, np.newaxis, :]  # add root to relative positions

    return joints_3d, root_trajectory


# SMPL-24 to COCO-17 mapping
# SMPL joint order (simplified):
# 0: pelvis, 1: left_hip, 2: right_hip, 3: spine1, 4: left_knee, 5: right_knee, 6: spine2,
# 7: left_ankle, 8: right_ankle, 9: spine3, 10: left_foot, 11: right_foot, 12: neck,
# 13: left_collar, 14: right_collar, 15: head, 16: left_shoulder, 17: right_shoulder,
# 18: left_elbow, 19: right_elbow, 20: left_wrist, 21: right_wrist, 22: left_hand, 23: right_hand

SMPL_TO_COCO17_DIRECT_MAPPING = {
    1: 11,   # left_hip
    2: 12,   # right_hip
    4: 13,   # left_knee
    5: 14,   # right_knee
    7: 15,   # left_ankle
    8: 16,   # right_ankle
    16: 5,   # left_shoulder
    17: 6,   # right_shoulder
    18: 7,   # left_elbow
    19: 8,   # right_elbow
    20: 9,   # left_wrist
    21: 10,  # right_wrist
}


def smpl_to_coco17(joints_smpl: np.ndarray) -> np.ndarray:
    """
    Convert SMPL 24-joint format to COCO-17 format.

    Args:
        joints_smpl: (num_frames, 24, 3) array of 3D joint positions

    Returns:
        joints_coco17: (num_frames, 17, 3) array of 3D joint positions in COCO-17 format
    """
    num_frames = joints_smpl.shape[0]
    joints_coco17 = np.zeros((num_frames, 17, 3), dtype=np.float32)

    # Direct mappings
    for smpl_idx, coco_idx in SMPL_TO_COCO17_DIRECT_MAPPING.items():
        joints_coco17[:, coco_idx, :] = joints_smpl[:, smpl_idx, :]

    # Head joint (idx 15 in SMPL) -> facial keypoints in COCO-17
    head_pos = joints_smpl[:, 15, :]
    neck_pos = joints_smpl[:, 12, :]

    # Compute head direction
    head_to_neck = head_pos - neck_pos
    head_length = np.linalg.norm(head_to_neck, axis=1, keepdims=True) + 1e-8
    head_direction = head_to_neck / head_length

    # Nose (idx 0): at head position + offset forward
    joints_coco17[:, 0, :] = head_pos + 0.1 * head_direction * head_length

    # Eyes and ears (synthesized with fixed offsets from head)
    # These values are approximate and work for typical human proportions
    joints_coco17[:, 1, :] = head_pos + np.array([0.03, -0.05, 0.08])  # left_eye
    joints_coco17[:, 2, :] = head_pos + np.array([-0.03, -0.05, 0.08])  # right_eye
    joints_coco17[:, 3, :] = head_pos + np.array([0.08, -0.03, 0.0])   # left_ear
    joints_coco17[:, 4, :] = head_pos + np.array([-0.08, -0.03, 0.0])  # right_ear

    return joints_coco17


# OpenPose to COCO-17 mapping (OpenPose uses BODY_25 or COCO format)
# If OpenPose is already in COCO format, it should be directly compatible
# OpenPose BODY_25 format has 25 joints, COCO format has 18 joints

OPENPOSE_COCO18_TO_COCO17_MAPPING = {
    # OpenPose COCO-18 format to COCO-17 format
    0: 0,    # nose
    1: 5,    # left_shoulder (OpenPose neck is not used)
    2: 6,    # right_shoulder
    3: 7,    # left_elbow
    4: 8,    # right_elbow
    5: 9,    # left_wrist
    6: 10,   # right_wrist
    7: 11,   # left_hip
    8: 12,   # right_hip
    9: 13,   # left_knee
    10: 14,  # right_knee
    11: 15,  # left_ankle
    12: 16,  # right_ankle
    13: 1,   # left_eye
    14: 2,   # right_eye
    15: 3,   # left_ear
    16: 4,   # right_ear
}


def openpose_to_coco17(keypoints_openpose: np.ndarray, format: str = "coco") -> np.ndarray:
    """
    Convert OpenPose keypoints to COCO-17 format.

    Args:
        keypoints_openpose: OpenPose keypoints array
        format: "coco" for 18-point or "body25" for 25-point format

    Returns:
        keypoints_coco17: (17, 2) or (17, 3) array of keypoints
    """
    if format == "coco":
        # OpenPose COCO format has 18 keypoints
        if keypoints_openpose.shape[0] >= 17:
            # Direct mapping for most joints
            keypoints_coco17 = np.zeros((17, keypoints_openpose.shape[1]), dtype=np.float32)
            for op_idx, coco_idx in OPENPOSE_COCO18_TO_COCO17_MAPPING.items():
                if op_idx < keypoints_openpose.shape[0]:
                    keypoints_coco17[coco_idx] = keypoints_openpose[op_idx]
            return keypoints_coco17
        else:
            raise ValueError(f"Expected at least 17 keypoints, got {keypoints_openpose.shape[0]}")
    else:
        raise NotImplementedError(f"Format {format} not yet implemented")
