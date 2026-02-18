"""
HumanML3DToCOCO3D Node
Converts HumanML3D format motion data (from HY-Motion or MotionDiffuse) to COCO3D_JOINTS.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple

from ..core.joint_mappings import humanml3d_to_coco17, extract_humanml3d_joints_from_feature_vector


class HumanML3DToCOCO3D:
    """
    Convert HumanML3D format motion data to standardized COCO3D_JOINTS format.

    This node handles motion data from text-to-motion models like HY-Motion and MotionDiffuse,
    which output 263-dimensional feature vectors per frame representing human motion.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_data": ("MOTION_DATA",),  # HumanML3D format motion data
            },
            "optional": {
                "fps": ("FLOAT", {
                    "default": 20.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("COCO3D_JOINTS",)
    RETURN_NAMES = ("coco3d_joints",)
    FUNCTION = "convert"
    CATEGORY = "MotionToSCAIL/Converters"

    def convert(
        self,
        motion_data: Dict[str, Any],
        fps: float = 20.0
    ) -> Tuple[Dict[str, Any]]:
        """
        Convert HumanML3D motion data to COCO3D format.

        Args:
            motion_data: Dictionary containing HumanML3D motion features
                - 'motion': tensor of shape (num_frames, 263)
                - 'motion_mask': optional mask
                - 'motion_length': number of valid frames
            fps: Frames per second (default 20.0 for HumanML3D)

        Returns:
            Tuple containing COCO3D_JOINTS dictionary
        """
        # Extract motion tensor
        if isinstance(motion_data, dict) and 'motion' in motion_data:
            motion_tensor = motion_data['motion']
            if isinstance(motion_tensor, torch.Tensor):
                motion_features = motion_tensor.cpu().numpy()
            else:
                motion_features = np.array(motion_tensor)

            # Get actual motion length if available
            if 'motion_length' in motion_data:
                motion_length = motion_data['motion_length']
                if isinstance(motion_length, torch.Tensor):
                    motion_length = motion_length.item()
                # Trim to actual length
                motion_features = motion_features[:motion_length]
        else:
            raise ValueError("motion_data must be a dictionary with 'motion' key")

        # Validate shape
        if motion_features.ndim != 2 or motion_features.shape[1] != 263:
            raise ValueError(
                f"Expected motion features of shape (num_frames, 263), "
                f"got {motion_features.shape}"
            )

        print(f"Converting HumanML3D motion: {motion_features.shape[0]} frames at {fps} FPS")

        # Extract 3D joint positions from feature vector
        joints_humanml3d, root_trajectory = extract_humanml3d_joints_from_feature_vector(
            motion_features
        )

        # Convert HumanML3D 22 joints to COCO-17 format
        joints_coco17 = humanml3d_to_coco17(joints_humanml3d)

        # Create COCO3D_JOINTS output
        coco3d_joints = {
            "joints_3d": joints_coco17.astype(np.float32),
            "fps": float(fps),
            "source_format": "humanml3d",
        }

        print(f"Converted to COCO3D: {joints_coco17.shape[0]} frames, 17 joints")

        return (coco3d_joints,)
