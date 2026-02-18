"""
COCO3DToSCAILPose Node
Main bridge node that converts COCO3D motion to SCAIL-compatible pose format.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
import torch


class COCO3DToSCAILPose:
    """
    Convert processed COCO3D motion data to SCAIL/NLF pose format.

    This is the main bridge between motion data and SCAIL's cylinder renderer.
    Handles retargeting to character proportions and camera alignment.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coco3d_joints": ("COCO3D_JOINTS",),
                "ref_skeleton": ("REF_SKELETON",),
                "target_width": ("INT", {
                    "default": 832,
                    "min": 64,
                    "max": 4096,
                    "step": 8
                }),
                "target_height": ("INT", {
                    "default": 480,
                    "min": 64,
                    "max": 4096,
                    "step": 8
                }),
            },
        }

    RETURN_TYPES = ("NLFPRED", "IMAGE")
    RETURN_NAMES = ("nlf_poses", "pose_preview")
    FUNCTION = "convert_to_scail"
    CATEGORY = "MotionToSCAIL/Core"

    def convert_to_scail(
        self,
        coco3d_joints: Dict[str, Any],
        ref_skeleton: Dict[str, Any],
        target_width: int,
        target_height: int,
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Convert COCO3D joints to SCAIL pose format.

        Args:
            coco3d_joints: Motion data in COCO3D format
            ref_skeleton: Reference character skeleton with proportions
            target_width: Generation width
            target_height: Generation height

        Returns:
            (nlf_poses, pose_preview)
        """
        joints_3d = coco3d_joints["joints_3d"]  # (num_frames, 17, 3)
        num_frames = joints_3d.shape[0]

        print(f"Converting {num_frames} frames to SCAIL pose format")
        print(f"  Target resolution: {target_width}x{target_height}")

        # Step 1: Retarget motion to character proportions
        retargeted_joints = self._retarget_to_character(joints_3d, ref_skeleton)

        # Step 2: Apply camera alignment
        aligned_joints = self._apply_camera_alignment(
            retargeted_joints, ref_skeleton, target_width, target_height
        )

        # Step 3: Package as NLFPRED format
        nlf_poses = self._create_nlfpred(aligned_joints)

        # Step 4: Create preview visualization
        pose_preview = self._create_pose_preview(aligned_joints, target_width, target_height)

        print(f"Converted to NLF format: {num_frames} frames")

        return (nlf_poses, pose_preview)

    def _retarget_to_character(
        self,
        joints_3d: np.ndarray,
        ref_skeleton: Dict[str, Any]
    ) -> np.ndarray:
        """
        Retarget motion to match character proportions.

        Uses limb-length scaling to adapt motion to reference character.
        """
        retargeted = joints_3d.copy()
        num_frames = joints_3d.shape[0]

        # Calculate limb lengths from motion data (first frame)
        motion_limb_lengths = self._calculate_limb_lengths(joints_3d[0])

        # Get reference limb lengths (in pixels, will scale)
        ref_limb_lengths = {
            "upper_arm": ref_skeleton["upper_arm_length"],
            "forearm": ref_skeleton["forearm_length"],
            "upper_leg": ref_skeleton["upper_leg_length"],
            "lower_leg": ref_skeleton["lower_leg_length"],
            "torso": ref_skeleton["torso_length"],
        }

        # Calculate scale factors
        scale_factors = {}
        for limb_name in motion_limb_lengths.keys():
            if limb_name in ref_limb_lengths:
                motion_len = motion_limb_lengths[limb_name]
                ref_len = ref_limb_lengths[limb_name]
                scale_factors[limb_name] = ref_len / (motion_len + 1e-8)

        # Apply scaling to limb chains
        # This is a simplified version - full implementation needs hierarchical FK
        avg_scale = np.mean(list(scale_factors.values()))

        # Scale all joints uniformly (simplified approach)
        retargeted = joints_3d * avg_scale

        print(f"  Retargeting scale factor: {avg_scale:.3f}")

        return retargeted

    def _calculate_limb_lengths(self, joints: np.ndarray) -> Dict[str, float]:
        """Calculate limb lengths from joint positions."""
        def dist(idx1, idx2):
            return np.linalg.norm(joints[idx1] - joints[idx2])

        return {
            "upper_arm": dist(5, 7),  # left_shoulder to left_elbow
            "forearm": dist(7, 9),  # left_elbow to left_wrist
            "upper_leg": dist(11, 13),  # left_hip to left_knee
            "lower_leg": dist(13, 15),  # left_knee to left_ankle
            "torso": np.linalg.norm(
                (joints[5] + joints[6]) / 2 - (joints[11] + joints[12]) / 2
            ),
        }

    def _apply_camera_alignment(
        self,
        joints_3d: np.ndarray,
        ref_skeleton: Dict[str, Any],
        target_width: int,
        target_height: int
    ) -> np.ndarray:
        """
        Apply camera projection to align skeleton with reference image.

        Replicates SCAIL's solve_new_camera_params_central logic.
        """
        num_frames = joints_3d.shape[0]

        # Compute virtual camera intrinsics
        # Focal length adjusted so skeleton projects to correct screen position
        focal_length = max(target_width, target_height)  # Simple heuristic

        # Principal point (image center)
        cx = target_width / 2.0
        cy = target_height / 2.0

        # Camera intrinsic matrix
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # For now, keep joints in 3D (SCAIL renderer will handle projection)
        # In full implementation, adjust Z-depth to match reference character scale

        # Center the skeleton in frame
        root_pos = (joints_3d[:, 11, :] + joints_3d[:, 12, :]) / 2  # Hip center
        joints_3d[:, :, 0] -= root_pos[:, 0:1]  # Center X
        joints_3d[:, :, 1] -= root_pos[:, 1:2]  # Center Y

        # Adjust Z-depth to place character at appropriate distance
        # Typical SCAIL depth is around 3-4 meters
        joints_3d[:, :, 2] += 3.5  # Move to appropriate depth

        return joints_3d

    def _create_nlfpred(self, joints_3d: np.ndarray) -> Dict[str, Any]:
        """
        Format joints as NLFPRED dictionary for SCAIL/Kijai compatibility.

        NLFPRED format (from Kijai's RenderNLFPoses):
        {
            "joints3d_nonparam": [array(1, num_joints, 3), ...] per frame
        }
        """
        num_frames = joints_3d.shape[0]

        # Create list of joint arrays (one per frame)
        # Each array shape: (1, 17, 3) - batch dimension is 1
        joints3d_nonparam = []
        for i in range(num_frames):
            frame_joints = joints_3d[i:i+1, :, :]  # (1, 17, 3)
            joints3d_nonparam.append(frame_joints)

        nlf_pred = {
            "joints3d_nonparam": joints3d_nonparam,
            # Add any other fields that RenderNLFPoses expects
        }

        return nlf_pred

    def _create_pose_preview(
        self,
        joints_3d: np.ndarray,
        width: int,
        height: int
    ) -> Any:
        """
        Create simple wireframe preview of poses.

        Returns placeholder for now - full implementation would render skeleton.
        """
        # Create blank image tensor (batch=num_frames, height, width, channels=3)
        num_frames = joints_3d.shape[0]
        preview = np.zeros((num_frames, height, width, 3), dtype=np.float32)

        # TODO: Draw skeleton wireframe using COCO17_SKELETON connections

        # Convert to torch tensor for ComfyUI
        preview_tensor = torch.from_numpy(preview)

        return preview_tensor
