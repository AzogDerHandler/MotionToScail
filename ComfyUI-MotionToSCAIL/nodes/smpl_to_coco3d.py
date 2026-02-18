"""
SMPLToCOCO3D Node
Converts SMPL parameters (from GVHMR) to COCO3D_JOINTS.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple

from ..core.joint_mappings import smpl_to_coco17


class SMPLToCOCO3D:
    """
    Convert SMPL parameters to standardized COCO3D_JOINTS format.

    This node handles SMPL body model parameters from video-to-motion pipelines
    like GVHMR, which outputs pose rotations, shape betas, and global transformations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smpl_data": ("SMPL_DATA",),  # SMPL parameters
            },
            "optional": {
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "use_global_coord": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("COCO3D_JOINTS",)
    RETURN_NAMES = ("coco3d_joints",)
    FUNCTION = "convert"
    CATEGORY = "MotionToSCAIL/Converters"

    def convert(
        self,
        smpl_data: Dict[str, Any],
        fps: float = 30.0,
        use_global_coord: bool = True
    ) -> Tuple[Dict[str, Any]]:
        """
        Convert SMPL parameters to COCO3D format.

        Args:
            smpl_data: Dictionary containing SMPL parameters
                - 'global' or 'incam': dict with body_pose, betas, global_orient, transl
                - 'K_fullimg': camera intrinsics (optional)
            fps: Frames per second
            use_global_coord: Use global coordinates vs camera coordinates

        Returns:
            Tuple containing COCO3D_JOINTS dictionary
        """
        # Select coordinate system
        coord_key = 'global' if use_global_coord and 'global' in smpl_data else 'incam'

        if coord_key not in smpl_data:
            raise ValueError(f"SMPL data missing '{coord_key}' key")

        smpl_params = smpl_data[coord_key]

        # Extract parameters
        body_pose = smpl_params['body_pose']
        betas = smpl_params['betas']
        global_orient = smpl_params['global_orient']
        transl = smpl_params['transl']

        # Convert tensors to numpy
        if isinstance(body_pose, torch.Tensor):
            body_pose = body_pose.cpu().numpy()
            betas = betas.cpu().numpy()
            global_orient = global_orient.cpu().numpy()
            transl = transl.cpu().numpy()

        num_frames = body_pose.shape[0]

        print(f"Converting SMPL motion: {num_frames} frames at {fps} FPS")
        print(f"Using {coord_key} coordinates")

        # Apply SMPL forward kinematics to get joint positions
        joints_smpl = self._smpl_forward_kinematics(
            body_pose, betas, global_orient, transl
        )

        # Convert SMPL 24 joints to COCO-17 format
        joints_coco17 = smpl_to_coco17(joints_smpl)

        # Create COCO3D_JOINTS output
        coco3d_joints = {
            "joints_3d": joints_coco17.astype(np.float32),
            "fps": float(fps),
            "source_format": "smpl",
        }

        print(f"Converted to COCO3D: {joints_coco17.shape[0]} frames, 17 joints")

        return (coco3d_joints,)

    def _smpl_forward_kinematics(
        self,
        body_pose: np.ndarray,
        betas: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray
    ) -> np.ndarray:
        """
        Simplified SMPL forward kinematics.

        This is a placeholder that uses a simplified FK approach.
        For production use, integrate with smplx library or implement full FK.

        Args:
            body_pose: (num_frames, 69) axis-angle rotations for 23 joints
            betas: (num_frames, 10) shape parameters
            global_orient: (num_frames, 3) root orientation
            transl: (num_frames, 3) global translation

        Returns:
            joints: (num_frames, 24, 3) 3D joint positions
        """
        num_frames = body_pose.shape[0]

        # Try to use smplx if available
        try:
            import smplx

            # Create SMPL model (neutral gender)
            smpl_model = smplx.create(
                model_path='./models',  # Update this path
                model_type='smpl',
                gender='neutral',
                use_face_contour=False,
                use_pca=False,
                num_betas=10,
            )

            # Convert to tensors
            body_pose_tensor = torch.tensor(body_pose, dtype=torch.float32)
            betas_tensor = torch.tensor(betas, dtype=torch.float32)
            global_orient_tensor = torch.tensor(global_orient, dtype=torch.float32)
            transl_tensor = torch.tensor(transl, dtype=torch.float32)

            # Run SMPL forward pass
            with torch.no_grad():
                output = smpl_model(
                    body_pose=body_pose_tensor,
                    betas=betas_tensor,
                    global_orient=global_orient_tensor,
                    transl=transl_tensor,
                    return_verts=False,
                )

            joints = output.joints.cpu().numpy()[:, :24, :]  # Take first 24 joints

            print("Using smplx library for forward kinematics")
            return joints

        except (ImportError, Exception) as e:
            print(f"smplx not available, using simplified FK: {e}")

            # Fallback: simplified FK using only global transform
            # This is a rough approximation - proper implementation needs full SMPL FK
            joints = np.zeros((num_frames, 24, 3), dtype=np.float32)

            # Use translation as approximate skeleton root
            # Apply global orientation (simplified - just use as offset)
            for i in range(num_frames):
                # Root position
                joints[i, 0] = transl[i]

                # Approximate other joints based on standard skeleton proportions
                # This is very rough and should be replaced with proper SMPL FK
                # For now, create a basic T-pose skeleton scaled by body height
                # Normalize height to ~1.7m; fallback to 1.0 if Y translation is near zero
                y_abs = abs(float(transl[i, 1]))
                height_scale = y_abs / 1.7 if y_abs > 0.01 else 1.0

                # Basic skeleton layout (T-pose)
                joints[i, 1] = transl[i] + np.array([0.1, -0.1, 0.0]) * height_scale  # left hip
                joints[i, 2] = transl[i] + np.array([-0.1, -0.1, 0.0]) * height_scale  # right hip
                joints[i, 4] = transl[i] + np.array([0.1, -0.5, 0.0]) * height_scale  # left knee
                joints[i, 5] = transl[i] + np.array([-0.1, -0.5, 0.0]) * height_scale  # right knee
                joints[i, 7] = transl[i] + np.array([0.1, -0.9, 0.0]) * height_scale  # left ankle
                joints[i, 8] = transl[i] + np.array([-0.1, -0.9, 0.0]) * height_scale  # right ankle
                joints[i, 12] = transl[i] + np.array([0.0, 0.4, 0.0]) * height_scale  # neck
                joints[i, 15] = transl[i] + np.array([0.0, 0.6, 0.0]) * height_scale  # head
                joints[i, 16] = transl[i] + np.array([0.2, 0.3, 0.0]) * height_scale  # left shoulder
                joints[i, 17] = transl[i] + np.array([-0.2, 0.3, 0.0]) * height_scale  # right shoulder
                joints[i, 18] = transl[i] + np.array([0.4, 0.1, 0.0]) * height_scale  # left elbow
                joints[i, 19] = transl[i] + np.array([-0.4, 0.1, 0.0]) * height_scale  # right elbow
                joints[i, 20] = transl[i] + np.array([0.6, 0.0, 0.0]) * height_scale  # left wrist
                joints[i, 21] = transl[i] + np.array([-0.6, 0.0, 0.0]) * height_scale  # right wrist

            print("WARNING: Using simplified FK approximation. Install smplx for accurate results.")
            return joints
