"""
MotionBlendIn Node
Generates smooth transition from reference A-pose to animation start.
"""

import numpy as np
from typing import Dict, Any, Tuple


class MotionBlendIn:
    """
    Generate smooth transition frames from reference A-pose to animation start.

    Helps SCAIL transition naturally from the reference image by blending
    from the character's A-pose to the first frame of the animation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_skeleton": ("REF_SKELETON",),
                "target_motion": ("COCO3D_JOINTS",),
                "blend_frames": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 30,
                    "step": 1
                }),
            },
            "optional": {
                "easing": (["linear", "ease_in", "ease_out", "ease_in_out"],),
            }
        }

    RETURN_TYPES = ("COCO3D_JOINTS", "INT")
    RETURN_NAMES = ("coco3d_joints", "frame_count")
    FUNCTION = "blend_in"
    CATEGORY = "MotionToSCAIL/Processing"

    def blend_in(
        self,
        ref_skeleton: Dict[str, Any],
        target_motion: Dict[str, Any],
        blend_frames: int,
        easing: str = "ease_out"
    ) -> Tuple[Dict[str, Any], int]:
        """
        Create blend-in transition.

        Args:
            ref_skeleton: Reference skeleton with canonical A-pose
            target_motion: Animation to blend into
            blend_frames: Number of transition frames to prepend
            easing: Easing function ("linear", "ease_in", "ease_out", "ease_in_out")

        Returns:
            Tuple of (motion_with_blend, total_frame_count)
        """
        # Extract reference A-pose (3D canonical pose)
        a_pose_3d = ref_skeleton["keypoints_3d"]  # (17, 3)

        # Extract target motion
        target_joints = target_motion["joints_3d"]  # (num_frames, 17, 3)
        target_frame_0 = target_joints[0]  # First frame of animation

        # Generate blend frames
        blended_frames = []

        for i in range(blend_frames):
            # Compute blend parameter t
            t = (i + 1) / blend_frames  # 0 → 1

            # Apply easing function
            t_eased = self._apply_easing(t, easing)

            # Linear interpolation between A-pose and target frame 0
            blended_frame = a_pose_3d * (1 - t_eased) + target_frame_0 * t_eased
            blended_frames.append(blended_frame)

        # Stack blend frames
        blend_array = np.stack(blended_frames, axis=0)  # (blend_frames, 17, 3)

        # Concatenate with target motion
        full_motion = np.concatenate([blend_array, target_joints], axis=0)

        output_coco3d = {
            "joints_3d": full_motion,
            "fps": target_motion["fps"],
            "source_format": target_motion["source_format"],
        }

        total_frames = full_motion.shape[0]

        print(f"Blend-in: Added {blend_frames} transition frames")
        print(f"  Total frames: {total_frames} ({blend_frames} blend + {target_joints.shape[0]} animation)")

        return (output_coco3d, total_frames)

    def _apply_easing(self, t: float, easing: str) -> float:
        """Apply easing function to blend parameter."""
        if easing == "linear":
            return t
        elif easing == "ease_in":
            return t * t
        elif easing == "ease_out":
            return 1 - (1 - t) ** 2
        elif easing == "ease_in_out":
            # Smoothstep
            return t * t * (3 - 2 * t)
        else:
            return t
