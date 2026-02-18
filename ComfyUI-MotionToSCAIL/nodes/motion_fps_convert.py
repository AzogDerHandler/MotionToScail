"""
MotionFPSConvert Node
Resamples motion data to a target FPS using interpolation.
"""

import numpy as np
from scipy import interpolate
from typing import Dict, Any, Tuple


class MotionFPSConvert:
    """
    Resample motion data to a target FPS.

    Useful for matching SCAIL's generation FPS or adjusting motion speed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coco3d_joints": ("COCO3D_JOINTS",),
                "target_fps": ("FLOAT", {
                    "default": 16.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "display": "number"
                }),
            },
            "optional": {
                "interpolation": (["linear", "cubic", "nearest"],),
            }
        }

    RETURN_TYPES = ("COCO3D_JOINTS", "INT")
    RETURN_NAMES = ("coco3d_joints", "frame_count")
    FUNCTION = "convert_fps"
    CATEGORY = "MotionToSCAIL/Processing"

    def convert_fps(
        self,
        coco3d_joints: Dict[str, Any],
        target_fps: float,
        interpolation: str = "linear"
    ) -> Tuple[Dict[str, Any], int]:
        """
        Resample motion to target FPS.

        Args:
            coco3d_joints: COCO3D_JOINTS dictionary
            target_fps: Target frames per second
            interpolation: Interpolation method ("linear", "cubic", "nearest")

        Returns:
            Tuple of (resampled_coco3d_joints, frame_count)
        """
        joints_3d = coco3d_joints["joints_3d"]
        source_fps = coco3d_joints["fps"]
        source_format = coco3d_joints["source_format"]

        num_frames, num_joints, _ = joints_3d.shape

        # Check if FPS are already matching
        if abs(source_fps - target_fps) < 0.01:
            print(f"Source FPS ({source_fps}) already matches target FPS ({target_fps}), passing through")
            return (coco3d_joints, num_frames)

        # Calculate durations and target frame count
        duration = num_frames / source_fps
        target_frame_count = int(duration * target_fps)

        print(f"Resampling from {source_fps} FPS to {target_fps} FPS")
        print(f"  Source: {num_frames} frames, {duration:.2f} seconds")
        print(f"  Target: {target_frame_count} frames, {duration:.2f} seconds")

        # Create time arrays
        source_times = np.linspace(0, duration, num_frames)
        target_times = np.linspace(0, duration, target_frame_count)

        # Resample each joint's position
        resampled_joints = np.zeros((target_frame_count, num_joints, 3), dtype=np.float32)

        for joint_idx in range(num_joints):
            for axis_idx in range(3):
                # Get source values for this joint and axis
                source_values = joints_3d[:, joint_idx, axis_idx]

                # Create interpolation function
                if interpolation == "cubic":
                    try:
                        f = interpolate.interp1d(
                            source_times, source_values,
                            kind='cubic', fill_value="extrapolate"
                        )
                    except ValueError:
                        # Fall back to linear if cubic fails
                        print(f"Cubic interpolation failed for joint {joint_idx}, using linear")
                        f = interpolate.interp1d(
                            source_times, source_values,
                            kind='linear', fill_value="extrapolate"
                        )
                elif interpolation == "nearest":
                    f = interpolate.interp1d(
                        source_times, source_values,
                        kind='nearest', fill_value="extrapolate"
                    )
                else:  # linear
                    f = interpolate.interp1d(
                        source_times, source_values,
                        kind='linear', fill_value="extrapolate"
                    )

                # Resample
                resampled_joints[:, joint_idx, axis_idx] = f(target_times)

        # Create output
        output_coco3d = {
            "joints_3d": resampled_joints,
            "fps": float(target_fps),
            "source_format": source_format,
        }

        return (output_coco3d, target_frame_count)
