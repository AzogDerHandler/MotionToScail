"""
MotionPositioner Node
Controls root motion behavior and animation looping.
"""

import numpy as np
from typing import Dict, Any, Tuple


class MotionPositioner:
    """
    Control root motion (walk-in-place vs walk-through-scene) and playback mode (loop, ping-pong).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coco3d_joints": ("COCO3D_JOINTS",),
                "root_motion": (["walk_in_place", "walk_through_scene"],),
                "playback_mode": (["one_shot", "loop", "ping_pong"],),
            },
            "optional": {
                "target_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "loop_blend_frames": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 30,
                    "step": 1
                }),
                "scene_offset_x": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01
                }),
                "scene_offset_z": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("COCO3D_JOINTS", "INT")
    RETURN_NAMES = ("coco3d_joints", "frame_count")
    FUNCTION = "position_motion"
    CATEGORY = "MotionToSCAIL/Processing"

    def position_motion(
        self,
        coco3d_joints: Dict[str, Any],
        root_motion: str,
        playback_mode: str,
        target_frames: int = 0,
        loop_blend_frames: int = 5,
        scene_offset_x: float = 0.0,
        scene_offset_z: float = 0.0,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Process motion root and looping.

        Args:
            coco3d_joints: Input motion data
            root_motion: "walk_in_place" or "walk_through_scene"
            playback_mode: "one_shot", "loop", or "ping_pong"
            target_frames: Target frame count for looping (0 = use original)
            loop_blend_frames: Frames to cross-fade at loop boundary
            scene_offset_x: Horizontal offset for walk_through_scene
            scene_offset_z: Depth offset for walk_through_scene

        Returns:
            Tuple of (processed_coco3d_joints, frame_count)
        """
        joints_3d = coco3d_joints["joints_3d"].copy()
        fps = coco3d_joints["fps"]
        source_format = coco3d_joints["source_format"]

        num_frames, num_joints, _ = joints_3d.shape

        # Step 1: Process root motion
        if root_motion == "walk_in_place":
            joints_3d = self._walk_in_place(joints_3d)
        else:  # walk_through_scene
            joints_3d = self._walk_through_scene(joints_3d, scene_offset_x, scene_offset_z)

        # Step 2: Process playback mode
        if playback_mode == "loop" and target_frames > 0:
            joints_3d = self._create_loop(joints_3d, target_frames, loop_blend_frames)
        elif playback_mode == "ping_pong" and target_frames > 0:
            joints_3d = self._create_ping_pong(joints_3d, target_frames, loop_blend_frames)
        # else: one_shot - pass through unchanged

        output_frame_count = joints_3d.shape[0]

        output_coco3d = {
            "joints_3d": joints_3d,
            "fps": fps,
            "source_format": source_format,
        }

        print(f"Motion positioning: {num_frames} → {output_frame_count} frames")
        print(f"  Root motion: {root_motion}, Playback: {playback_mode}")

        return (output_coco3d, output_frame_count)

    def _walk_in_place(self, joints_3d: np.ndarray) -> np.ndarray:
        """
        Zero out root XZ translation, keeping Y (vertical bounce).
        Character stays centered.
        """
        # Compute root position (midpoint of hips)
        left_hip = joints_3d[:, 11, :]   # COCO-17 idx 11
        right_hip = joints_3d[:, 12, :]  # COCO-17 idx 12
        root_pos = (left_hip + right_hip) / 2.0

        # Get XZ offset to center at origin
        root_xz_offset = root_pos[:, [0, 2]]  # Take X and Z
        root_xz_offset = root_xz_offset - root_xz_offset[0:1]  # Relative to first frame

        # Apply offset to all joints (keep Y unchanged)
        joints_3d[:, :, 0] -= root_xz_offset[:, 0:1]  # X
        joints_3d[:, :, 2] -= root_xz_offset[:, 1:2]  # Z

        return joints_3d

    def _walk_through_scene(
        self,
        joints_3d: np.ndarray,
        offset_x: float,
        offset_z: float
    ) -> np.ndarray:
        """
        Preserve original root translation, optionally add scene offset.
        """
        joints_3d[:, :, 0] += offset_x  # X offset
        joints_3d[:, :, 2] += offset_z  # Z offset
        return joints_3d

    def _create_loop(
        self,
        joints_3d: np.ndarray,
        target_frames: int,
        blend_frames: int
    ) -> np.ndarray:
        """
        Create seamless loop by repeating and blending.
        """
        num_frames = joints_3d.shape[0]

        if target_frames <= num_frames:
            # Just trim
            return joints_3d[:target_frames]

        # Calculate number of cycles needed
        num_cycles = int(np.ceil(target_frames / num_frames))
        blend_frames = min(blend_frames, num_frames // 4)  # Limit blend to 25% of cycle

        # Build looped sequence
        looped_frames = []
        for cycle in range(num_cycles):
            if cycle == 0:
                # First cycle: full
                looped_frames.append(joints_3d)
            else:
                # Subsequent cycles: blend start with previous end
                cycle_data = joints_3d.copy()

                # Cross-fade first blend_frames of this cycle with last blend_frames of previous
                for i in range(blend_frames):
                    alpha = i / blend_frames  # 0 → 1
                    # Blend with end of previous cycle
                    prev_frame = looped_frames[-1][-(blend_frames - i)]
                    curr_frame = cycle_data[i]
                    cycle_data[i] = prev_frame * (1 - alpha) + curr_frame * alpha

                looped_frames.append(cycle_data)

        # Concatenate and trim to target length
        looped_joints = np.concatenate(looped_frames, axis=0)
        return looped_joints[:target_frames]

    def _create_ping_pong(
        self,
        joints_3d: np.ndarray,
        target_frames: int,
        blend_frames: int
    ) -> np.ndarray:
        """
        Play forward then reverse, with blending at boundaries.
        """
        num_frames = joints_3d.shape[0]

        # Reverse sequence, excluding first frame to avoid duplicate at seam
        # Forward: [0, 1, ..., N-1], Reverse: [N-2, N-3, ..., 0]
        reversed_joints = joints_3d[-2::-1].copy()

        # Blend at the turnaround
        blend_frames = min(blend_frames, num_frames // 4)

        # Create full ping-pong cycle (no duplicate frame at boundary)
        pingpong_cycle = np.concatenate([joints_3d, reversed_joints], axis=0)

        # Cross-fade at the forward→reverse seam for smooth turnaround
        # The seam is at index num_frames-1 (last forward frame) / num_frames (first reverse frame)
        seam = num_frames - 1
        half_blend = blend_frames // 2
        for i in range(blend_frames):
            alpha = i / blend_frames
            frame_idx = seam - half_blend + i
            if 0 <= frame_idx < pingpong_cycle.shape[0]:
                # Blend between the forward and reverse trajectories
                fwd_idx = min(frame_idx, num_frames - 1)
                rev_offset = frame_idx - seam
                rev_idx = max(0, min(rev_offset, reversed_joints.shape[0] - 1))
                pingpong_cycle[frame_idx] = (
                    joints_3d[fwd_idx] * (1 - alpha) +
                    reversed_joints[rev_idx] * alpha
                )

        # Repeat and trim to target
        if target_frames > pingpong_cycle.shape[0]:
            num_repeats = int(np.ceil(target_frames / pingpong_cycle.shape[0]))
            pingpong_joints = np.tile(pingpong_cycle, (num_repeats, 1, 1))
        else:
            pingpong_joints = pingpong_cycle

        return pingpong_joints[:target_frames]
