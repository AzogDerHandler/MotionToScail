"""
RefSkeletonExtractor Node
Extracts character proportions from reference images and pose data.
"""

import numpy as np
import json
from typing import Dict, Any, Tuple, Optional


class RefSkeletonExtractor:
    """
    Extract character skeleton and proportions from reference image.

    Supports multiple input methods (cascade):
    1. DWPose detection (highest quality)
    2. OpenPose detection (fallback)
    3. Mask-based geometric fitting (always available)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front_image": ("IMAGE",),
                "front_mask": ("MASK",),
                "actual_ref_image": ("IMAGE",),
            },
            "optional": {
                "actual_ref_mask": ("MASK",),
                "dw_pose_front": ("DWPOSES",),
                "openpose_front": ("POSE_KEYPOINT",),
                "view_preset": ([
                    "front", "front_3/4_left", "front_3/4_right",
                    "side_left", "side_right", "back",
                    "back_3/4_left", "back_3/4_right", "top_down", "custom"
                ],),
                "rotation_x": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0
                }),
                "rotation_y": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0
                }),
                "rotation_z": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0
                }),
            }
        }

    RETURN_TYPES = ("REF_SKELETON", "DWPOSES", "IMAGE", "STRING")
    RETURN_NAMES = ("ref_skeleton", "ref_dw_pose", "debug_image", "source_method")
    FUNCTION = "extract_skeleton"
    CATEGORY = "MotionToSCAIL/Reference"

    def extract_skeleton(
        self,
        front_image: Any,
        front_mask: Any,
        actual_ref_image: Any,
        actual_ref_mask: Optional[Any] = None,
        dw_pose_front: Optional[Dict] = None,
        openpose_front: Optional[Dict] = None,
        view_preset: str = "front",
        rotation_x: float = 0.0,
        rotation_y: float = 0.0,
        rotation_z: float = 0.0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Any, str]:
        """
        Extract skeleton from reference image.

        Returns:
            (ref_skeleton, ref_dw_pose, debug_image, source_method)
        """
        # Determine which method to use (cascade)
        keypoints_2d = None
        source_method = None

        # Priority 1: DWPose
        if dw_pose_front is not None:
            try:
                keypoints_2d, source_method = self._extract_from_dwpose(dw_pose_front)
                print("Using DWPose for skeleton extraction")
            except Exception as e:
                print(f"DWPose extraction failed: {e}")

        # Priority 2: OpenPose
        if keypoints_2d is None and openpose_front is not None:
            try:
                keypoints_2d, source_method = self._extract_from_openpose(openpose_front)
                print("Using OpenPose for skeleton extraction")
            except Exception as e:
                print(f"OpenPose extraction failed: {e}")

        # Priority 3: Mask-based fitting
        if keypoints_2d is None:
            try:
                keypoints_2d, source_method = self._extract_from_mask(front_mask, front_image)
                print("Using mask-based fitting for skeleton extraction")
            except Exception as e:
                raise RuntimeError(f"All skeleton extraction methods failed. Mask error: {e}")

        # Calculate proportions from 2D keypoints
        proportions = self._calculate_proportions(keypoints_2d)

        # Build canonical 3D A-pose
        keypoints_3d = self._build_canonical_3d_pose(keypoints_2d, proportions)

        # Get image dimensions
        image_height, image_width = self._get_image_dimensions(front_image)

        # Determine view angle
        view_angle = self._get_view_angle(view_preset, rotation_x, rotation_y, rotation_z)

        # Project to target viewpoint
        projected_2d = self._project_to_viewpoint(keypoints_3d, view_angle, image_width, image_height)

        # Format as DWPOSES for SCAIL compatibility
        ref_dw_pose = self._format_as_dwposes(projected_2d, image_width, image_height)

        # Create REF_SKELETON output
        ref_skeleton = {
            "keypoints_2d": keypoints_2d,
            "keypoints_3d": keypoints_3d,
            "body_height": proportions["body_height"],
            "shoulder_width": proportions["shoulder_width"],
            "torso_length": proportions["torso_length"],
            "upper_arm_length": proportions["upper_arm_length"],
            "forearm_length": proportions["forearm_length"],
            "upper_leg_length": proportions["upper_leg_length"],
            "lower_leg_length": proportions["lower_leg_length"],
            "head_height": proportions["head_height"],
            "image_height": image_height,
            "image_width": image_width,
            "view_angle": view_angle,
            "source": source_method,
        }

        # Create debug visualization
        debug_image = self._create_debug_visualization(
            front_image, keypoints_2d, projected_2d
        )

        return (ref_skeleton, ref_dw_pose, debug_image, source_method)

    def _extract_from_dwpose(self, dw_pose_data: Dict) -> Tuple[np.ndarray, str]:
        """Extract COCO-17 keypoints from DWPose detection."""
        # DWPose data is a list with one dict per detected person
        if isinstance(dw_pose_data, list) and len(dw_pose_data) > 0:
            person = dw_pose_data[0]
            if "people" in person and len(person["people"]) > 0:
                person_data = person["people"][0]
                pose_kp = person_data["pose_keypoints_2d"]

                # pose_keypoints_2d is flat array: [x0, y0, conf0, x1, y1, conf1, ...]
                keypoints = np.array(pose_kp).reshape(-1, 3)[:, :2]  # Take only x, y

                # DWPose already uses COCO format, take first 17
                if keypoints.shape[0] >= 17:
                    return keypoints[:17], "dwpose"

        raise ValueError("Invalid DWPose data format")

    def _extract_from_openpose(self, openpose_data: Dict) -> Tuple[np.ndarray, str]:
        """Extract COCO-17 keypoints from OpenPose detection."""
        if "people" in openpose_data and len(openpose_data["people"]) > 0:
            person = openpose_data["people"][0]
            if "pose_keypoints_2d" in person:
                # OpenPose format: nested array [[x, y, conf], ...]
                pose_kp = person["pose_keypoints_2d"]

                if isinstance(pose_kp, list) and len(pose_kp) > 0:
                    # Unwrap extra nesting: [[[x,y,c], ...]] → [[x,y,c], ...]
                    if (isinstance(pose_kp[0], list) and len(pose_kp[0]) > 0
                            and isinstance(pose_kp[0][0], list)):
                        pose_kp = pose_kp[0]

                    if isinstance(pose_kp[0], list):
                        # Nested array format: [[x, y, conf], ...]
                        keypoints = np.array([[kp[0], kp[1]] for kp in pose_kp])
                    else:
                        # Flat array format: [x, y, conf, x, y, conf, ...]
                        keypoints = np.array(pose_kp).reshape(-1, 3)[:, :2]

                    if keypoints.shape[0] >= 17:
                        return keypoints[:17], "openpose"

        raise ValueError("Invalid OpenPose data format")

    def _extract_from_mask(self, mask: Any, image: Any) -> Tuple[np.ndarray, str]:
        """Extract skeleton from binary mask using geometric fitting."""
        # Convert mask to numpy array
        # This is a placeholder - actual implementation needs proper mask handling
        # For now, create a basic skeleton based on typical proportions

        image_height, image_width = self._get_image_dimensions(image)

        # Create a basic A-pose skeleton centered in image
        center_x = image_width / 2
        center_y = image_height / 2
        height_scale = image_height / 2046  # Scale based on reference height

        # Create COCO-17 keypoints in A-pose
        keypoints = np.zeros((17, 2), dtype=np.float32)

        # Head keypoints
        keypoints[0] = [center_x, center_y - 800 * height_scale]  # nose
        keypoints[1] = [center_x + 30, center_y - 820 * height_scale]  # left_eye
        keypoints[2] = [center_x - 30, center_y - 820 * height_scale]  # right_eye
        keypoints[3] = [center_x + 60, center_y - 810 * height_scale]  # left_ear
        keypoints[4] = [center_x - 60, center_y - 810 * height_scale]  # right_ear

        # Shoulders
        keypoints[5] = [center_x + 150, center_y - 700 * height_scale]  # left_shoulder
        keypoints[6] = [center_x - 150, center_y - 700 * height_scale]  # right_shoulder

        # Elbows (A-pose, arms out)
        keypoints[7] = [center_x + 350, center_y - 400 * height_scale]  # left_elbow
        keypoints[8] = [center_x - 350, center_y - 400 * height_scale]  # right_elbow

        # Wrists
        keypoints[9] = [center_x + 550, center_y - 100 * height_scale]  # left_wrist
        keypoints[10] = [center_x - 550, center_y - 100 * height_scale]  # right_wrist

        # Hips
        keypoints[11] = [center_x + 100, center_y + 50 * height_scale]  # left_hip
        keypoints[12] = [center_x - 100, center_y + 50 * height_scale]  # right_hip

        # Knees
        keypoints[13] = [center_x + 80, center_y + 500 * height_scale]  # left_knee
        keypoints[14] = [center_x - 80, center_y + 500 * height_scale]  # right_knee

        # Ankles
        keypoints[15] = [center_x + 70, center_y + 900 * height_scale]  # left_ankle
        keypoints[16] = [center_x - 70, center_y + 900 * height_scale]  # right_ankle

        return keypoints, "mask"

    def _calculate_proportions(self, keypoints_2d: np.ndarray) -> Dict[str, float]:
        """Calculate body proportions from 2D keypoints."""
        # Helper function to calculate distance between indexed joints
        def dist(p1, p2):
            return np.linalg.norm(keypoints_2d[p1] - keypoints_2d[p2])

        # Helper for distance between already-computed points
        def dist_pts(pt1, pt2):
            return np.linalg.norm(pt1 - pt2)

        mid_shoulder = (keypoints_2d[5] + keypoints_2d[6]) / 2
        mid_hip = (keypoints_2d[11] + keypoints_2d[12]) / 2

        proportions = {
            "shoulder_width": dist(5, 6),  # left_shoulder to right_shoulder
            "upper_arm_length": dist(5, 7),  # left_shoulder to left_elbow
            "forearm_length": dist(7, 9),  # left_elbow to left_wrist
            "torso_length": dist_pts(mid_shoulder, mid_hip),  # shoulders to hips
            "upper_leg_length": dist(11, 13),  # left_hip to left_knee
            "lower_leg_length": dist(13, 15),  # left_knee to left_ankle
            "head_height": dist_pts(keypoints_2d[0], mid_shoulder),  # nose to shoulders
        }

        # Total body height
        proportions["body_height"] = (
            proportions["head_height"] +
            proportions["torso_length"] +
            proportions["upper_leg_length"] +
            proportions["lower_leg_length"]
        )

        return proportions

    def _build_canonical_3d_pose(
        self,
        keypoints_2d: np.ndarray,
        proportions: Dict[str, float]
    ) -> np.ndarray:
        """Build canonical 3D A-pose from 2D keypoints."""
        # Create 3D skeleton from 2D, adding depth dimension
        keypoints_3d = np.zeros((17, 3), dtype=np.float32)

        # X, Y from 2D keypoints (centered)
        center_x = keypoints_2d[:, 0].mean()
        center_y = keypoints_2d[:, 1].mean()

        keypoints_3d[:, 0] = keypoints_2d[:, 0] - center_x  # X (horizontal)
        keypoints_3d[:, 1] = keypoints_2d[:, 1] - center_y  # Y (vertical)
        keypoints_3d[:, 2] = 0  # Z (depth) - flat for A-pose

        # Add slight depth variation for realism
        # Shoulders slightly back
        keypoints_3d[5, 2] = -10  # left_shoulder
        keypoints_3d[6, 2] = -10  # right_shoulder

        return keypoints_3d

    def _get_view_angle(
        self,
        preset: str,
        rot_x: float,
        rot_y: float,
        rot_z: float
    ) -> Dict[str, Any]:
        """Get view angle from preset or custom rotation."""
        presets = {
            "front": [0, 0, 0],
            "front_3/4_left": [0, -45, 0],
            "front_3/4_right": [0, 45, 0],
            "side_left": [0, -90, 0],
            "side_right": [0, 90, 0],
            "back": [0, 180, 0],
            "back_3/4_left": [0, -135, 0],
            "back_3/4_right": [0, 135, 0],
            "top_down": [-90, 0, 0],
        }

        if preset == "custom":
            rotation_deg = [rot_x, rot_y, rot_z]
        else:
            rotation_deg = presets.get(preset, [0, 0, 0])

        return {
            "preset": preset,
            "rotation_deg": rotation_deg,
        }

    def _project_to_viewpoint(
        self,
        keypoints_3d: np.ndarray,
        view_angle: Dict[str, Any],
        width: int,
        height: int
    ) -> np.ndarray:
        """Project 3D skeleton to 2D at target viewpoint."""
        rotation_deg = view_angle["rotation_deg"]

        # Convert to radians
        rx, ry, rz = np.radians(rotation_deg)

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])

        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        # Combined rotation
        R = Rz @ Ry @ Rx

        # Apply rotation
        rotated = keypoints_3d @ R.T

        # Simple orthographic projection (X, Y) + recenter
        projected_2d = rotated[:, :2]
        projected_2d[:, 0] += width / 2
        projected_2d[:, 1] += height / 2

        return projected_2d

    def _format_as_dwposes(
        self,
        keypoints_2d: np.ndarray,
        width: int,
        height: int
    ) -> Dict[str, Any]:
        """Format keypoints as DWPOSES dictionary."""
        # Flatten keypoints with confidence scores
        pose_kp = []
        for kp in keypoints_2d:
            pose_kp.extend([float(kp[0]), float(kp[1]), 1.0])  # x, y, confidence

        dwposes = [{
            "people": [{
                "pose_keypoints_2d": pose_kp,
            }],
            "canvas_width": width,
            "canvas_height": height,
        }]

        return dwposes

    def _get_image_dimensions(self, image: Any) -> Tuple[int, int]:
        """Get image dimensions from ComfyUI IMAGE tensor."""
        # ComfyUI images are typically (batch, height, width, channels)
        if hasattr(image, 'shape'):
            if len(image.shape) == 4:
                return image.shape[1], image.shape[2]
            elif len(image.shape) == 3:
                return image.shape[0], image.shape[1]
        # Default fallback
        return 2046, 946

    def _create_debug_visualization(
        self,
        image: Any,
        keypoints_front: np.ndarray,
        keypoints_projected: np.ndarray
    ) -> Any:
        """Create debug visualization showing fitted skeleton."""
        # Placeholder - return input image for now
        # Full implementation would draw skeleton overlay
        return image
