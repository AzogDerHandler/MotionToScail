"""
AddFaceAndHands Node
Enriches SCAIL pose data with face and hand keypoints.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List


class AddFaceAndHands:
    """
    Enrich pose data with face and hand keypoints.

    3-tier priority:
    1. External motion data (if provided)
    2. Reference image DWPose (if available)
    3. Geometric synthesis (fallback)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED",),
            },
            "optional": {
                "ref_dw_pose": ("DWPOSES",),
                "face_motion": ("DWPOSES",),
                "hand_motion": ("DWPOSES",),
                "enable_face": ("BOOLEAN", {"default": True}),
                "enable_hands": ("BOOLEAN", {"default": True}),
                "temporal_match": (["stretch", "loop", "hold_last"],),
            }
        }

    RETURN_TYPES = ("NLFPRED", "DWPOSES")
    RETURN_NAMES = ("nlf_poses", "dw_poses")
    FUNCTION = "add_face_hands"
    CATEGORY = "MotionToSCAIL/Core"

    def add_face_hands(
        self,
        nlf_poses: Dict[str, Any],
        ref_dw_pose: Optional[Dict] = None,
        face_motion: Optional[Dict] = None,
        hand_motion: Optional[Dict] = None,
        enable_face: bool = True,
        enable_hands: bool = True,
        temporal_match: str = "stretch",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Add face and hand keypoints to pose data.

        Returns:
            (enriched_nlf_poses, dw_poses_formatted)
        """
        # Extract body joints from NLF poses
        joints3d_list = nlf_poses["joints3d_nonparam"]
        num_frames = len(joints3d_list)

        print(f"Adding face/hands to {num_frames} frames")
        print(f"  Face: {enable_face}, Hands: {enable_hands}")

        # Initialize face and hand keypoints
        face_keypoints = None
        hand_left_keypoints = None
        hand_right_keypoints = None

        # Process face keypoints (3-tier cascade)
        if enable_face:
            if face_motion is not None:
                # Tier 1: External face motion
                face_keypoints = self._extract_external_face(face_motion, num_frames, temporal_match)
                print("  Face: Using external motion data")
            elif ref_dw_pose is not None:
                # Tier 2: Reference image DWPose
                face_keypoints = self._extract_reference_face(ref_dw_pose, num_frames, joints3d_list)
                if face_keypoints is not None:
                    print("  Face: Using reference DWPose template")

            if face_keypoints is None:
                # Tier 3: Geometric synthesis
                face_keypoints = self._synthesize_face(num_frames, joints3d_list)
                print("  Face: Using geometric synthesis")

        # Process hand keypoints (3-tier cascade)
        if enable_hands:
            if hand_motion is not None:
                # Tier 1: External hand motion
                hand_left_keypoints, hand_right_keypoints = self._extract_external_hands(
                    hand_motion, num_frames, temporal_match
                )
                print("  Hands: Using external motion data")
            elif ref_dw_pose is not None:
                # Tier 2: Reference image DWPose
                hands = self._extract_reference_hands(ref_dw_pose, num_frames, joints3d_list)
                if hands is not None:
                    hand_left_keypoints, hand_right_keypoints = hands
                    print("  Hands: Using reference DWPose template")

            if hand_left_keypoints is None:
                # Tier 3: Geometric synthesis
                hand_left_keypoints, hand_right_keypoints = self._synthesize_hands(
                    num_frames, joints3d_list
                )
                print("  Hands: Using geometric synthesis")

        # Format as DWPOSES
        dw_poses = self._format_as_dwposes(
            num_frames, face_keypoints, hand_left_keypoints, hand_right_keypoints
        )

        # NLF poses remain unchanged (face/hands are added via dw_poses input to renderer)
        enriched_nlf = nlf_poses.copy()

        return (enriched_nlf, dw_poses)

    def _extract_external_face(
        self,
        face_motion: Dict,
        num_frames: int,
        temporal_match: str
    ) -> np.ndarray:
        """Extract and temporally match external face motion data."""
        # Placeholder - would extract 68-point face data and resample
        return np.zeros((num_frames, 68, 2), dtype=np.float32)

    def _extract_reference_face(
        self,
        ref_dw_pose: Dict,
        num_frames: int,
        joints3d_list: List
    ) -> Optional[np.ndarray]:
        """Extract face template from reference DWPose and animate it."""
        # Check if reference has face keypoints
        if isinstance(ref_dw_pose, list) and len(ref_dw_pose) > 0:
            person = ref_dw_pose[0]
            if "people" in person and len(person["people"]) > 0:
                person_data = person["people"][0]
                if "face_keypoints_2d" in person_data:
                    # Extract face template
                    face_template = np.array(person_data["face_keypoints_2d"]).reshape(-1, 3)[:, :2]

                    # Replicate for all frames and place at head position
                    face_seq = np.tile(face_template, (num_frames, 1, 1))

                    # TODO: Adjust position/rotation based on body head joint movement

                    return face_seq

        return None

    def _synthesize_face(self, num_frames: int, joints3d_list: List) -> np.ndarray:
        """Synthesize generic 68-point face from head joint positions."""
        face_keypoints = np.zeros((num_frames, 68, 2), dtype=np.float32)

        for i, joints in enumerate(joints3d_list):
            # Get head position (approximate from nose, eyes, ears)
            joints_frame = joints[0]  # (17, 3)
            nose_pos = joints_frame[0, :2]  # Take X, Y

            # Create generic face centered at nose
            # Multi-PIE 68-point face template (simplified)
            face_template = self._create_generic_face_template()

            # Scale and position
            face_keypoints[i] = face_template + nose_pos

        return face_keypoints

    def _create_generic_face_template(self) -> np.ndarray:
        """Create generic 68-point face template."""
        # Simplified template - full implementation would use proper Multi-PIE layout
        template = np.zeros((68, 2), dtype=np.float32)

        # Face outline (17 points: 0-16)
        for i in range(17):
            angle = -np.pi/2 + (i / 16) * np.pi
            template[i] = [np.cos(angle) * 50, np.sin(angle) * 60]

        # Eyebrows (10 points: 17-26)
        # Eyes (12 points: 36-47)
        # Nose (9 points: 27-35)
        # Mouth (20 points: 48-67)

        # TODO: Fill in remaining points with proper template

        return template

    def _extract_external_hands(
        self,
        hand_motion: Dict,
        num_frames: int,
        temporal_match: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and temporally match external hand motion data."""
        # Placeholder
        left = np.zeros((num_frames, 21, 2), dtype=np.float32)
        right = np.zeros((num_frames, 21, 2), dtype=np.float32)
        return left, right

    def _extract_reference_hands(
        self,
        ref_dw_pose: Dict,
        num_frames: int,
        joints3d_list: List
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract hand templates from reference DWPose."""
        if isinstance(ref_dw_pose, list) and len(ref_dw_pose) > 0:
            person = ref_dw_pose[0]
            if "people" in person and len(person["people"]) > 0:
                person_data = person["people"][0]

                has_left = "hand_left_keypoints_2d" in person_data
                has_right = "hand_right_keypoints_2d" in person_data

                if has_left or has_right:
                    left_template = None
                    right_template = None

                    if has_left:
                        left_template = np.array(person_data["hand_left_keypoints_2d"]).reshape(-1, 3)[:, :2]

                    if has_right:
                        right_template = np.array(person_data["hand_right_keypoints_2d"]).reshape(-1, 3)[:, :2]

                    # Replicate and position
                    if left_template is not None:
                        left_seq = np.tile(left_template, (num_frames, 1, 1))
                    else:
                        left_seq = self._synthesize_single_hand(num_frames, joints3d_list, "left")

                    if right_template is not None:
                        right_seq = np.tile(right_template, (num_frames, 1, 1))
                    else:
                        right_seq = self._synthesize_single_hand(num_frames, joints3d_list, "right")

                    return left_seq, right_seq

        return None

    def _synthesize_hands(
        self,
        num_frames: int,
        joints3d_list: List
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synthesize generic 21-point hands from wrist positions."""
        left_hand = self._synthesize_single_hand(num_frames, joints3d_list, "left")
        right_hand = self._synthesize_single_hand(num_frames, joints3d_list, "right")
        return left_hand, right_hand

    def _synthesize_single_hand(
        self,
        num_frames: int,
        joints3d_list: List,
        side: str
    ) -> np.ndarray:
        """Synthesize single hand (21 points)."""
        hand_keypoints = np.zeros((num_frames, 21, 2), dtype=np.float32)
        wrist_idx = 9 if side == "left" else 10

        for i, joints in enumerate(joints3d_list):
            joints_frame = joints[0]  # (17, 3)
            wrist_pos = joints_frame[wrist_idx, :2]  # X, Y

            # Create generic hand template
            hand_template = self._create_generic_hand_template()

            # Position at wrist
            hand_keypoints[i] = hand_template + wrist_pos

        return hand_keypoints

    def _create_generic_hand_template(self) -> np.ndarray:
        """Create generic 21-point hand template (MediaPipe format)."""
        template = np.zeros((21, 2), dtype=np.float32)

        # Wrist
        template[0] = [0, 0]

        # Thumb (4 points)
        template[1:5] = [[5, -10], [10, -15], [15, -18], [20, -20]]

        # Index finger (4 points)
        template[5:9] = [[8, -25], [10, -35], [11, -45], [12, -55]]

        # Middle finger (4 points)
        template[9:13] = [[0, -25], [0, -38], [0, -50], [0, -62]]

        # Ring finger (4 points)
        template[13:17] = [[-8, -23], [-10, -34], [-11, -44], [-12, -53]]

        # Pinky (4 points)
        template[17:21] = [[-15, -18], [-18, -26], [-20, -33], [-22, -39]]

        return template

    def _format_as_dwposes(
        self,
        num_frames: int,
        face: Optional[np.ndarray],
        hand_left: Optional[np.ndarray],
        hand_right: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Format face/hand data as DWPOSES for renderer."""
        # Create frame-by-frame DWPOSES structure
        dw_poses_sequence = []

        for i in range(num_frames):
            frame_data = {"people": [{}]}

            if face is not None:
                # Flatten with confidence
                face_kp = []
                for kp in face[i]:
                    face_kp.extend([float(kp[0]), float(kp[1]), 1.0])
                frame_data["people"][0]["face_keypoints_2d"] = face_kp

            if hand_left is not None:
                hand_left_kp = []
                for kp in hand_left[i]:
                    hand_left_kp.extend([float(kp[0]), float(kp[1]), 1.0])
                frame_data["people"][0]["hand_left_keypoints_2d"] = hand_left_kp

            if hand_right is not None:
                hand_right_kp = []
                for kp in hand_right[i]:
                    hand_right_kp.extend([float(kp[0]), float(kp[1]), 1.0])
                frame_data["people"][0]["hand_right_keypoints_2d"] = hand_right_kp

            dw_poses_sequence.append(frame_data)

        return dw_poses_sequence
