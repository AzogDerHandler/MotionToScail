"""
ComfyUI-MotionToSCAIL
A ComfyUI custom node package for bridging motion data to SCAIL/Wan 2.1
"""

from .nodes.humanml3d_to_coco3d import HumanML3DToCOCO3D
from .nodes.smpl_to_coco3d import SMPLToCOCO3D
from .nodes.motion_fps_convert import MotionFPSConvert
from .nodes.motion_positioner import MotionPositioner
from .nodes.motion_blend_in import MotionBlendIn
from .nodes.ref_skeleton_extractor import RefSkeletonExtractor
from .nodes.coco3d_to_scail_pose import COCO3DToSCAILPose
from .nodes.add_face_and_hands import AddFaceAndHands

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "HumanML3DToCOCO3D": HumanML3DToCOCO3D,
    "SMPLToCOCO3D": SMPLToCOCO3D,
    "MotionFPSConvert": MotionFPSConvert,
    "MotionPositioner": MotionPositioner,
    "MotionBlendIn": MotionBlendIn,
    "RefSkeletonExtractor": RefSkeletonExtractor,
    "COCO3DToSCAILPose": COCO3DToSCAILPose,
    "AddFaceAndHands": AddFaceAndHands,
}

# Display names for ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "HumanML3DToCOCO3D": "HumanML3D to COCO3D",
    "SMPLToCOCO3D": "SMPL to COCO3D",
    "MotionFPSConvert": "Motion FPS Convert",
    "MotionPositioner": "Motion Positioner",
    "MotionBlendIn": "Motion Blend In",
    "RefSkeletonExtractor": "Reference Skeleton Extractor",
    "COCO3DToSCAILPose": "COCO3D to SCAIL Pose",
    "AddFaceAndHands": "Add Face and Hands",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
