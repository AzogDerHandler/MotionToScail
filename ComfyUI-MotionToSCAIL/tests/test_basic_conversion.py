"""
Basic test for motion conversion nodes using the provided test data.
"""

import sys
import os
import json
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.humanml3d_to_coco3d import HumanML3DToCOCO3D
from nodes.smpl_to_coco3d import SMPLToCOCO3D
from nodes.motion_fps_convert import MotionFPSConvert
from nodes.ref_skeleton_extractor import RefSkeletonExtractor


def load_motion_diff_data(filepath):
    """Load MotionDiff data from file."""
    with open(filepath, 'r') as f:
        content = f.read()
        # Parse the Python dict representation
        data = eval(content)
    return data


def load_smpl_data(filepath):
    """Load SMPL data from file."""
    with open(filepath, 'r') as f:
        content = f.read()
        # Parse the Python dict representation
        data = eval(content)
    return data


def load_pose_data(filepath):
    """Load pose data (JSON format)."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def test_humanml3d_conversion():
    """Test HumanML3D to COCO3D conversion."""
    print("\n=== Testing HumanML3D Conversion ===")

    # Load test data
    test_data_path = "../motiondiff-motiondata"
    motion_data = load_motion_diff_data(test_data_path)

    # Create node instance
    node = HumanML3DToCOCO3D()

    # Convert
    result = node.convert(motion_data, fps=20.0)
    coco3d_joints = result[0]

    # Validate output
    assert "joints_3d" in coco3d_joints
    assert "fps" in coco3d_joints
    assert "source_format" in coco3d_joints

    joints = coco3d_joints["joints_3d"]
    print(f"  Output shape: {joints.shape}")
    print(f"  FPS: {coco3d_joints['fps']}")
    print(f"  Source format: {coco3d_joints['source_format']}")

    assert joints.shape[1] == 17, "Should have 17 COCO joints"
    assert joints.shape[2] == 3, "Should have 3D coordinates"

    print("  ✓ HumanML3D conversion successful")
    return coco3d_joints


def test_smpl_conversion():
    """Test SMPL to COCO3D conversion."""
    print("\n=== Testing SMPL Conversion ===")

    # Load test data
    test_data_path = "../smpl-motiondata.smpl"
    smpl_data = load_smpl_data(test_data_path)

    # Create node instance
    node = SMPLToCOCO3D()

    # Convert
    result = node.convert(smpl_data, fps=30.0, use_global_coord=True)
    coco3d_joints = result[0]

    # Validate output
    joints = coco3d_joints["joints_3d"]
    print(f"  Output shape: {joints.shape}")
    print(f"  FPS: {coco3d_joints['fps']}")

    assert joints.shape[1] == 17, "Should have 17 COCO joints"
    assert joints.shape[2] == 3, "Should have 3D coordinates"

    print("  ✓ SMPL conversion successful")
    return coco3d_joints


def test_fps_conversion(coco3d_joints):
    """Test FPS conversion."""
    print("\n=== Testing FPS Conversion ===")

    node = MotionFPSConvert()

    # Convert to 16 FPS (SCAIL default)
    result = node.convert_fps(coco3d_joints, target_fps=16.0, interpolation="linear")
    converted_joints, frame_count = result

    print(f"  Input: {coco3d_joints['joints_3d'].shape[0]} frames @ {coco3d_joints['fps']} FPS")
    print(f"  Output: {frame_count} frames @ {converted_joints['fps']} FPS")

    assert converted_joints["fps"] == 16.0
    assert converted_joints["joints_3d"].shape[0] == frame_count

    print("  ✓ FPS conversion successful")
    return converted_joints


def test_ref_skeleton_extraction():
    """Test reference skeleton extraction."""
    print("\n=== Testing Reference Skeleton Extraction ===")

    # Load DWPose data
    dw_pose_path = "../dw-pose-front.json"
    dw_pose_data = load_pose_data(dw_pose_path)

    # Create dummy image tensors (placeholder)
    dummy_image = torch.zeros((1, 2046, 946, 3))
    dummy_mask = torch.zeros((1, 2046, 946))

    node = RefSkeletonExtractor()

    # Extract skeleton
    result = node.extract_skeleton(
        front_image=dummy_image,
        front_mask=dummy_mask,
        actual_ref_image=dummy_image,
        dw_pose_front=dw_pose_data,
        view_preset="front"
    )

    ref_skeleton, ref_dw_pose, debug_image, source_method = result

    print(f"  Source method: {source_method}")
    print(f"  Body height: {ref_skeleton['body_height']:.2f}px")
    print(f"  Shoulder width: {ref_skeleton['shoulder_width']:.2f}px")
    print(f"  Keypoints 3D shape: {ref_skeleton['keypoints_3d'].shape}")

    assert source_method in ["dwpose", "openpose", "mask"]
    assert ref_skeleton["keypoints_3d"].shape == (17, 3)

    print("  ✓ Reference skeleton extraction successful")
    return ref_skeleton


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running ComfyUI-MotionToSCAIL Tests")
    print("=" * 60)

    try:
        # Test converters
        coco3d_humanml = test_humanml3d_conversion()
        coco3d_smpl = test_smpl_conversion()

        # Test processing
        converted_motion = test_fps_conversion(coco3d_humanml)

        # Test reference extraction
        ref_skeleton = test_ref_skeleton_extraction()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Change to test data directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(test_dir)

    run_all_tests()
