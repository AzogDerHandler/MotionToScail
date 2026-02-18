# Quick Start Guide - ComfyUI-MotionToSCAIL

## Installation

```bash
# Navigate to ComfyUI custom nodes directory
cd /path/to/ComfyUI/custom_nodes

# Copy the package
cp -r /path/to/ComfyUI-MotionToSCAIL ./

# Install dependencies in your ComfyUI Python environment
# (Activate your venv/conda env first if using one)
pip install numpy scipy
```

## Data Flow Overview

```
┌─────────────────────────────────────────────────────────┐
│               MOTION SOURCE                             │
├─────────────────────────────────────────────────────────┤
│  MotionDiffuse    │    GVHMR/SMPL                      │
│  (HumanML3D)      │    (SMPL params)                   │
└────────┬──────────┴──────────┬─────────────────────────┘
         │                     │
         ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│ HumanML3DToCOCO3D│  │  SMPLToCOCO3D    │
└────────┬─────────┘  └────────┬─────────┘
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │   COCO3D_JOINTS     │ ← Central format (17 joints, 3D)
         └─────────┬───────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│ MotionFPSConvert │  │ MotionPositioner │
│ (optional)       │  │ (optional)       │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └─────────┬───────────┘
                   ▼
         ┌──────────────────┐
         │  MotionBlendIn   │ (optional)
         └────────┬─────────┘
                  │
                  ▼
         ┌─────────────────────┐
         │   COCO3D_JOINTS     │
         │   (processed)       │
         └─────────┬───────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    │              ▼              │
    │   ┌───────────────────┐    │
    │   │ RefSkeletonExtract│    │
    │   │ (from ref image)  │    │
    │   └─────────┬─────────┘    │
    │             │               │
    │             ▼               │
    │      REF_SKELETON           │
    │                             │
    └──────────────┬──────────────┘
                   │
                   ▼
         ┌──────────────────────┐
         │ COCO3DToSCAILPose    │ ← Main bridge
         └─────────┬────────────┘
                   │
                   ▼
         ┌──────────────────────┐
         │     NLFPRED          │
         └─────────┬────────────┘
                   │
                   ▼
         ┌──────────────────────┐
         │  AddFaceAndHands     │ (optional)
         └─────────┬────────────┘
                   │
                   ▼
         ┌──────────────────────┐
         │  NLFPRED + DWPOSES   │
         └─────────┬────────────┘
                   │
                   ▼
         ┌──────────────────────┐
         │  RenderNLFPoses      │ ← Kijai's node
         │  (cylinder renderer) │
         └─────────┬────────────┘
                   │
                   ▼
         ┌──────────────────────┐
         │ WanVideoAddSCAILPose │
         │      Embeds          │
         └─────────┬────────────┘
                   │
                   ▼
         ┌──────────────────────┐
         │   SCAIL Generation   │
         └──────────────────────┘
```

## Minimal Workflow (MotionDiffuse → SCAIL)

### Step 1: Load Your Motion Data
Your MotionDiffuse output should be a dictionary like:
```python
{
    'motion': tensor([196, 263]),  # 196 frames, 263 features
    'motion_mask': tensor([1, 196]),
    'motion_length': tensor([196])
}
```

### Step 2: Convert to COCO3D
**Node**: `HumanML3DToCOCO3D`
- Input: motion_data
- Input: fps = 20.0
- Output: coco3d_joints

### Step 3: Match SCAIL FPS
**Node**: `MotionFPSConvert`
- Input: coco3d_joints (from step 2)
- Input: target_fps = 16.0 (SCAIL default)
- Output: coco3d_joints (resampled)

### Step 4: Extract Reference Skeleton
**Node**: `RefSkeletonExtractor`
- Input: front_image (your character reference)
- Input: front_mask (character mask)
- Input: actual_ref_image (same or different view)
- Input: dw_pose_front (your dw-pose-front.json data)
- Input: view_preset = "front"
- Output: ref_skeleton

### Step 5: Convert to SCAIL Format
**Node**: `COCO3DToSCAILPose`
- Input: coco3d_joints (from step 3)
- Input: ref_skeleton (from step 4)
- Input: target_width = 832
- Input: target_height = 480
- Output: nlf_poses

### Step 6: Render and Generate
**Node**: `RenderNLFPoses` (Kijai)
- Input: nlf_poses
- Output: pose_images

**Node**: `WanVideoAddSCAILPoseEmbeds`
- Input: pose_images
- → Continue with SCAIL generation

## Full Workflow (with all options)

### Motion Processing
```
HumanML3DToCOCO3D
  ↓
MotionFPSConvert (16 FPS)
  ↓
MotionPositioner
  - root_motion: "walk_in_place"
  - playback_mode: "loop"
  - target_frames: 81
  - loop_blend_frames: 5
  ↓
MotionBlendIn
  - blend_frames: 8
  - easing: "ease_out"
```

### Reference Setup
```
RefSkeletonExtractor
  - front_image: [your A-pose image]
  - front_mask: [clean mask]
  - actual_ref_image: [animation reference view]
  - dw_pose_front: [DWPose detection]
  - view_preset: "front_3/4_left"
```

### SCAIL Conversion
```
COCO3DToSCAILPose
  - coco3d_joints: [processed motion]
  - ref_skeleton: [from extractor]
  - target_width: 832
  - target_height: 480
  ↓
AddFaceAndHands
  - nlf_poses: [from above]
  - ref_dw_pose: [from RefSkeletonExtractor]
  - enable_face: true
  - enable_hands: true
```

## Node Parameters Quick Reference

### HumanML3DToCOCO3D
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| motion_data | MOTION_DATA | required | HumanML3D format motion |
| fps | float | 20.0 | Source FPS |

### SMPLToCOCO3D
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| smpl_data | SMPL_DATA | required | SMPL parameters |
| fps | float | 30.0 | Source FPS |
| use_global_coord | bool | true | Use global vs camera coords |

### MotionFPSConvert
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| coco3d_joints | COCO3D_JOINTS | required | Motion to resample |
| target_fps | float | 16.0 | Target FPS |
| interpolation | enum | "linear" | linear/cubic/nearest |

### MotionPositioner
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| coco3d_joints | COCO3D_JOINTS | required | Motion to process |
| root_motion | enum | required | walk_in_place/walk_through_scene |
| playback_mode | enum | required | one_shot/loop/ping_pong |
| target_frames | int | 0 | Target frame count (0=unchanged) |
| loop_blend_frames | int | 5 | Blend frames at loop boundary |
| scene_offset_x | float | 0.0 | X offset for walk_through_scene |
| scene_offset_z | float | 0.0 | Z offset for walk_through_scene |

### MotionBlendIn
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| ref_skeleton | REF_SKELETON | required | Reference A-pose |
| target_motion | COCO3D_JOINTS | required | Motion to blend into |
| blend_frames | int | 8 | Transition frame count |
| easing | enum | "ease_out" | linear/ease_in/ease_out/ease_in_out |

### RefSkeletonExtractor
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| front_image | IMAGE | required | Front-view A-pose image |
| front_mask | MASK | required | Character mask |
| actual_ref_image | IMAGE | required | Actual reference view |
| actual_ref_mask | MASK | optional | Reference mask |
| dw_pose_front | DWPOSES | optional | DWPose detection (recommended) |
| openpose_front | POSE_KEYPOINT | optional | OpenPose detection |
| view_preset | enum | "front" | View angle preset |
| rotation_x/y/z | float | 0.0 | Custom rotation (if preset="custom") |

### COCO3DToSCAILPose
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| coco3d_joints | COCO3D_JOINTS | required | Processed motion |
| ref_skeleton | REF_SKELETON | required | Character skeleton |
| target_width | int | 832 | Generation width |
| target_height | int | 480 | Generation height |

### AddFaceAndHands
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| nlf_poses | NLFPRED | required | Body poses |
| ref_dw_pose | DWPOSES | optional | Reference pose (recommended) |
| face_motion | DWPOSES | optional | External face motion |
| hand_motion | DWPOSES | optional | External hand motion |
| enable_face | bool | true | Add face keypoints |
| enable_hands | bool | true | Add hand keypoints |
| temporal_match | enum | "stretch" | stretch/loop/hold_last |

## Common Issues & Solutions

### Issue: "Motion data format not recognized"
**Solution**: Ensure your motion data is a dictionary with 'motion' key (HumanML3D) or 'global'/'incam' keys (SMPL)

### Issue: "RefSkeletonExtractor failed"
**Solution**:
1. Provide DWPose or OpenPose data (priority 1 & 2)
2. Ensure mask is clean binary mask
3. Check image dimensions are reasonable

### Issue: "SMPL conversion inaccurate"
**Solution**:
```bash
pip install smplx
# Download SMPL model and place in ./models/smpl/
```

### Issue: "Output FPS doesn't match SCAIL"
**Solution**: Use MotionFPSConvert with target_fps=16.0 (SCAIL default)

### Issue: "Loop has visible seam"
**Solution**: Increase loop_blend_frames in MotionPositioner (try 10-15)

## Tips for Best Results

1. **Always use DWPose if available** - Best quality for RefSkeletonExtractor
2. **Match SCAIL FPS** - Use MotionFPSConvert to set 16 FPS
3. **Use walk-in-place** - More stable than walk-through for most cases
4. **Add blend-in frames** - Smoother start transition (8-12 frames recommended)
5. **Enable face/hands** - Better quality if you have ref_dw_pose with face/hand data

## Example Values for Your Test Data

### For motiondiff-motiondata:
```
HumanML3DToCOCO3D:
  motion_data: [load from file]
  fps: 20.0

MotionFPSConvert:
  target_fps: 16.0
  → Output: ~157 frames (from 196 at 20 FPS to ~157 at 16 FPS for same duration)
```

### For smpl-motiondata.smpl:
```
SMPLToCOCO3D:
  smpl_data: [load from file]
  fps: 30.0
  use_global_coord: true

MotionFPSConvert:
  target_fps: 16.0
  → Output: ~27 frames (from 50 at 30 FPS to ~27 at 16 FPS)
```

### For dw-pose-front.json:
```
RefSkeletonExtractor:
  dw_pose_front: [load from file]
  view_preset: "front"
  → Will use DWPose (Tier 1, best quality)
  → Has face (68 points) and hands (21×2 points) for AddFaceAndHands
```

## Next Steps

1. Copy package to ComfyUI custom_nodes
2. Restart ComfyUI
3. Create workflow with nodes from "MotionToSCAIL" category
4. Test with your motion data
5. Verify output with SCAIL generation

## Support

Check console output for helpful debug messages. Each node prints its progress and any issues encountered.
