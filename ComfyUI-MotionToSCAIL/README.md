# ComfyUI-MotionToSCAIL

A ComfyUI custom node package that enables pre-generated motion data (from GVHMR, HY-Motion, MotionDiffuse) to drive SCAIL (Wan 2.1) character animation, bypassing SCAIL's native video-based pose extraction pipeline.

## Features

- **Multiple Motion Sources**: Support for SMPL (GVHMR), HumanML3D (MotionDiffuse, HY-Motion)
- **Motion Processing**: FPS conversion, looping, walk-in-place, blend-in transitions
- **Character Retargeting**: Automatic skeleton scaling to match character proportions
- **Pose Data Extraction**: DWPose, OpenPose, or mask-based skeleton fitting
- **Face & Hand Enrichment**: Optional face and hand keypoint synthesis or integration

## Installation

### Method 1: Git Clone (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-MotionToSCAIL.git
cd ComfyUI-MotionToSCAIL
pip install -r requirements.txt
```

### Method 2: Manual Installation

1. Download this repository
2. Extract to `ComfyUI/custom_nodes/ComfyUI-MotionToSCAIL`
3. Install dependencies: `pip install -r requirements.txt`

## Node Categories

### MotionToSCAIL/Converters

- **HumanML3DToCOCO3D**: Convert HumanML3D motion (MotionDiffuse, HY-Motion) to COCO3D format
- **SMPLToCOCO3D**: Convert SMPL parameters (GVHMR) to COCO3D format

### MotionToSCAIL/Processing

- **MotionFPSConvert**: Resample motion to target FPS
- **MotionPositioner**: Control root motion (walk-in-place) and looping
- **MotionBlendIn**: Create smooth transition from A-pose to animation

### MotionToSCAIL/Reference

- **RefSkeletonExtractor**: Extract character skeleton from reference images

### MotionToSCAIL/Core

- **COCO3DToSCAILPose**: Convert COCO3D motion to SCAIL-compatible pose format
- **AddFaceAndHands**: Add face and hand keypoints to poses

## Basic Workflow

### Text-to-Motion → SCAIL Pipeline

```
MotionDiffuse Output (HumanML3D)
  ↓
HumanML3DToCOCO3D
  ↓
MotionFPSConvert (match SCAIL FPS)
  ↓
MotionPositioner (walk-in-place/loop)
  ↓
[Reference Image + Mask] → RefSkeletonExtractor
  ↓
COCO3DToSCAILPose ← [COCO3D Motion + RefSkeleton]
  ↓
(Optional) AddFaceAndHands
  ↓
RenderNLFPoses (Kijai node)
  ↓
WanVideoAddSCAILPoseEmbeds
  ↓
SCAIL Generation
```

## Data Formats

### COCO3D_JOINTS (Internal Format)

Standardized intermediate format for all motion data:

```python
{
    "joints_3d": np.ndarray,  # (num_frames, 17, 3)
    "fps": float,
    "source_format": "smpl" | "humanml3d"
}
```

### REF_SKELETON

Character proportions and reference skeleton:

```python
{
    "keypoints_2d": np.ndarray,  # (17, 2)
    "keypoints_3d": np.ndarray,  # (17, 3)
    "body_height": float,
    "shoulder_width": float,
    # ... limb lengths
    "view_angle": dict,
    "source": "dwpose" | "openpose" | "mask"
}
```

## Example Workflows

See `example_workflows/` directory for complete workflow JSON files:

- `text_to_motion_scail.json`: MotionDiffuse → SCAIL
- `video_to_motion_scail.json`: GVHMR → SCAIL
- `full_pipeline_face_hands.json`: Complete pipeline with face/hand enrichment

## SMPL Model Setup (Optional)

For accurate SMPL forward kinematics:

1. Install smplx: `pip install smplx`
2. Download SMPL model from https://smpl.is.tue.mpg.de/
3. Place `SMPL_NEUTRAL.pkl` in `./models/smpl/`

Without smplx, a simplified approximation will be used.

## Troubleshooting

### "Motion data format not recognized"
- Ensure motion data is in the expected format (dict with 'motion' key for HumanML3D)
- Check tensor shapes match documentation

### "RefSkeletonExtractor failed"
- Verify reference image has clear character silhouette
- Try providing DWPose or OpenPose detection results
- Ensure mask is clean binary mask

### "SMPL conversion inaccurate"
- Install smplx library and download SMPL model
- Simplified FK is used as fallback but may be less accurate

## Credits

- Based on SCAIL/Wan 2.1 architecture
- Compatible with Kijai's ComfyUI-KJNodes
- Motion formats: SMPL, HumanML3D
- Pose detection: DWPose, OpenPose

## License

MIT License - See LICENSE file for details

## Citation

If you use this in your research, please cite:

```bibtex
@software{comfyui_motiontoscail,
  title={ComfyUI-MotionToSCAIL},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ComfyUI-MotionToSCAIL}
}
```

## Contributing

Contributions welcome! Please open an issue or pull request.

## Roadmap

- [ ] Multi-person support
- [ ] Manual skeleton override node
- [ ] Improved SMPL FK without smplx dependency
- [ ] Better mask-based skeleton fitting
- [ ] Advanced face expression mapping
- [ ] Hand gesture library
