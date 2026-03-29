# ComfyUI-FoundationStereo

ComfyUI custom nodes wrapping [NVIDIA FoundationStereo](https://github.com/NVlabs/FoundationStereo) for stereoscopic VFX workflows — accurate disparity estimation, stereo warping, and 3D point cloud generation.

## Nodes

| Node | Description |
|------|-------------|
| **FoundationStereo Model Loader** | Load and cache a FoundationStereo checkpoint |
| **FoundationStereo Run** | Estimate disparity from a rectified stereo image pair |
| **Stereo Warp** | Warp left-eye image to right-eye view (or vice versa) using disparity |
| **Occlusion Mask** | Detect disoccluded regions for inpainting |
| **Disparity to Depth** | Convert disparity to metric depth (meters) |
| **Depth Visualization** | Apply pseudocolor colormaps (turbo, viridis, etc.) |
| **Depth to Point Cloud** | Export 3D `.ply` point cloud from depth + color |

## Installation

### 1. Clone into ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/lightsailvr/ComfyUI-FoundationStereo.git
```

If you already cloned without `--recursive`:
```bash
cd ComfyUI-FoundationStereo
git submodule update --init
```

### 2. Install dependencies

```bash
cd ComfyUI/custom_nodes/ComfyUI-FoundationStereo
pip install -r requirements.txt
```

### 3. Model weights (auto-downloaded)

**Models download automatically** from HuggingFace the first time you use the Model Loader node. No manual setup needed.

Available models:
- **23-51-11** (ViT-Large, ~3.3 GB) — Best quality, recommended
- **11-33-40** (ViT-Small, ~788 MB) — Faster inference

Weights are cached in `FoundationStereo/pretrained_models/` so they only download once.

If you prefer to download manually, get them from [Google Drive](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing) and place them in:
```
ComfyUI-FoundationStereo/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
ComfyUI-FoundationStereo/FoundationStereo/pretrained_models/23-51-11/cfg.yaml
```

### 4. Restart ComfyUI

Nodes will appear under the **FoundationStereo** category.

## VFX Stereo Workflow

The primary use case is warping left-eye VFX work (paint-outs, roto, etc.) into the right eye:

```
LoadImage (left)  ──┐
                    ├──→ FoundationStereo Run ──→ disparity_raw
LoadImage (right) ──┘           │
                                │
LoadImage (left VFX comp) ──────┤
                                ↓
                          Stereo Warp ──→ warped_right_vfx
                                │
                                └──→ occlusion_mask
                                         │
                                         ↓
                              ComfyUI Inpaint ──→ final right eye
```

1. Load your original stereo pair (rectified, undistorted)
2. Run FoundationStereo to get the disparity map
3. Feed your left-eye VFX comp + disparity into **Stereo Warp**
4. Use the occlusion mask with any ComfyUI inpainting node to fill holes
5. Result: right-eye VFX that matches the left eye

## Input Requirements

- Stereo images must be **rectified and undistorted** (epipolar lines horizontal)
- PNG format recommended (lossless)
- Works with RGB, monochrome, or IR images

## Tips

- **High-res images (>1000px)**: Enable `use_hierarchical` in FoundationStereo Run
- **Memory constrained**: Reduce `scale` to process at lower resolution
- **Faster inference**: Use the `11-33-40` model and lower `iterations` (16-20)
- **Best quality**: Use `23-51-11` model with 32 iterations

## License

This wrapper is MIT licensed. FoundationStereo itself is subject to [NVIDIA's license](https://github.com/NVlabs/FoundationStereo/blob/main/LICENSE).
