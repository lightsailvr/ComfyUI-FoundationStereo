"""ComfyUI custom nodes for FoundationStereo — stereoscopic depth estimation and VFX warping."""

import os
import sys

# Add the FoundationStereo submodule to sys.path so its internal imports work
_FOUNDATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "FoundationStereo"))
if _FOUNDATION_DIR not in sys.path:
    sys.path.insert(0, _FOUNDATION_DIR)

# Add this package root so `from utils import ...` works inside node modules
_PACKAGE_DIR = os.path.dirname(__file__)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

from nodes.model_loader import FoundationStereoModelLoader
from nodes.run_inference import FoundationStereoRun
from nodes.stereo_warp import StereoWarp
from nodes.occlusion_mask import OcclusionMask
from nodes.disparity_to_depth import DisparityToDepth
from nodes.depth_visualization import DepthVisualization
from nodes.depth_to_pointcloud import DepthToPointCloud

NODE_CLASS_MAPPINGS = {
    "FoundationStereoModelLoader": FoundationStereoModelLoader,
    "FoundationStereoRun": FoundationStereoRun,
    "StereoWarp": StereoWarp,
    "OcclusionMask": OcclusionMask,
    "DisparityToDepth": DisparityToDepth,
    "DepthVisualization": DepthVisualization,
    "DepthToPointCloud": DepthToPointCloud,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoundationStereoModelLoader": "FoundationStereo Model Loader",
    "FoundationStereoRun": "FoundationStereo Run",
    "StereoWarp": "Stereo Warp",
    "OcclusionMask": "Occlusion Mask",
    "DisparityToDepth": "Disparity to Depth",
    "DepthVisualization": "Depth Visualization",
    "DepthToPointCloud": "Depth to Point Cloud",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
