"""ComfyUI custom nodes for FoundationStereo — stereoscopic depth estimation and VFX warping."""

import os
import sys
import types

# FoundationStereo's Utils.py unconditionally imports heavy libraries (trimesh,
# pandas, joblib, open3d) that aren't needed for inference and may not be
# installed. Inject lightweight stubs so the import chain doesn't crash.
for _mod_name in ("trimesh", "joblib", "pandas"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

# open3d is only needed at runtime by the PointCloud node (lazy-imported there).
# Stub it here so Utils.py's top-level `import open3d as o3d` doesn't fail,
# but only if it isn't already installed.
if "open3d" not in sys.modules:
    _o3d_stub = types.ModuleType("open3d")
    _o3d_stub.geometry = types.ModuleType("open3d.geometry")
    _o3d_stub.utility = types.ModuleType("open3d.utility")
    _o3d_stub.io = types.ModuleType("open3d.io")
    _o3d_stub.visualization = types.ModuleType("open3d.visualization")
    sys.modules["open3d"] = _o3d_stub
    sys.modules["open3d.geometry"] = _o3d_stub.geometry
    sys.modules["open3d.utility"] = _o3d_stub.utility
    sys.modules["open3d.io"] = _o3d_stub.io
    sys.modules["open3d.visualization"] = _o3d_stub.visualization

# Add the FoundationStereo submodule to sys.path so its internal imports work
_FOUNDATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "FoundationStereo"))
if _FOUNDATION_DIR not in sys.path:
    sys.path.insert(0, _FOUNDATION_DIR)

# Add this package root so `from stereo_utils import ...` works inside node modules
_PACKAGE_DIR = os.path.dirname(__file__)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

from foundation_stereo_nodes.model_loader import FoundationStereoModelLoader
from foundation_stereo_nodes.run_inference import FoundationStereoRun
from foundation_stereo_nodes.stereo_warp import StereoWarp
from foundation_stereo_nodes.occlusion_mask import OcclusionMask
from foundation_stereo_nodes.disparity_to_depth import DisparityToDepth
from foundation_stereo_nodes.depth_visualization import DepthVisualization
from foundation_stereo_nodes.depth_to_pointcloud import DepthToPointCloud

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
