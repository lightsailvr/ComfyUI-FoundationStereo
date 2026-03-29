import numpy as np

from stereo_utils import numpy_to_comfy_image


class DisparityToDepth:
    """Convert a raw disparity map to metric depth.

    Uses the stereo formula: depth = focal_length * baseline / disparity
    """

    CATEGORY = "FoundationStereo"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE", "DEPTH_RAW")
    RETURN_NAMES = ("depth_image", "depth_raw")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "disparity_raw": ("DISPARITY_RAW",),
                "focal_length": ("FLOAT", {"default": 1000.0, "min": 1.0, "max": 100000.0, "step": 0.1}),
                "baseline": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 100.0, "step": 0.001}),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
        }

    def convert(self, disparity_raw, focal_length, baseline, scale=1.0):
        disp = disparity_raw.copy()

        # Adjust focal length if images were downscaled before inference
        adjusted_focal = focal_length * scale

        # Avoid division by zero
        safe_disp = np.where(disp > 0.01, disp, 0.01)
        depth = adjusted_focal * baseline / safe_disp

        # Mark invalid regions (where disparity was ~0 or inf)
        invalid = (disparity_raw <= 0.01) | ~np.isfinite(disparity_raw)
        depth[invalid] = 0.0

        depth_image = numpy_to_comfy_image(depth)
        return (depth_image, depth)
