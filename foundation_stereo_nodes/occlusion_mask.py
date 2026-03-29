import torch
import numpy as np

from stereo_utils import compute_occlusion_mask


class OcclusionMask:
    """Generate an occlusion mask from a disparity map.

    Identifies pixels in the target view that have no corresponding source pixel
    (disoccluded regions). Useful for masking before inpainting.
    """

    CATEGORY = "FoundationStereo"
    FUNCTION = "compute"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("occlusion_mask",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "disparity_raw": ("DISPARITY_RAW",),
                "direction": (["left_to_right", "right_to_left"],),
                "dilate_pixels": ("INT", {"default": 2, "min": 0, "max": 50, "step": 1}),
            },
        }

    def compute(self, disparity_raw, direction, dilate_pixels):
        mask = compute_occlusion_mask(disparity_raw, direction, dilate_pixels)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)
        return (mask_tensor,)
