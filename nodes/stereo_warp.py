import torch
import numpy as np

from stereo_utils import comfy_image_to_numpy, forward_warp


class StereoWarp:
    """Warp an image from one eye to the other using a disparity map.

    Uses forward warping with z-buffering. Outputs the warped image and an
    occlusion mask identifying disoccluded regions (holes).
    """

    CATEGORY = "FoundationStereo"
    FUNCTION = "warp"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("warped_image", "occlusion_mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "disparity_raw": ("DISPARITY_RAW",),
                "direction": (["left_to_right", "right_to_left"],),
                "fill_mode": (["none", "nearest"],),
            },
        }

    def warp(self, image, disparity_raw, direction, fill_mode):
        img_np = comfy_image_to_numpy(image)
        disp_np = disparity_raw.copy()

        # Handle size mismatch (if disparity was computed at different scale)
        H_img, W_img = img_np.shape[:2]
        H_disp, W_disp = disp_np.shape[:2]
        if (H_img, W_img) != (H_disp, W_disp):
            import cv2
            scale_x = W_img / W_disp
            disp_np = cv2.resize(disp_np, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
            disp_np *= scale_x  # scale disparity values too

        warped, occ_mask = forward_warp(img_np, disp_np, direction, fill_mode)

        # Convert to ComfyUI formats
        warped_tensor = torch.from_numpy(warped).float().unsqueeze(0)  # (1, H, W, C)
        mask_tensor = torch.from_numpy(occ_mask.astype(np.float32)).unsqueeze(0)  # (1, H, W)

        return (warped_tensor, mask_tensor)
