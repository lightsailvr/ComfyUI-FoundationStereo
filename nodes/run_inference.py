import torch
import numpy as np
import cv2

import sys, os
_FOUNDATION_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "FoundationStereo")
)
if _FOUNDATION_DIR not in sys.path:
    sys.path.insert(0, _FOUNDATION_DIR)

from core.utils.utils import InputPadder
from stereo_utils import comfy_image_to_foundation, numpy_to_comfy_image


class FoundationStereoRun:
    """Run stereo disparity estimation on a rectified image pair."""

    CATEGORY = "FoundationStereo"
    FUNCTION = "run_inference"
    RETURN_TYPES = ("IMAGE", "DISPARITY_RAW")
    RETURN_NAMES = ("disparity_image", "disparity_raw")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FOUNDATION_STEREO_MODEL",),
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
                "iterations": ("INT", {"default": 32, "min": 1, "max": 64, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "use_hierarchical": ("BOOLEAN", {"default": False}),
            },
        }

    def run_inference(self, model, left_image, right_image, iterations, scale, use_hierarchical):
        net, args, device = model

        # Convert ComfyUI tensors to FoundationStereo format
        img0 = comfy_image_to_foundation(left_image).to(device)
        img1 = comfy_image_to_foundation(right_image).to(device)

        # Optional downscale
        if scale < 1.0:
            _, _, H, W = img0.shape
            new_H, new_W = int(H * scale), int(W * scale)
            img0 = torch.nn.functional.interpolate(
                img0, size=(new_H, new_W), mode="bilinear", align_corners=False
            )
            img1 = torch.nn.functional.interpolate(
                img1, size=(new_H, new_W), mode="bilinear", align_corners=False
            )

        _, _, H, W = img0.shape

        # Pad to multiple of 32
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                if use_hierarchical:
                    disp = net.run_hierachical(
                        img0, img1, iters=iterations, test_mode=True, small_ratio=0.5
                    )
                else:
                    disp = net.forward(img0, img1, iters=iterations, test_mode=True)

        disp = padder.unpad(disp.float())
        disp_np = disp.data.cpu().numpy().reshape(H, W)

        # Normalized grayscale preview
        disp_image = numpy_to_comfy_image(disp_np)

        return (disp_image, disp_np)
