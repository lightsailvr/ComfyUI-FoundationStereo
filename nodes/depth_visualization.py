import numpy as np
import cv2
import torch


class DepthVisualization:
    """Apply a pseudocolor colormap to a grayscale depth or disparity image."""

    CATEGORY = "FoundationStereo"
    FUNCTION = "visualize"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)

    COLORMAPS = {
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "magma": cv2.COLORMAP_MAGMA,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colormap": (["turbo", "viridis", "inferno", "plasma", "magma"],),
            },
        }

    def visualize(self, image, colormap):
        img_np = image[0].cpu().numpy()  # (H, W, C)

        # Use first channel as grayscale source
        gray = img_np[:, :, 0] if img_np.ndim == 3 else img_np

        # Normalize to 0-255 uint8 for colormap
        valid = np.isfinite(gray)
        if valid.any():
            g_min, g_max = gray[valid].min(), gray[valid].max()
        else:
            g_min, g_max = 0.0, 1.0
        denom = g_max - g_min if g_max != g_min else 1.0
        normalized = np.clip((gray - g_min) / denom, 0, 1)
        normalized[~valid] = 0
        uint8_map = (normalized * 255).astype(np.uint8)

        # Apply OpenCV colormap (returns BGR)
        colored_bgr = cv2.applyColorMap(uint8_map, self.COLORMAPS[colormap])
        colored_rgb = colored_bgr[:, :, ::-1].copy()  # BGR → RGB

        # Convert to ComfyUI IMAGE (1, H, W, 3) float [0, 1]
        result = torch.from_numpy(colored_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        return (result,)
