import numpy as np
import torch


def comfy_image_to_foundation(image_tensor: torch.Tensor) -> torch.Tensor:
    """Convert ComfyUI image tensor to FoundationStereo format.

    ComfyUI:          (B, H, W, C) float32 [0, 1]
    FoundationStereo: (1, C, H, W) float32 [0, 255]
    """
    img = image_tensor[0]  # take first in batch
    return (img.permute(2, 0, 1).unsqueeze(0) * 255.0).contiguous()


def numpy_to_comfy_image(arr: np.ndarray) -> torch.Tensor:
    """Convert a (H, W) or (H, W, C) numpy array to ComfyUI IMAGE tensor.

    Normalizes to [0, 1] and returns (1, H, W, 3) float32.
    """
    if arr.ndim == 2:
        arr_min = arr[np.isfinite(arr)].min() if np.isfinite(arr).any() else 0
        arr_max = arr[np.isfinite(arr)].max() if np.isfinite(arr).any() else 1
        denom = arr_max - arr_min if arr_max != arr_min else 1.0
        normalized = np.clip((arr - arr_min) / denom, 0, 1)
        normalized[~np.isfinite(arr)] = 0
        rgb = np.stack([normalized] * 3, axis=-1)
    else:
        rgb = arr.astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
    return torch.from_numpy(rgb).float().unsqueeze(0)


def comfy_image_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI IMAGE (B,H,W,C) to numpy (H,W,C) float32 [0,1]."""
    return image_tensor[0].cpu().numpy().astype(np.float32)


def forward_warp(image: np.ndarray, disparity: np.ndarray,
                 direction: str = "left_to_right",
                 fill_mode: str = "none") -> tuple:
    """Forward-warp an image using a disparity map (vectorized).

    Args:
        image: (H, W, C) float32 [0, 1]
        disparity: (H, W) float32 in pixels
        direction: "left_to_right" or "right_to_left"
        fill_mode: "none" or "nearest"

    Returns:
        warped: (H, W, C) float32 [0, 1]
        occlusion_mask: (H, W) bool — True where no source pixel mapped
    """
    H, W, C = image.shape
    sign = -1.0 if direction == "left_to_right" else 1.0

    # Source pixel coordinates
    ys, xs = np.mgrid[0:H, 0:W]
    x_target = xs.astype(np.float64) + sign * disparity.astype(np.float64)

    # Integer neighbors for bilinear splatting
    x0 = np.floor(x_target).astype(np.int64)
    x1 = x0 + 1
    w1 = (x_target - x0).astype(np.float32)
    w0 = 1.0 - w1

    warped = np.zeros((H, W, C), dtype=np.float32)
    z_buffer = np.full((H, W), -np.inf, dtype=np.float64)

    disp_flat = disparity.ravel()
    ys_flat = ys.ravel()

    # Process both bilinear neighbors
    for x_nb, w_nb in [(x0, w0), (x1, w1)]:
        x_nb_flat = x_nb.ravel()
        w_nb_flat = w_nb.ravel()

        valid = (x_nb_flat >= 0) & (x_nb_flat < W) & (w_nb_flat > 0)
        src_y = ys_flat[valid]
        dst_x = x_nb_flat[valid]
        d_vals = disp_flat[valid]

        # Sort by disparity ascending so later (closer) pixels overwrite farther ones
        order = np.argsort(d_vals)
        src_y = src_y[order]
        src_x = xs.ravel()[valid][order]
        dst_x = dst_x[order]
        d_vals = d_vals[order]

        # Scatter — last write wins (highest disparity = closest)
        warped[src_y, dst_x] = image[src_y, src_x]
        z_buffer[src_y, dst_x] = np.maximum(z_buffer[src_y, dst_x], d_vals)

    occlusion_mask = ~np.isfinite(z_buffer) | (z_buffer == -np.inf)

    if fill_mode == "nearest":
        _fill_nearest(warped, occlusion_mask)

    return warped, occlusion_mask


def _fill_nearest(warped: np.ndarray, mask: np.ndarray):
    """Fill occluded pixels with nearest non-occluded neighbor (horizontal)."""
    H, W, C = warped.shape
    for y in range(H):
        row_mask = mask[y]
        if not row_mask.any():
            continue
        # Forward pass — fill from left
        last_valid = None
        for x in range(W):
            if not row_mask[x]:
                last_valid = x
            elif last_valid is not None:
                warped[y, x] = warped[y, last_valid]
        # Backward pass — fill from right (only if still masked)
        last_valid = None
        for x in range(W - 1, -1, -1):
            if not row_mask[x]:
                last_valid = x
            elif last_valid is not None and mask[y, x]:
                warped[y, x] = warped[y, last_valid]


def depth2xyzmap(depth: np.ndarray, K, uvs: np.ndarray = None, zmin=0.1):
    """Convert a depth map to an XYZ point map using camera intrinsics.

    Adapted from FoundationStereo/Utils.py to avoid heavy transitive imports.
    """
    invalid_mask = depth < zmin
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map


def toOpen3dCloud(points, colors=None, normals=None):
    """Create an Open3D point cloud from numpy arrays.

    Adapted from FoundationStereo/Utils.py to avoid heavy transitive imports.
    """
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def compute_occlusion_mask(disparity: np.ndarray,
                           direction: str = "left_to_right",
                           dilate_pixels: int = 0) -> np.ndarray:
    """Compute occlusion mask from disparity without warping.

    Returns: (H, W) float32 mask, 1.0 = occluded, 0.0 = visible.
    """
    H, W = disparity.shape
    sign = -1.0 if direction == "left_to_right" else 1.0

    ys, xs = np.mgrid[0:H, 0:W]
    x_target = np.round(xs + sign * disparity).astype(np.int64)

    covered = np.zeros((H, W), dtype=bool)
    valid = (x_target >= 0) & (x_target < W)
    covered[ys[valid], x_target[valid]] = True

    mask = (~covered).astype(np.float32)

    if dilate_pixels > 0:
        import cv2
        kernel = np.ones((dilate_pixels * 2 + 1, dilate_pixels * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask
