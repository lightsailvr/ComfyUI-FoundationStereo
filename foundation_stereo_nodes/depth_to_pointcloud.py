import os
import numpy as np

from stereo_utils import comfy_image_to_numpy, depth2xyzmap, toOpen3dCloud


class DepthToPointCloud:
    """Generate a 3D point cloud (.ply) from a metric depth map and color image."""

    CATEGORY = "FoundationStereo"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_raw": ("DEPTH_RAW",),
                "color_image": ("IMAGE",),
                "focal_length": ("FLOAT", {"default": 1000.0, "min": 1.0, "max": 100000.0, "step": 0.1}),
                "cx": ("FLOAT", {"default": 320.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "cy": ("FLOAT", {"default": 240.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "output_path": ("STRING", {"default": "output/cloud.ply"}),
            },
            "optional": {
                "z_far": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 1000.0, "step": 0.1}),
                "denoise": ("BOOLEAN", {"default": True}),
                "denoise_nb_points": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1}),
                "denoise_radius": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 1.0, "step": 0.001}),
            },
        }

    def generate(self, depth_raw, color_image, focal_length, cx, cy, output_path,
                 z_far=10.0, denoise=True, denoise_nb_points=30, denoise_radius=0.03):
        try:
            import open3d  # noqa: F401 — verify real open3d is installed
            open3d.geometry.PointCloud  # will fail on stub
        except (ImportError, AttributeError):
            raise RuntimeError(
                "open3d is required for point cloud export but is not installed. "
                "Your Python version may not be supported — open3d currently requires Python <=3.12. "
                "Install with: pip install open3d"
            )
        depth = depth_raw.copy()
        color_np = comfy_image_to_numpy(color_image)  # (H, W, 3) float [0,1]
        color_uint8 = (color_np * 255).astype(np.uint8)

        H, W = depth.shape[:2]

        # Build intrinsic matrix
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1],
        ], dtype=np.float32)

        # Resize color to match depth if needed
        if (color_uint8.shape[0], color_uint8.shape[1]) != (H, W):
            import cv2
            color_uint8 = cv2.resize(color_uint8, (W, H), interpolation=cv2.INTER_LINEAR)

        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), color_uint8.reshape(-1, 3))

        # Clip by z_far
        points = np.asarray(pcd.points)
        keep = (points[:, 2] > 0) & (points[:, 2] <= z_far)
        pcd = pcd.select_by_index(np.where(keep)[0])

        if denoise and len(pcd.points) > 0:
            _, ind = pcd.remove_radius_outlier(
                nb_points=denoise_nb_points, radius=denoise_radius
            )
            pcd = pcd.select_by_index(ind)

        # Ensure output directory exists
        abs_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        import open3d as o3d
        o3d.io.write_point_cloud(abs_path, pcd)

        return (abs_path,)
