import os
import logging
import torch
from omegaconf import OmegaConf

from core.foundation_stereo import FoundationStereo

logger = logging.getLogger("ComfyUI-FoundationStereo")

# Resolve the FoundationStereo submodule root
_FOUNDATION_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "FoundationStereo")
)


class FoundationStereoModelLoader:
    """Loads a FoundationStereo checkpoint and caches it for reuse."""

    CATEGORY = "FoundationStereo"
    FUNCTION = "load_model"
    RETURN_TYPES = ("FOUNDATION_STEREO_MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["23-51-11", "11-33-40"],),
                "device": (["cuda", "cpu"],),
            },
        }

    def load_model(self, model_name, device):
        ckpt_dir = os.path.join(
            _FOUNDATION_DIR, "pretrained_models", model_name, "model_best_bp2.pth"
        )
        cfg_path = os.path.join(
            _FOUNDATION_DIR, "pretrained_models", model_name, "cfg.yaml"
        )

        if not os.path.isfile(ckpt_dir):
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_dir}. "
                f"Download weights from the FoundationStereo repo and place the "
                f"'{model_name}' folder in FoundationStereo/pretrained_models/"
            )

        cfg = OmegaConf.load(cfg_path)
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        args = OmegaConf.create(cfg)

        logger.info(f"Loading FoundationStereo model '{model_name}' on {device}")
        model = FoundationStereo(args)
        ckpt = torch.load(ckpt_dir, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()

        return ((model, args, device),)
