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

# HuggingFace repos that include both cfg.yaml and model_best_bp2.pth
_HF_REPOS = {
    "23-51-11": {
        "repo_id": "pablovela5620/foundation-stereo",
        "ckpt_filename": "model_best_bp2.pth",
        "cfg_filename": "cfg.yaml",
        "vit_size_default": "vitl",
    },
    "11-33-40": {
        "repo_id": "Livioni/foundationstereos",
        "ckpt_filename": "model_best_bp2.pth",
        "cfg_filename": None,  # not included — we bundle a default
        "vit_size_default": "vits",
    },
}

# Built-in configs matching the official checkpoints.
# These are used as fallback if cfg.yaml isn't downloaded or is missing.
_DEFAULT_CONFIGS = {
    "23-51-11": {
        "hidden_dims": [128, 128, 128],
        "max_disp": 416,
        "n_gru_layers": 3,
        "n_downsample": 2,
        "corr_levels": 2,
        "corr_radius": 4,
        "corr_implementation": "reg",
        "mixed_precision": True,
        "vit_size": "vitl",
        "valid_iters": 32,
        "slow_fast_gru": False,
    },
    "11-33-40": {
        "hidden_dims": [128, 128, 128],
        "max_disp": 416,
        "n_gru_layers": 3,
        "n_downsample": 2,
        "corr_levels": 2,
        "corr_radius": 4,
        "corr_implementation": "reg",
        "mixed_precision": True,
        "vit_size": "vits",
        "valid_iters": 32,
        "slow_fast_gru": False,
    },
}


def _download_model(model_name: str) -> tuple:
    """Download model from HuggingFace Hub. Returns (ckpt_path, cfg_path_or_dict)."""
    from huggingface_hub import hf_hub_download

    info = _HF_REPOS[model_name]
    local_dir = os.path.join(_FOUNDATION_DIR, "pretrained_models", model_name)
    os.makedirs(local_dir, exist_ok=True)

    logger.info(f"Downloading FoundationStereo '{model_name}' from HuggingFace ({info['repo_id']})...")

    ckpt_path = hf_hub_download(
        repo_id=info["repo_id"],
        filename=info["ckpt_filename"],
        local_dir=local_dir,
    )

    cfg_path = None
    if info["cfg_filename"]:
        try:
            cfg_path = hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["cfg_filename"],
                local_dir=local_dir,
            )
        except Exception:
            logger.warning("cfg.yaml not found in HF repo, using built-in defaults")

    logger.info(f"Download complete: {ckpt_path}")
    return ckpt_path, cfg_path


class FoundationStereoModelLoader:
    """Loads a FoundationStereo checkpoint, auto-downloading from HuggingFace if needed."""

    CATEGORY = "FoundationStereo"
    FUNCTION = "load_model"
    RETURN_TYPES = ("FOUNDATION_STEREO_MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["23-51-11 (ViT-Large, best quality)", "11-33-40 (ViT-Small, faster)"],),
                "device": (["cuda", "cpu"],),
            },
        }

    def load_model(self, model_name, device):
        # Parse display name back to model ID
        model_id = model_name.split(" ")[0]

        local_dir = os.path.join(_FOUNDATION_DIR, "pretrained_models", model_id)
        ckpt_path = os.path.join(local_dir, "model_best_bp2.pth")
        cfg_path = os.path.join(local_dir, "cfg.yaml")

        # Auto-download if not present
        if not os.path.isfile(ckpt_path):
            logger.info(f"Model not found locally at {ckpt_path}, downloading...")
            ckpt_path, downloaded_cfg = _download_model(model_id)
            if downloaded_cfg:
                cfg_path = downloaded_cfg

        # Load config
        if os.path.isfile(cfg_path):
            cfg = OmegaConf.load(cfg_path)
        else:
            logger.info(f"Using built-in default config for {model_id}")
            cfg = OmegaConf.create(_DEFAULT_CONFIGS[model_id])

        if "vit_size" not in cfg:
            cfg["vit_size"] = _HF_REPOS[model_id]["vit_size_default"]

        args = OmegaConf.create(cfg)

        logger.info(f"Loading FoundationStereo model '{model_id}' on {device}")
        model = FoundationStereo(args)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()

        return ((model, args, device),)
