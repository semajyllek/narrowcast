"""Loading a frozen encoder and turning images into vectors.

The encoder is never trained here. What gets built per user is a head plus two
thresholds -- at 20 labels and 512 dimensions that is ~40 KB against an encoder
of tens of megabytes, which is why this costs CPU-seconds rather than GPU-hours.

Torch and open_clip are imported lazily so that `plan`, `card`, and any workflow
using `--embeddings` run without them installed.
"""

import numpy as np

ENCODERS = {
    "mobileclip2_s0": {"loader": "open_clip", "spec": "hf-hub:timm/MobileCLIP2-S0-OpenCLIP"},
    "mobileclip2_s2": {"loader": "open_clip", "spec": "hf-hub:timm/MobileCLIP2-S2-OpenCLIP"},
    "bioclip2": {"loader": "open_clip", "spec": "hf-hub:imageomics/bioclip-2"},
    "plantclef24": {
        "loader": "timm_checkpoint",
        "spec": "vit_base_patch14_reg4_dinov2.lvd142m",
        "repo": "vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights",
        "filename": "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all.safetensors",
        "img_size": 518,
    },
}
BATCH_SIZE = 64


class _ImageTower:
    """Adapt an open_clip model to a plain callable returning image embeddings."""

    def __init__(self, clip_model):
        self.clip = clip_model

    def __call__(self, x):
        return self.clip.encode_image(x)

    def eval(self):
        self.clip.eval()
        return self

    def to(self, device):
        self.clip.to(device)
        return self

    def parameters(self):
        return self.clip.parameters()


def device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(variant: str):
    """-> (callable, preprocess, device). Weights are frozen and never updated."""
    import torch
    cfg = ENCODERS.get(variant)
    if cfg is None:
        raise ValueError(f"unknown encoder {variant!r}; have {sorted(ENCODERS)}")
    dev = device()

    if cfg["loader"] == "open_clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(cfg["spec"])
        model = _ImageTower(model)
    elif cfg["loader"] == "timm_checkpoint":
        import timm
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        model = timm.create_model(cfg["spec"], pretrained=False, num_classes=0,
                                  img_size=cfg["img_size"])
        state = load_file(hf_hub_download(cfg["repo"], cfg["filename"]))
        model.load_state_dict({k: v for k, v in state.items()
                               if not k.startswith("head.")}, strict=False)
        dc = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**dc, is_training=False)
    else:  # pragma: no cover - guarded by ENCODERS
        raise ValueError(f"unknown loader {cfg['loader']}")

    model = model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, dev


def embed_images(paths, model, preprocess, dev, batch_size=BATCH_SIZE, desc=""):
    """Batched inference over a list of image paths -> (n, dim) float32."""
    import torch
    from PIL import Image
    out = []
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i + batch_size]:
            with Image.open(p) as im:
                batch.append(preprocess(im.convert("RGB")))
        with torch.no_grad():
            v = model(torch.stack(batch).to(dev))
        out.append(v.float().cpu().numpy())
    return np.vstack(out).astype("float32")
