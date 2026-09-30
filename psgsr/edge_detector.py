"""Lazy TEED inference for the optional reference-image edge detector."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


CHECKPOINT = Path(__file__).parent / "vendor/teed/5_model.pth"
MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


@lru_cache(maxsize=2)
def _load_model(device_name):
    import torch

    from vendor.teed.model import TED

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"TEED checkpoint not found: {CHECKPOINT}")
    device = torch.device(device_name)
    model = TED().to(device)
    try:
        state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    return model.eval()


def teed_probability(image):
    """Return the fused TEED probability map at the input image resolution."""
    import torch

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("TEED expects an RGB image")
    original_size = rgb.shape[1], rgb.shape[0]
    bgr = rgb[:, :, ::-1]
    if min(bgr.shape[:2]) < 512:
        bgr = cv2.resize(bgr, None, fx=1.5, fy=1.5)
    width = ((bgr.shape[1] + 7) // 8) * 8
    height = ((bgr.shape[0] + 7) // 8) * 8
    bgr = cv2.resize(bgr, (width, height)).astype(np.float32)
    tensor = torch.from_numpy(
        (bgr - MEAN_BGR).transpose(2, 0, 1).copy(),
    ).unsqueeze(0)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(device_name)
    with torch.inference_mode():
        fused = torch.sigmoid(model(tensor.to(device_name))[-1])
    probability = fused[0, 0].cpu().numpy()
    return cv2.resize(probability, original_size, interpolation=cv2.INTER_LINEAR)


def teed_edges(image):
    """Return the fused TEED response without thresholding or tracing."""
    probability = teed_probability(image)
    response = cv2.normalize(
        probability, None, 0, 255, cv2.NORM_MINMAX,
    ).astype(np.uint8)
    return response, {
        "method": "teed",
        "output": "fused_probability",
        "postprocess": "none",
        "checkpoint": str(CHECKPOINT),
    }
