"""High-resolution fixed-grid DINO direction loss for rendered candidates."""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def relative_object_crop(rgb, mask, size=1036, padding_fraction=0.08):
    """Mask, crop, and resize an object while preserving relative coordinates."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(mask) > 0
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != mask.shape:
        raise ValueError("DINO direction image and mask shapes must agree")
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("DINO direction loss requires a non-empty mask")

    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    pad_x = max(1, int(round((x1 - x0) * float(padding_fraction))))
    pad_y = max(1, int(round((y1 - y0) * float(padding_fraction))))
    left, right = x0 - pad_x, x1 + pad_x
    top, bottom = y0 - pad_y, y1 + pad_y
    crop = np.full((bottom - top, right - left, 3), 127, dtype=np.uint8)
    crop_mask = np.zeros((bottom - top, right - left), dtype=np.uint8)

    source_left, source_right = max(left, 0), min(right, rgb.shape[1])
    source_top, source_bottom = max(top, 0), min(bottom, rgb.shape[0])
    target_left, target_top = source_left - left, source_top - top
    target_right = target_left + source_right - source_left
    target_bottom = target_top + source_bottom - source_top
    source_mask = mask[source_top:source_bottom, source_left:source_right]
    region = crop[target_top:target_bottom, target_left:target_right]
    source_rgb = rgb[source_top:source_bottom, source_left:source_right]
    region[source_mask] = source_rgb[source_mask]
    crop_mask[target_top:target_bottom, target_left:target_right] = source_mask

    return (
        cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4),
        cv2.resize(
            crop_mask, (size, size), interpolation=cv2.INTER_NEAREST,
        ) > 0,
    )


def shared_camera_object_crops(images_and_masks, size=1036, padding_fraction=0.08):
    """Crop several masked images in one shared camera-coordinate square."""
    values = [
        (np.asarray(rgb, dtype=np.uint8), np.asarray(mask) > 0)
        for rgb, mask in images_and_masks
    ]
    if not values:
        return []
    shape = values[0][1].shape
    if any(rgb.shape[:2] != shape or mask.shape != shape for rgb, mask in values):
        raise ValueError("shared DINO direction images and masks must agree")
    union = np.logical_or.reduce([mask for _, mask in values])
    rows, columns = np.nonzero(union)
    if not len(rows):
        raise ValueError("DINO direction loss requires a non-empty mask")
    width = int(columns.max() - columns.min() + 1)
    height = int(rows.max() - rows.min() + 1)
    side = max(1, int(np.ceil(max(width, height) * (1.0 + 2.0 * padding_fraction))))
    center_x = 0.5 * float(columns.min() + columns.max())
    center_y = 0.5 * float(rows.min() + rows.max())
    left = int(np.floor(center_x - 0.5 * side))
    top = int(np.floor(center_y - 0.5 * side))
    right, bottom = left + side, top + side
    source_left, source_right = max(left, 0), min(right, shape[1])
    source_top, source_bottom = max(top, 0), min(bottom, shape[0])
    target_left, target_top = source_left - left, source_top - top
    target_right = target_left + source_right - source_left
    target_bottom = target_top + source_bottom - source_top
    results = []
    for rgb, mask in values:
        crop = np.full((side, side, 3), 238, dtype=np.uint8)
        crop_mask = np.zeros((side, side), dtype=np.uint8)
        source_mask = mask[source_top:source_bottom, source_left:source_right]
        source_rgb = rgb[source_top:source_bottom, source_left:source_right]
        region = crop[target_top:target_bottom, target_left:target_right]
        region[source_mask] = source_rgb[source_mask]
        crop_mask[target_top:target_bottom, target_left:target_right] = source_mask
        results.append((
            cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4),
            cv2.resize(
                crop_mask, (size, size), interpolation=cv2.INTER_NEAREST,
            ) > 0,
        ))
    return results


def reference_camera_object_crops(
    images_and_masks, reference_mask, size=728, padding_fraction=0.2,
):
    """Crop every candidate with one square derived only from the reference."""
    values = [
        (np.asarray(rgb, dtype=np.uint8), np.asarray(mask) > 0)
        for rgb, mask in images_and_masks
    ]
    reference_mask = np.asarray(reference_mask) > 0
    rows, columns = np.nonzero(reference_mask)
    if not len(rows):
        raise ValueError("DINO direction loss requires a non-empty reference mask")
    shape = reference_mask.shape
    if any(rgb.shape[:2] != shape or mask.shape != shape for rgb, mask in values):
        raise ValueError("reference-coordinate DINO images and masks must agree")
    width = int(columns.max() - columns.min() + 1)
    height = int(rows.max() - rows.min() + 1)
    side = max(1, int(np.ceil(max(width, height) * (1.0 + 2.0 * padding_fraction))))
    center_x = 0.5 * float(columns.min() + columns.max())
    center_y = 0.5 * float(rows.min() + rows.max())
    left = int(np.floor(center_x - 0.5 * side))
    top = int(np.floor(center_y - 0.5 * side))
    source_left, source_right = max(left, 0), min(left + side, shape[1])
    source_top, source_bottom = max(top, 0), min(top + side, shape[0])
    target_left, target_top = source_left - left, source_top - top
    target_right = target_left + source_right - source_left
    target_bottom = target_top + source_bottom - source_top
    results = []
    for rgb, mask in values:
        crop = np.full((side, side, 3), 238, dtype=np.uint8)
        crop_mask = np.zeros((side, side), dtype=np.uint8)
        source_mask = mask[source_top:source_bottom, source_left:source_right]
        source_rgb = rgb[source_top:source_bottom, source_left:source_right]
        region = crop[target_top:target_bottom, target_left:target_right]
        region[source_mask] = source_rgb[source_mask]
        crop_mask[target_top:target_bottom, target_left:target_right] = source_mask
        results.append((
            cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4),
            cv2.resize(crop_mask, (size, size), interpolation=cv2.INTER_NEAREST) > 0,
        ))
    return results


def _structure_weights(image, mask, grid_size, *, normal_map=False):
    """Return patch weights emphasizing internal RGB or surface-normal changes."""
    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask) > 0
    if normal_map:
        value = image.astype(np.float32) / 127.5 - 1.0
        value /= np.maximum(np.linalg.norm(value, axis=2, keepdims=True), 1e-6)
    else:
        value = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    value = cv2.GaussianBlur(value, (0, 0), 1.0)
    magnitude = np.sqrt(sum(
        cv2.Sobel(value[..., channel], cv2.CV_32F, dx, dy, ksize=3) ** 2
        for channel in range(value.shape[2])
        for dx, dy in ((1, 0), (0, 1))
    ))
    interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    positive = magnitude[interior & (magnitude > 0)]
    scale = float(np.quantile(positive, 0.9)) if len(positive) else 1.0
    weights = 0.1 + 0.9 * np.clip(magnitude / max(scale, 1e-6), 0.0, 1.0)
    weights[~mask] = 0.0
    return cv2.resize(
        weights, (grid_size, grid_size), interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


def dense_relative_position_loss(
    reference_features, candidate_features, reference_mask, candidate_mask,
    reference_weights, candidate_weights, *, device,
):
    """Match DINO patches by appearance, then score their unchanged coordinates."""
    import torch

    reference_mask = np.asarray(reference_mask, dtype=bool)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    reference_ids = np.flatnonzero(reference_mask)
    candidate_ids = np.flatnonzero(candidate_mask)
    if not len(reference_ids) or not len(candidate_ids):
        return 1.0, {"forward_matches": 0, "backward_matches": 0}
    torch_device = torch.device(
        device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    )
    reference = torch.as_tensor(
        np.asarray(reference_features).reshape(-1, reference_features.shape[-1])[
            reference_ids
        ], device=torch_device, dtype=torch.float32,
    )
    candidate = torch.as_tensor(
        np.asarray(candidate_features).reshape(-1, candidate_features.shape[-1])[
            candidate_ids
        ], device=torch_device, dtype=torch.float32,
    )
    similarity = reference @ candidate.T
    forward_similarity, forward_match = similarity.max(dim=1)
    backward_similarity, backward_match = similarity.max(dim=0)
    reference_y, reference_x = np.unravel_index(reference_ids, reference_mask.shape)
    candidate_y, candidate_x = np.unravel_index(candidate_ids, candidate_mask.shape)
    reference_xy = torch.as_tensor(
        np.column_stack((reference_x, reference_y)), device=torch_device,
        dtype=torch.float32,
    ) / max(reference_mask.shape[0] - 1, 1)
    candidate_xy = torch.as_tensor(
        np.column_stack((candidate_x, candidate_y)), device=torch_device,
        dtype=torch.float32,
    ) / max(candidate_mask.shape[0] - 1, 1)
    reference_weight = torch.as_tensor(
        np.asarray(reference_weights).reshape(-1)[reference_ids],
        device=torch_device, dtype=torch.float32,
    )
    candidate_weight = torch.as_tensor(
        np.asarray(candidate_weights).reshape(-1)[candidate_ids],
        device=torch_device, dtype=torch.float32,
    )

    def weighted_position(source_xy, target_xy, matches, similarity_value, weights):
        distance = torch.linalg.vector_norm(source_xy - target_xy[matches], dim=1)
        confidence = similarity_value.clamp(min=0.0).square()
        weights = weights * confidence.clamp(min=0.05)
        return (
            (distance * weights).sum()
            / weights.sum().clamp(min=1e-6)
            / math.sqrt(2.0)
        )

    forward = weighted_position(
        reference_xy, candidate_xy, forward_match, forward_similarity,
        reference_weight * candidate_weight[forward_match],
    )
    backward = weighted_position(
        candidate_xy, reference_xy, backward_match, backward_similarity,
        candidate_weight * reference_weight[backward_match],
    )
    loss = 0.5 * (forward + backward)
    return float(loss.item()), {
        "forward_matches": int(len(reference_ids)),
        "backward_matches": int(len(candidate_ids)),
        "forward_position_loss": float(forward.item()),
        "backward_position_loss": float(backward.item()),
        "mean_forward_similarity": float(forward_similarity.mean().item()),
        "mean_backward_similarity": float(backward_similarity.mean().item()),
    }


def dense_relative_position_loss_batch(
    pairs, *, device, batch_size=4,
):
    """Score fixed-grid correspondence pairs in chunked tensor batches."""
    import torch

    if not pairs:
        return []
    valid = [
        bool(np.any(pair[2])) and bool(np.any(pair[3]))
        for pair in pairs
    ]
    if not all(valid):
        scored = iter(dense_relative_position_loss_batch(
            [pair for pair, keep in zip(pairs, valid) if keep],
            device=device, batch_size=batch_size,
        ))
        empty = (1.0, {
            "forward_matches": 0,
            "backward_matches": 0,
            "forward_position_loss": 1.0,
            "backward_position_loss": 1.0,
            "mean_forward_similarity": 0.0,
            "mean_backward_similarity": 0.0,
        })
        return [next(scored) if keep else empty for keep in valid]
    torch_device = torch.device(
        device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    )
    grid_shape = np.asarray(pairs[0][2]).shape
    if len(grid_shape) != 2 or any(
        np.asarray(pair[2]).shape != grid_shape
        or np.asarray(pair[3]).shape != grid_shape
        for pair in pairs
    ):
        raise ValueError("batched DINO position masks must share one grid shape")
    height, width = grid_shape
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    coordinates = torch.stack((xs, ys), dim=-1).reshape(-1, 2)
    coordinates = coordinates / max(height - 1, 1)

    def score_chunks(current_batch_size):
        from torch.nn.utils.rnn import pad_sequence

        outputs = []
        for start in range(0, len(pairs), current_batch_size):
            chunk = pairs[start:start + current_batch_size]
            reference_masks = [
                torch.as_tensor(pair[2], dtype=torch.bool).reshape(-1)
                for pair in chunk
            ]
            candidate_masks = [
                torch.as_tensor(pair[3], dtype=torch.bool).reshape(-1)
                for pair in chunk
            ]
            reference = pad_sequence([
                torch.as_tensor(pair[0], dtype=torch.float32).reshape(
                    -1, pair[0].shape[-1],
                )[mask]
                for pair, mask in zip(chunk, reference_masks)
            ], batch_first=True).to(torch_device)
            candidate = pad_sequence([
                torch.as_tensor(pair[1], dtype=torch.float32).reshape(
                    -1, pair[1].shape[-1],
                )[mask]
                for pair, mask in zip(chunk, candidate_masks)
            ], batch_first=True).to(torch_device)
            reference_weight = pad_sequence([
                torch.as_tensor(pair[4], dtype=torch.float32).reshape(-1)[mask]
                for pair, mask in zip(chunk, reference_masks)
            ], batch_first=True).to(torch_device)
            candidate_weight = pad_sequence([
                torch.as_tensor(pair[5], dtype=torch.float32).reshape(-1)[mask]
                for pair, mask in zip(chunk, candidate_masks)
            ], batch_first=True).to(torch_device)
            reference_xy = pad_sequence([
                coordinates[mask] for mask in reference_masks
            ], batch_first=True).to(torch_device)
            candidate_xy = pad_sequence([
                coordinates[mask] for mask in candidate_masks
            ], batch_first=True).to(torch_device)
            reference_count = torch.as_tensor([
                int(mask.sum()) for mask in reference_masks
            ], device=torch_device)
            candidate_count = torch.as_tensor([
                int(mask.sum()) for mask in candidate_masks
            ], device=torch_device)
            reference_valid = (
                torch.arange(reference.shape[1], device=torch_device)[None]
                < reference_count[:, None]
            )
            candidate_valid = (
                torch.arange(candidate.shape[1], device=torch_device)[None]
                < candidate_count[:, None]
            )
            similarity = torch.bmm(reference, candidate.transpose(1, 2))
            forward_similarity, forward_match = similarity.masked_fill(
                ~candidate_valid[:, None, :], -torch.inf,
            ).max(dim=2)
            backward_similarity, backward_match = similarity.masked_fill(
                ~reference_valid[:, :, None], -torch.inf,
            ).max(dim=1)
            forward_distance = torch.linalg.vector_norm(
                reference_xy - candidate_xy.gather(
                    1, forward_match[..., None].expand(-1, -1, 2),
                ), dim=2,
            )
            backward_distance = torch.linalg.vector_norm(
                candidate_xy - reference_xy.gather(
                    1, backward_match[..., None].expand(-1, -1, 2),
                ), dim=2,
            )
            forward_weight = (
                reference_weight
                * candidate_weight.gather(1, forward_match)
                * forward_similarity.clamp(min=0.0).square().clamp(min=0.05)
            )
            backward_weight = (
                candidate_weight
                * reference_weight.gather(1, backward_match)
                * backward_similarity.clamp(min=0.0).square().clamp(min=0.05)
            )
            forward = (
                (forward_distance * forward_weight).sum(dim=1)
                / forward_weight.sum(dim=1).clamp(min=1e-6)
                / math.sqrt(2.0)
            )
            backward = (
                (backward_distance * backward_weight).sum(dim=1)
                / backward_weight.sum(dim=1).clamp(min=1e-6)
                / math.sqrt(2.0)
            )
            losses = 0.5 * (forward + backward)
            for index in range(len(chunk)):
                current_reference_valid = reference_valid[index]
                current_candidate_valid = candidate_valid[index]
                outputs.append((float(losses[index].item()), {
                    "forward_matches": int(reference_count[index].item()),
                    "backward_matches": int(candidate_count[index].item()),
                    "forward_position_loss": float(forward[index].item()),
                    "backward_position_loss": float(backward[index].item()),
                    "mean_forward_similarity": float(
                        forward_similarity[index][current_reference_valid].mean().item()
                    ),
                    "mean_backward_similarity": float(
                        backward_similarity[index][current_candidate_valid].mean().item()
                    ),
                }))
        return outputs

    current_batch_size = min(max(1, int(batch_size)), len(pairs))
    while True:
        try:
            return score_chunks(current_batch_size)
        except torch.cuda.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()


def fixed_grid_cosine_loss(
    reference_features, candidate_features, reference_mask, candidate_mask,
):
    """Compare same-position DINO tokens over the union foreground region."""
    import torch
    import torch.nn.functional as functional

    reference = torch.as_tensor(reference_features, dtype=torch.float32)
    candidate = torch.as_tensor(candidate_features, dtype=torch.float32)
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("DINO feature grids must have the same HxWxC shape")
    reference_mask = torch.as_tensor(reference_mask, dtype=torch.bool)
    candidate_mask = torch.as_tensor(candidate_mask, dtype=torch.bool)
    if reference_mask.shape != reference.shape[:2]:
        raise ValueError("reference patch mask does not match its feature grid")
    if candidate_mask.shape != candidate.shape[:2]:
        raise ValueError("candidate patch mask does not match its feature grid")
    foreground = reference_mask | candidate_mask
    if not bool(foreground.any()):
        return 1.0
    similarity = (
        functional.normalize(reference, dim=-1)
        * functional.normalize(candidate, dim=-1)
    ).sum(dim=-1)
    return float((1.0 - similarity[foreground]).mean().item())


class DinoGridDirectionScorer:
    """Fuse RGB/rendered-RGB and RGB/rendered-normal fixed-grid DINO losses."""

    backend = "dino_v2_fixed_relative_grid_rgb_normal"
    loss_formula = (
        "0.5*mean_union(1-cos(DINO(real_rgb),DINO(rendered_rgb)))+"
        "0.5*mean_union(1-cos(DINO(real_rgb),DINO(rendered_normal)))"
    )
    requires_full_resolution = True

    def __init__(
        self, root, device, input_size=1036,
        crop_mode="independent",
    ):
        self.root = Path(root).expanduser().resolve()
        self.device = device
        self.input_size = int(input_size)
        self.crop_mode = str(crop_mode)
        if self.input_size <= 0 or self.input_size % 14:
            raise ValueError("DINO direction input size must be a positive multiple of 14")
        if self.crop_mode not in ("independent", "shared_camera"):
            raise ValueError("DINO direction crop mode must be independent or shared_camera")
        self._model = None
        self._batch_size = 4
        self._lock = threading.Lock()
        self._preload_guard = threading.Lock()
        self._preload_thread = None
        self._preload_error = None

    def preload(self):
        """Load the shared DINO model before the first scored candidate batch."""
        with self._lock:
            if self._model is not None:
                return 0.0
            import torch

            start = time.perf_counter()
            self._load()
            device = torch.device(self.device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
        print(f"[direction loss] DINO preload={seconds:.3f}s", flush=True)
        return seconds

    def preload_async(self):
        """Overlap model loading with point-cloud and Blender preparation."""
        with self._preload_guard:
            if self._model is not None or self._preload_thread is not None:
                return

            def load():
                try:
                    self.preload()
                except BaseException as error:  # Re-raised on the scoring thread.
                    self._preload_error = error

            self._preload_thread = threading.Thread(
                target=load, name="dino-direction-preload", daemon=True,
            )
            self._preload_thread.start()

    def _wait_for_preload(self):
        thread = self._preload_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        if self._preload_error is not None:
            raise RuntimeError("asynchronous DINO preload failed") from self._preload_error

    def close(self):
        """Release DINO before front refinement and Blender final renders."""
        import gc
        import torch

        thread = self._preload_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._lock:
            model, self._model = self._model, None
        del model
        gc.collect()
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _load(self):
        if self._model is not None:
            return self._model
        import torch

        checkpoint = self.root / "checkpoints/dinov2_vitl14_pretrain.pth"
        roma_root = self.root / "third_party/RoMa_minima"
        for path in (checkpoint, roma_root):
            if not path.exists():
                raise FileNotFoundError(f"DINO direction runtime asset is missing: {path}")
        sys.path.insert(0, str(roma_root))
        from romatch.models.transformer import vit_large

        device = torch.device(self.device)
        model = vit_large(
            img_size=518, patch_size=14, init_values=1.0,
            ffn_layer="mlp", block_chunks=0,
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        if device.type == "cuda":
            model = model.half()
        self._model = model.eval()
        return self._model

    def _feature_batch(self, images):
        import torch
        import torch.nn.functional as functional

        device = torch.device(self.device)
        model = self._load()
        grid_size = self.input_size // 14
        batch = np.stack([np.asarray(image, dtype=np.uint8) for image in images])
        tensor = torch.from_numpy(np.ascontiguousarray(batch)).permute(
            0, 3, 1, 2,
        ).to(device=device, dtype=torch.float32) / 255.0
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=device, dtype=tensor.dtype,
        )[None, :, None, None]
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=device, dtype=tensor.dtype,
        )[None, :, None, None]
        tensor = ((tensor - mean) / std).to(
            dtype=next(model.parameters()).dtype,
        )
        with torch.inference_mode():
            tokens = model.forward_features(tensor)["x_norm_patchtokens"]
        features = tokens.reshape(len(tensor), grid_size, grid_size, -1).float()
        return list(functional.normalize(features, dim=-1).cpu().unbind(0))

    def _features_many(self, images):
        import torch

        batch_size = min(self._batch_size, len(images))
        while True:
            try:
                results = []
                for start in range(0, len(images), batch_size):
                    results.extend(self._feature_batch(
                        images[start:start + batch_size],
                    ))
                self._batch_size = batch_size
                return results
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                next_batch_size = max(1, batch_size // 2)
                print(
                    "[direction loss] DINO CUDA OOM at batch "
                    f"{batch_size}; retrying with {next_batch_size}",
                    flush=True,
                )
                batch_size = next_batch_size
                self._batch_size = batch_size
                torch.cuda.empty_cache()

    @staticmethod
    def _patch_mask(mask, grid_size):
        return cv2.resize(
            np.asarray(mask, dtype=np.uint8), (grid_size, grid_size),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    def score_candidates(
        self, reference_key, reference_rgb, reference_mask, camera, candidates,
    ):
        self._wait_for_preload()
        del reference_key, camera
        if not candidates:
            return {}
        grid_size = self.input_size // 14
        candidate_items = list(candidates.items())
        raw_inputs = [(reference_rgb, reference_mask)] + [
            (image, candidate["mask"])
            for _, candidate in candidate_items
            for image in (candidate["rgb"], candidate["normal_map"])
        ]
        if self.crop_mode == "shared_camera":
            crops = shared_camera_object_crops(raw_inputs, self.input_size)
        else:
            crops = [
                relative_object_crop(rgb, mask, self.input_size)
                for rgb, mask in raw_inputs
            ]
        reference, reference_crop_mask = crops[0]
        reference_patch_mask = self._patch_mask(reference_crop_mask, grid_size)
        prepared = []
        for index, (key, candidate) in enumerate(candidate_items):
            candidate_rgb, candidate_rgb_mask = crops[1 + 2 * index]
            candidate_normal, candidate_normal_mask = crops[2 + 2 * index]
            prepared.append((
                key, candidate, candidate_rgb, candidate_rgb_mask,
                candidate_normal, candidate_normal_mask,
            ))
        with self._lock:
            import torch

            device = torch.device(self.device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            features = self._features_many([
                reference,
                *[
                    image
                    for _, _, rgb, _, normal, _ in prepared
                    for image in (rgb, normal)
                ],
            ])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - start
        reference_features = features[0]
        results = {}
        for index, (
            key, candidate, _, candidate_rgb_mask, _, candidate_normal_mask,
        ) in enumerate(prepared):
            rgb_features, normal_features = features[1 + 2 * index:3 + 2 * index]
            rgb_loss = fixed_grid_cosine_loss(
                reference_features, rgb_features, reference_patch_mask,
                self._patch_mask(candidate_rgb_mask, grid_size),
            )
            normal_loss = fixed_grid_cosine_loss(
                reference_features, normal_features, reference_patch_mask,
                self._patch_mask(candidate_normal_mask, grid_size),
            )
            loss = 0.5 * (rgb_loss + normal_loss)
            results[key] = {
                "loss": float(loss),
                "rgb_fixed_grid_loss": float(rgb_loss),
                "normal_fixed_grid_loss": float(normal_loss),
                "foreground_mean_confidence": float(1.0 - loss),
                "batched_inference_seconds": float(inference_seconds),
                "batch_image_count": len(features),
                "inference_batch_size": self._batch_size,
                "input_size": self.input_size,
                "crop_mode": self.crop_mode,
                "patch_grid": [grid_size, grid_size],
                "backend": self.backend,
                "reference_modality": "real_rgb",
                "candidate_modality": "rendered_rgb_and_camera_normal",
                "rendered_rgb_source": candidate["candidate_modality"],
                "selection_source": f"fixed_grid_dino_rgb_normal_{self.crop_mode}",
            }
        return results


class DinoRelativePositionDirectionScorer(DinoGridDirectionScorer):
    """Rank candidates by DINO match positions in the unchanged reference frame."""

    backend = "dino_v2_reference_frame_relative_position_normal_weighted"
    loss_formula = (
        "mean_bidirectional_distance(DINO(real_RGB),DINO(rendered_RGB),"
        "weight=real_RGB_structure*rendered_normal_structure)"
    )
    stage3_uses_stage2_prior = False

    def __init__(self, root, device, input_size=728):
        super().__init__(
            root, device=device, input_size=input_size, crop_mode="shared_camera",
        )

    def score_candidates(
        self, reference_key, reference_rgb, reference_mask, camera, candidates,
    ):
        return self.score_candidate_groups({None: {
            "reference_key": reference_key,
            "reference_rgb": reference_rgb,
            "reference_mask": reference_mask,
            "camera": camera,
            "candidates": candidates,
        }})[None]

    def score_candidate_groups(self, groups):
        """Score multiple object references with one chunked DINO pass."""
        self._wait_for_preload()
        grid_size = self.input_size // 14
        prepared_groups = []
        feature_images = []
        for group_key, group in groups.items():
            candidate_items = list(group["candidates"].items())
            if not candidate_items:
                prepared_groups.append((group_key, group, None, []))
                continue
            reference_mask = group["reference_mask"]
            raw_inputs = [(group["reference_rgb"], reference_mask)] + [
                (image, candidate["mask"])
                for _, candidate in candidate_items
                for image in (candidate["rgb"], candidate["normal_map"])
            ]
            crops = reference_camera_object_crops(
                raw_inputs, reference_mask, self.input_size,
            )
            reference, reference_crop_mask = crops[0]
            reference_feature_index = len(feature_images)
            feature_images.append(reference)
            prepared = []
            for index, (key, candidate) in enumerate(candidate_items):
                candidate_rgb, candidate_crop_mask = crops[1 + 2 * index]
                candidate_normal, _ = crops[2 + 2 * index]
                feature_index = len(feature_images)
                feature_images.append(candidate_rgb)
                prepared.append((
                    key, candidate, candidate_normal, candidate_crop_mask,
                    feature_index,
                ))
            prepared_groups.append((
                group_key, group, (reference, reference_crop_mask,
                reference_feature_index), prepared,
            ))
        if not feature_images:
            return {group_key: {} for group_key in groups}
        with self._lock:
            import torch

            torch_device = torch.device(self.device)
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            start = time.perf_counter()
            features = self._features_many(feature_images)
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            inference_seconds = time.perf_counter() - start
        pair_keys = []
        pairs = []
        for group_key, group, reference_data, prepared in prepared_groups:
            if reference_data is None:
                continue
            reference, reference_crop_mask, reference_feature_index = reference_data
            reference_patch_mask = self._patch_mask(
                reference_crop_mask, grid_size,
            )
            reference_weights = _structure_weights(
                reference, reference_crop_mask, grid_size,
            )
            for (
                key, candidate, candidate_normal, candidate_crop_mask,
                feature_index,
            ) in prepared:
                candidate_patch_mask = self._patch_mask(
                    candidate_crop_mask, grid_size,
                )
                candidate_weights = _structure_weights(
                    candidate_normal, candidate_crop_mask, grid_size,
                    normal_map=True,
                )
                pair_keys.append((group_key, key, candidate))
                pairs.append((
                    features[reference_feature_index], features[feature_index],
                    reference_patch_mask, candidate_patch_mask,
                    reference_weights, candidate_weights,
                ))
        pair_scores = dense_relative_position_loss_batch(
            pairs, device=self.device,
        )
        grouped_results = {group_key: {} for group_key in groups}
        for (group_key, key, candidate), (loss, details) in zip(
            pair_keys, pair_scores,
        ):
            grouped_results[group_key][key] = {
                "loss": loss,
                "rgb_relative_position_loss": loss,
                "normal_structure_weighted_loss": loss,
                **details,
                "foreground_mean_confidence": float(max(0.0, 1.0 - loss)),
                "batched_inference_seconds": float(inference_seconds),
                "batch_image_count": len(features),
                "inference_batch_size": self._batch_size,
                "input_size": self.input_size,
                "patch_grid": [grid_size, grid_size],
                "crop_mode": "reference_bbox_shared_camera_coordinates",
                "backend": self.backend,
                "reference_view": group["reference_key"],
                "reference_modality": "real_rgb",
                "candidate_modality": "rendered_rgb",
                "weight_modality": "real_rgb_and_rendered_camera_normal_structure",
                "rendered_rgb_source": candidate["candidate_modality"],
                "selection_source": self.backend,
            }
        return grouped_results
